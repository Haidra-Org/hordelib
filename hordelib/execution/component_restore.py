"""Return a cached model component to the state it was loaded in, on demand.

The component cache hands out loaded components by reference, so a job that patches one is patching the
object the next job will be given. Correctness on the load path does not depend on this module: comfy
restores a component lazily at its next load, unpatching and re-uploading whenever the patch set the
module's weights hold differs from the loading patcher's. What this module provides is the explicit
lever for doing it early, which is how a resident component's device memory is given back without
evicting its weights from host RAM, plus :func:`has_patch_residue` for reporting which components are
currently carrying another job's patches.

Two kinds of state need returning:

**Patch residue.** ComfyUI's ``ModelPatcher.clone`` shares both the underlying ``nn.Module`` and the
``backup`` dict by reference, so a LoRA clone patches the same weights the cached base holds.
``unpatch_model`` is comfy's own complete restore (weights, the ``weight_function`` and
``bias_function`` lists via ``wipe_lowvram_weight``, ``model_lowvram``, ``comfy_patched_weights``, and
``current_weight_patches_uuid``), so restoration calls it rather than reimplementing it. It is called
with the patcher's offload device, matching comfy's own detach, so the weights end up where the
weight-memory accounting it resets says they are. Restoring therefore returns a component to host RAM,
which is where this cache's budget already accounts for it, and costs a re-upload at its next use.

**Seamless tiling.** Tiling retunes ``Conv2d.padding_mode`` directly on the module, with no patcher and
no backup. Resetting it means knowing what the padding was to begin with, which is why
:func:`capture_pristine_state` records the as-constructed mode per convolution at load time. Resetting
to a hardcoded ``"zeros"`` is only correct for architectures built that way: the pinned tree contains
convolutions constructed with ``"replicate"`` (the Genmo/Mochi and ACE VAEs), for which a blanket reset
would silently change decode semantics.

Dispatch is on the object rather than on the cache key's kind, because one kind can arrive in more than
one payload shape (a bare VAE component sits at slot 0, a standalone VAE at slot 2) and because a
component type this module has never seen should degrade to doing nothing rather than guessing.
"""

from __future__ import annotations

from typing import Any

import torch
from loguru import logger

__all__ = [
    "capture_pristine_state",
    "has_patch_residue",
    "restore_component",
    "restore_conv2d_padding",
    "restore_payload",
    "restore_pristine_padding",
]

_PRISTINE_PADDING_ATTR = "_hordelib_pristine_padding_mode"


def _root_modules(component: Any) -> list[torch.nn.Module]:
    """Return the ``nn.Module`` roots reachable from *component* whose convolutions tiling can retune.

    A ``ModelPatcher`` exposes its module as ``model``; a comfy VAE exposes ``first_stage_model`` (which
    is ``None`` on architectures where tiling does not apply); a bare module is its own root.
    """
    roots: list[torch.nn.Module] = []
    for attribute in ("model", "first_stage_model"):
        candidate = getattr(component, attribute, None)
        if isinstance(candidate, torch.nn.Module):
            roots.append(candidate)
    if not roots and isinstance(component, torch.nn.Module):
        roots.append(component)
    return roots


def _patchers(component: Any) -> list[Any]:
    """Return the patcher-shaped objects reachable from *component*.

    A ``ModelPatcher`` is its own patcher. A comfy ``CLIP`` (and, on some architectures, a ``VAE``)
    wraps one as ``patcher``, and LoRA patches the text encoder through exactly that.
    """
    found = []
    for candidate in (component, getattr(component, "patcher", None)):
        if candidate is not None and callable(getattr(candidate, "unpatch_model", None)):
            found.append(candidate)
    return found


def has_patch_residue(payload: Any) -> bool:
    """Return whether any component in *payload* currently holds patches from another patcher.

    Comfy records the patch set a module's weights were written with in the module's
    ``current_weight_patches_uuid``; a patcher whose own ``patches_uuid`` differs from it is holding
    weights some other patcher (a LoRA clone) wrote. A module with nothing applied records ``None`` and
    is clean.

    Accepts the same payload shapes as :func:`restore_payload`. Attribute reads are defensive and the
    whole walk is guarded, so an unfamiliar shape answers False rather than raising into a residency
    report.
    """
    if payload is None:
        return False
    slots = payload if isinstance(payload, (tuple, list)) else (payload,)
    try:
        for slot in slots:
            if slot is None:
                continue
            for patcher in _patchers(slot):
                applied = getattr(getattr(patcher, "model", None), "current_weight_patches_uuid", None)
                if applied is not None and applied != getattr(patcher, "patches_uuid", None):
                    return True
    except Exception as probe_error:
        logger.debug(
            "Could not probe a component for patch residue: {} {}",
            type(probe_error).__name__,
            probe_error,
        )
    return False


def capture_pristine_state(component: Any) -> None:
    """Record *component*'s as-constructed convolution padding, so a later reset returns it exactly.

    Idempotent and safe to call on an already-captured component: a convolution that already carries a
    recorded mode keeps it, so calling this after tiling has been applied cannot record the tiled state
    as though it were the original. Call it on a freshly loaded component before any tiling is applied.
    """
    if component is None:
        return
    for root in _root_modules(component):
        for module in root.modules():
            if isinstance(module, torch.nn.Conv2d) and not hasattr(module, _PRISTINE_PADDING_ATTR):
                setattr(module, _PRISTINE_PADDING_ATTR, module.padding_mode)


def restore_conv2d_padding(module: torch.nn.Module) -> None:
    """Reset one convolution's padding to what :func:`capture_pristine_state` recorded for it.

    A convolution with no recorded mode is left alone: it was never captured, so its current mode is the
    best available account of its original and overwriting it would be the very guess this avoids. Takes
    a single module so it can be used as an ``nn.Module.apply`` visitor.
    """
    if not isinstance(module, torch.nn.Conv2d):
        return
    original = getattr(module, _PRISTINE_PADDING_ATTR, None)
    if original is not None:
        module.padding_mode = original


def restore_pristine_padding(component: Any) -> None:
    """Reset every convolution in *component* to the padding it was constructed with."""
    if component is None:
        return
    for root in _root_modules(component):
        root.apply(restore_conv2d_padding)


def restore_component(component: Any) -> int:
    """Return one component to its loaded state; report the device bytes its weights gave up.

    The count comes from the patcher's ``model_loaded_weight_memory`` read before unpatching, which is
    the figure ``unpatch_model`` zeroes, so it is exactly the weight memory the component stops claiming.
    A component with nothing loaded returns zero, which is also the falsey answer for callers that only
    want to know whether anything happened.

    Never raises. A component that cannot be restored is logged at debug and reported as zero, so a
    reclaim request degrades to reclaiming less rather than faulting.
    """
    if component is None:
        return 0

    released = 0
    for patcher in _patchers(component):
        try:
            claimed = int(getattr(patcher.model, "model_loaded_weight_memory", 0) or 0)
            # The patcher's own offload device, which is what comfy's detach and partially_load both
            # pass. Passing None instead would leave the weights wherever they are (possibly VRAM) while
            # unpatch_model still zeroes model_loaded_weight_memory, so comfy's accounting would claim
            # nothing is resident while the card still held it. A restored component must be resident
            # where the accounting says it is, because the worker sizes its reclaim decisions on that.
            patcher.unpatch_model(device_to=patcher.offload_device, unpatch_weights=True)
            # Counted only after the unpatch returns, so a patcher that raised partway is not reported
            # as having given up memory it may still hold.
            released += claimed
        except Exception as restore_error:
            logger.debug(
                "Could not unpatch a cached component: {} {}",
                type(restore_error).__name__,
                restore_error,
            )

    try:
        restore_pristine_padding(component)
    except Exception as padding_error:
        logger.debug(
            "Could not reset cached component padding: {} {}",
            type(padding_error).__name__,
            padding_error,
        )

    return released


def restore_payload(payload: Any) -> int:
    """Restore every component in a cached payload; report the device bytes the payload gave up.

    Payload shapes differ by load path (``(model, clip, vae, clipvision)`` for a checkpoint,
    ``(component, None, None)`` for a bare component, ``(None, None, vae, None)`` for a standalone VAE),
    so every slot is offered to :func:`restore_component` and the empty ones fall through. Summing rather
    than reducing to a flag also means every slot is visited: stopping at the first slot that reported
    work would leave the rest of the payload patched.
    """
    if payload is None:
        return 0
    if not isinstance(payload, (tuple, list)):
        return restore_component(payload)
    return sum(restore_component(slot) for slot in payload)
