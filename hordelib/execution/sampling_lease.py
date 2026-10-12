"""Cross-process GPU sampling lease.

A single GPU can only meaningfully run one denoising loop at a time, but the *rest* of a
job's pipeline — loading the model into VRAM, encoding the prompt, building latents — is
GPU-light and can run while another process samples. Left uncoordinated, multiple worker
inference processes tend to fall into lockstep: they all sample together, then all do their
setup/decode together, leaving the GPU idle during the shared setup window.

This module lets the host (e.g. the AI-Horde worker) inject a cross-process semaphore that
hordelib acquires around ``comfy.samplers.CFGGuider.sample``. Every ComfyUI sampling path reaches that
method: ``KSampler``-family nodes through ``comfy.sample.sample``, ``SamplerCustom`` through
``comfy.sample.sample_custom``, and ``SamplerCustomAdvanced`` by calling ``guider.sample`` on a
``CFGGuider`` subclass directly. Hooking either ``comfy.sample`` entry point would leave the last path
sampling outside the lease. ``CFGGuider.sample`` brackets the diffusion model's load-to-VRAM
(``prepare_sampling`` in ``outer_sample``) *and* the denoise loop, but not the upstream checkpoint
disk load or prompt encoding, nor the downstream VAE decode. With more inference processes than lease
slots, one process samples while the others stage their next pipeline (checkpoint load, prompt encode)
up to the sampling call; when the sampler finishes, an already-staged process takes the lease and
samples immediately, keeping the GPU busy back-to-back. Because the VRAM load sits inside the lease,
this only helps when models are kept resident (the host's high-memory / no-unload mode); otherwise it
serializes the RAM->VRAM transfer behind sampling.

A host that also sets a stage lease (``set_gpu_stage_lease``) splits that window in two. The hook takes the
stage lease first and, on its grant, uploads the guider's model with the call ``prepare_sampling`` makes, then
releases it and takes the sampling lease. The later ``prepare_sampling`` finds the model loaded, so the sampling
lease covers only the denoise loop and its onset. The stage permit covers that upload alone. A host never grants a
stage permit without a sampling permit following, and both acquires share one deadline of the acquire timeout, so
the longest wait is unchanged. With no stage lease set the path is the single acquire above.

A guider subclass that overrides ``sample`` without calling ``super().sample`` bypasses the lease. A
nested hooked call on the same thread passes through without a second acquire.

When no lease is set (standalone hordelib, or the feature disabled) the wrapper is a no-op.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from time import monotonic
from typing import Protocol, runtime_checkable

from loguru import logger


@runtime_checkable
class SamplingLease(Protocol):
    """The subset of ``multiprocessing.Semaphore`` / ``threading.Semaphore`` we rely on."""

    def acquire(self, block: bool = ..., timeout: float | None = ...) -> bool:
        """Acquire the lease, optionally with a timeout; returns whether it was acquired."""
        ...

    def release(self) -> None:
        """Release the lease."""
        ...


@runtime_checkable
class ClearanceTruthLease(Protocol):
    """An optional extension a lease implements when its grant carries the host's device-free figure.

    A host that grants the lease after measuring the card has a fresher reading than the one the run's
    free-VRAM clamp was entered with. After a successful acquire the hook asks for it and rebases the
    clamp. A lease without this method, or one that returns None, leaves the clamp as it was.
    """

    def take_cleared_device_free_mb(self) -> float | None:
        """Return the device free VRAM (MB) the grant just acquired was admitted against, once, or None."""
        ...


_sampling_lease: SamplingLease | None = None
_stage_lease: SamplingLease | None = None
_acquire_timeout_seconds: float = 120.0
_installed: bool = False
_lease_scope: threading.local = threading.local()
"""Per-thread marker that a hooked guider sample is already running, so a nested one does not acquire again."""


def set_gpu_sampling_lease(lease: SamplingLease | None, *, acquire_timeout_seconds: float = 120.0) -> None:
    """Register (or clear) the cross-process lease hordelib holds around the sampling loop.

    Args:
        lease: A semaphore-like object shared across the host's inference processes, or None
            to disable coordination.
        acquire_timeout_seconds: Safety cap on the lease wait. If acquisition times out the
            job samples anyway (degraded coordination beats a hung worker).
    """
    global _sampling_lease, _acquire_timeout_seconds
    _sampling_lease = lease
    _acquire_timeout_seconds = acquire_timeout_seconds
    if lease is not None:
        logger.info(f"GPU sampling lease registered (acquire timeout {acquire_timeout_seconds:.0f}s)")


def set_gpu_stage_lease(lease: SamplingLease | None) -> None:
    """Register (or clear) the lease hordelib holds while it uploads the sampling model ahead of the sampling lease.

    The stage acquire shares the sampling lease's acquire timeout: both are taken within one deadline. The hook stages
    only while a sampling lease is also set.

    Args:
        lease: A semaphore-like object the host grants when the model's weights fit on the device, or None to load
            the model inside the sampling lease.
    """
    global _stage_lease
    _stage_lease = lease
    if lease is not None:
        logger.info("GPU stage lease registered (shares the sampling lease acquire timeout)")


def _prefetch_sampling_model_weights(guider: object) -> None:
    """Kick a background page-in of the guider's CPU-resident diffusion weights; never raises."""
    try:
        model_patcher = getattr(guider, "model_patcher", None)
        module = getattr(model_patcher, "model", None)
        if module is None:
            return
        from hordelib.execution.weight_prefetch import prefetch_module_weights_async

        prefetch_module_weights_async(module, label="sample")
    except Exception as exc:
        logger.debug(f"Sampling weight prefetch skipped: {exc}")


def _rebase_clamp_on_grant(lease: SamplingLease) -> None:
    """Rebase the run's free-VRAM clamp on the figure an acquired grant carries; never raises."""
    if not isinstance(lease, ClearanceTruthLease):
        return
    try:
        device_free_mb = lease.take_cleared_device_free_mb()
        if device_free_mb is None:
            return
        from hordelib.execution.comfy_patches import rebase_free_memory_view_clamp

        rebase_free_memory_view_clamp(device_free_mb)
    except Exception as exc:
        logger.warning(f"Could not rebase the free-VRAM clamp on the lease grant ({exc}); keeping the run's figure")


def install_sampling_lease_hook() -> bool:
    """Monkey-patch ``comfy.samplers.CFGGuider.sample`` to hold the lease around the model load and denoise loop.

    The patch replaces the base class attribute, so every guider subclass that inherits ``sample`` (or reaches it
    through ``super().sample``) is covered. Idempotent and independent of logfire instrumentation.

    Returns:
        True if the hook is in place (now or already), False if ComfyUI's samplers module could not be patched.
    """
    global _installed
    if _installed:
        return True

    try:
        from comfy import samplers as samplers_module
    except Exception as e:
        logger.warning(f"Could not install GPU sampling lease hook: {e}")
        return False

    guider_class = samplers_module.CFGGuider
    original_guider_sample = guider_class.sample

    def _leased_guider_sample(self: object, *args: object, **kwargs: object) -> object:
        # A guider whose sampling nests another hooked call already holds the lease (or already gave up waiting
        # for it); a second acquire would wait on its own grant.
        if getattr(_lease_scope, "active", False):
            return original_guider_sample(self, *args, **kwargs)
        _lease_scope.active = True
        try:
            return _sample_under_lease(self, original_guider_sample, args, kwargs)
        finally:
            _lease_scope.active = False

    guider_class.sample = _leased_guider_sample
    _installed = True
    logger.debug("Installed GPU sampling lease hook on comfy.samplers.CFGGuider.sample")
    return True


def _sample_under_lease(
    guider: object,
    original_guider_sample: Callable[..., object],
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> object:
    """Run the original guider sample inside the lease, if one is set; never blocks past the acquire timeout."""
    # The model's RAM->VRAM move runs inside the lease and pages its weights in from the checkpoint mapping on
    # first touch; asking the OS to prefetch them now lets that disk read overlap the wait for clearance instead
    # of sitting in the sampling window.
    _prefetch_sampling_model_weights(guider)
    lease = _sampling_lease
    stage_lease = _stage_lease
    acquired = False
    if lease is not None:
        try:
            if stage_lease is None:
                acquired = bool(lease.acquire(True, _acquire_timeout_seconds))
            else:
                acquired = _stage_then_acquire_sampling(guider, lease, stage_lease, args, kwargs)
        except Exception as e:
            logger.warning(f"GPU sampling lease acquire failed ({e}); sampling without it")
            acquired = False
        if acquired:
            _rebase_clamp_on_grant(lease)
        else:
            logger.warning("GPU sampling lease acquire timed out; sampling without it")
    try:
        return original_guider_sample(guider, *args, **kwargs)
    finally:
        if acquired and lease is not None:
            try:
                lease.release()
            except Exception as e:
                logger.error(f"Failed to release GPU sampling lease: {e}")


def _stage_then_acquire_sampling(
    guider: object,
    lease: SamplingLease,
    stage_lease: SamplingLease,
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> bool:
    """Upload the guider's model under the stage lease, then acquire the sampling lease within the same deadline.

    Returns:
        Whether the sampling lease was acquired. With nothing left of the deadline the acquire is a non-blocking try.
    """
    deadline = monotonic() + _acquire_timeout_seconds
    _stage_under_lease(guider, stage_lease, deadline, args, kwargs)
    remaining_seconds = deadline - monotonic()
    if remaining_seconds <= 0:
        return bool(lease.acquire(False))
    return bool(lease.acquire(True, remaining_seconds))


def _stage_under_lease(
    guider: object,
    stage_lease: SamplingLease,
    deadline: float,
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> None:
    """Acquire the stage lease up to the deadline and, on grant, upload the guider's model; never raises."""
    try:
        stage_granted = bool(stage_lease.acquire(True, max(0.0, deadline - monotonic())))
    except Exception as e:
        logger.warning(f"GPU stage lease acquire failed ({e}); the model loads at sampling")
        return
    if not stage_granted:
        logger.warning("GPU stage lease acquire timed out; the model loads at sampling")
        return
    try:
        _rebase_clamp_on_grant(stage_lease)
        # ``CFGGuider.sample(noise, latent_image, ...)`` keeps its conditioning in ``original_conds``; ``conds`` is a
        # per-call copy that exists only while ``sample`` runs.
        noise = args[0] if args else kwargs.get("noise")
        _stage_sampling_model(
            guider,
            noise,
            getattr(guider, "original_conds", None),
            getattr(guider, "model_options", None),
        )
    finally:
        try:
            stage_lease.release()
        except Exception as e:
            logger.error(f"Failed to release GPU stage lease: {e}")


def _stage_sampling_model(guider: object, noise: object, conds: object, model_options: object) -> None:
    """Upload the guider's model to the device with the call ``prepare_sampling`` makes, ahead of it; never raises.

    The model list, memory estimate and keyword arguments match ``comfy.sampler_helpers._prepare_sampling``, so the
    sampling-time ``prepare_sampling`` finds the same models loaded and moves nothing, and the ``load_models_gpu``
    hijacks apply as they do there. A failure leaves the load to sampling. Nested noise is packed by ``sample``
    before ``prepare_sampling`` reads its shape, so its estimate would differ here; it is not staged.
    """
    try:
        model_patcher = getattr(guider, "model_patcher", None)
        noise_shape = getattr(noise, "shape", None)
        if model_patcher is None or noise_shape is None or not isinstance(conds, dict):
            logger.debug("Early sampling model upload skipped: no model patcher, noise or conditioning")
            return
        if getattr(noise, "is_nested", False):
            logger.debug("Early sampling model upload skipped: nested noise is packed only inside sample")
            return

        import comfy.model_management
        from comfy.sampler_helpers import (
            estimate_memory,
            get_additional_models,
            get_additional_models_from_model_options,
        )

        additional_models, inference_memory = get_additional_models(conds, model_patcher.model_dtype())
        additional_models += get_additional_models_from_model_options(model_options)
        additional_models += model_patcher.get_nested_additional_models()
        memory_required, minimum_memory_required = estimate_memory(model_patcher, noise_shape, conds)
        comfy.model_management.load_models_gpu(
            [model_patcher] + additional_models,
            memory_required=memory_required + inference_memory,
            minimum_memory_required=minimum_memory_required + inference_memory,
            force_full_load=False,
        )
    except Exception as e:
        logger.warning(f"Early sampling model upload failed ({type(e).__name__}: {e}); the model loads at sampling")
