"""Keep a module's CPU-side weight tensors across a VRAM load so an unload restores them instead of copying back.

ComfyUI moves a model to the GPU with ``Module.to(device)``, which replaces each parameter's data with a
device copy and drops the CPU tensor it came from; unloading then runs ``Module.to(offload_device)``, a
device-to-host copy of every weight into freshly allocated private memory. On a 5 GB UNet that copy holds the
card for over a second on every eviction, and it turns a checkpoint that was a shared, page-cache-backed
mapping (see ``zero_copy_load``) into a private copy per process, so a model cached in RAM by several
inference processes is duplicated once per process.

Neither cost is necessary when the device copy was never written to. This module records each CPU tensor
just before the load replaces it and, at unload, points the parameter back at that tensor; the device
allocation is released as the last reference to it goes and the ``.to(cpu)`` that follows is a no-op. Only
weights ComfyUI did not patch are restored this way: a LoRA-patched key is set to a fresh patched tensor by
the patcher and restored from the patcher's own backup, so those are left to it. The retained tensor is
byte-identical to what a copy-back would have produced, so nothing downstream can tell the difference
except the missing pause and the private-memory growth that no longer happens.

A partial unload (ComfyUI trimming a loaded model to make room) moves modules back with ``Module.to``, which
is the same copy into private memory, and it pins the copy. ``release_copied_back_weights`` runs after it
and points each unpatched weight back at its origin, so the copy is freed and a later load restores to the
mapping again rather than to the copy.

The kill switch ``HORDELIB_DISABLE_CPU_WEIGHT_RETENTION`` restores ComfyUI's copy-back behaviour.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

import torch
from loguru import logger

_DISABLE_ENV_VAR = "HORDELIB_DISABLE_CPU_WEIGHT_RETENTION"
_TRUTHY_VALUES = {"1", "true", "yes", "on"}
_ORIGIN_ATTR = "_hordelib_cpu_origin"


def cpu_weight_retention_disabled() -> bool:
    """Return whether the kill-switch env var disables CPU weight retention (default: enabled)."""
    return os.environ.get(_DISABLE_ENV_VAR, "").strip().lower() in _TRUTHY_VALUES


def _named_weights(module: torch.nn.Module) -> list[tuple[str, torch.Tensor]]:
    return [
        *((name, param) for name, param in module.named_parameters(recurse=True)),
        *((name, buf) for name, buf in module.named_buffers(recurse=True)),
    ]


def stash_cpu_origins(module: torch.nn.Module) -> int:
    """Record the current CPU tensor of every CPU-resident weight and buffer; returns how many were recorded.

    A weight already carrying a record keeps it (its data has not moved since), so repeated loads of a resident
    model cost one attribute read per weight.
    """
    if cpu_weight_retention_disabled():
        return 0
    recorded = 0
    for _, tensor in _named_weights(module):
        data = tensor.data
        if data.device.type != "cpu" or data.is_meta:
            continue
        existing = getattr(tensor, _ORIGIN_ATTR, None)
        if existing is not None and existing.data_ptr() == data.data_ptr():
            continue
        try:
            setattr(tensor, _ORIGIN_ATTR, data)
        except AttributeError:
            continue
        recorded += 1
    return recorded


def carry_cpu_origin(previous: torch.Tensor, replacement: torch.Tensor) -> bool:
    """Copy ``previous``'s recorded CPU origin onto ``replacement``; returns whether a record was carried.

    ``Module._apply`` keeps each Parameter object and swaps its data, so the record survives a move. ComfyUI's
    quantized ops register a new Parameter per move instead, and without the carry the record is lost and the
    next unload copies the weight back into private memory. Only a plain ``Parameter`` takes the record: the
    restore assigns ``.data``, which on a tensor subclass wrapper such as ``QuantizedTensor`` replaces the outer
    tensor's metadata and leaves the wrapped payload where it was.
    """
    origin: Any = getattr(previous, _ORIGIN_ATTR, None)
    if origin is None or type(replacement) is not torch.nn.Parameter:
        return False
    try:
        setattr(replacement, _ORIGIN_ATTR, origin)
    except AttributeError:
        return False
    return True


def restore_cpu_origins(
    module: torch.nn.Module,
    *,
    skip_keys: set[str] | frozenset[str] | None = None,
) -> tuple[int, int]:
    """Point every device-resident weight with a recorded CPU origin back at that origin.

    ``skip_keys`` names weights (by their ``named_parameters``/``named_buffers`` name) that must be left for the
    caller to restore, typically the keys ComfyUI patched and backs up itself. Returns ``(restored, bytes)``.
    """
    if cpu_weight_retention_disabled():
        return 0, 0
    restored = 0
    restored_bytes = 0
    for name, tensor in _named_weights(module):
        if skip_keys is not None and name in skip_keys:
            continue
        origin: Any = getattr(tensor, _ORIGIN_ATTR, None)
        if origin is None:
            continue
        data = tensor.data
        if data.device.type == "cpu":
            continue
        if origin.shape != data.shape or origin.dtype != data.dtype:
            # The device tensor is not the one the origin was recorded for; leave it to the copy-back.
            continue
        restored_bytes += data.numel() * data.element_size()
        tensor.data = origin
        restored += 1
    if restored:
        logger.debug(
            "Restored {} CPU-origin weight(s) ({} MB) instead of copying back from the device",
            restored,
            restored_bytes // (1024 * 1024),
        )
    return restored, restored_bytes


def release_copied_back_weights(
    module: torch.nn.Module,
    *,
    skip_keys: set[str] | frozenset[str] | None = None,
    unpin: Callable[[str], None] | None = None,
) -> tuple[int, int]:
    """Point every CPU weight that is a private copy of its recorded origin back at the origin.

    Runs after a copy-back has already happened (ComfyUI's partial unload), so the copy is released rather
    than avoided. ``unpin`` is called with each weight's key before its copy is dropped, since the copy may be
    registered as pinned host memory and a registration must not outlive its allocation. ``skip_keys`` is as
    for :func:`restore_cpu_origins`. Returns ``(released, bytes)``.
    """
    if cpu_weight_retention_disabled():
        return 0, 0
    released = 0
    released_bytes = 0
    for name, tensor in _named_weights(module):
        if skip_keys is not None and name in skip_keys:
            continue
        origin: Any = getattr(tensor, _ORIGIN_ATTR, None)
        if origin is None:
            continue
        data = tensor.data
        if data.device.type != "cpu" or data.data_ptr() == origin.data_ptr():
            continue
        if origin.shape != data.shape or origin.dtype != data.dtype:
            continue
        if unpin is not None:
            unpin(name)
        released_bytes += data.numel() * data.element_size()
        tensor.data = origin
        released += 1
    if released:
        logger.debug(
            "Released {} copied-back weight(s) ({} MB) to their CPU origins after a partial unload",
            released,
            released_bytes // (1024 * 1024),
        )
    return released, released_bytes
