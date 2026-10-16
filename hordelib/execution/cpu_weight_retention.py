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

A partial unload (ComfyUI trimming a loaded model to make room) moves modules back one at a time with
``Module.to`` and pins each moved weight. ComfyUI's plain op classes move through
``comfy_patches._plain_op_apply_hijack``, which inside ``restoring_cpu_origins`` points each unpatched device weight
at its origin before torch's move runs (``restore_plain_origins``). The move finds the weight on the CPU and makes no
copy, and ComfyUI pins the origin. ``release_copied_back_weights`` runs after the partial unload for the weights whose
moves still copy, such as those of modules that are not comfy op classes, and points each copy back at its origin so
the copy is freed.

A tensor-subclass weight (a comfy_kitchen ``QuantizedTensor``, as in fp8 checkpoints) is kept differently.
ComfyUI's quantized ops register a new ``Parameter`` on every move, and assigning ``.data`` on a wrapper
replaces only its outer metadata, so neither the attribute record nor the ``.data`` restore works for it.
``stash_cpu_origins`` keeps the CPU ``Parameter`` object itself in a table on its owning module, keyed by the
parameter's name, and the quantized move (``comfy_patches._quantized_apply_hijack``) registers that object
again when a move back to the CPU runs inside ``restoring_cpu_origins``. The device-to-host copy is never
made, so a host short of commit charge cannot fail it.

The kill switch ``HORDELIB_DISABLE_CPU_WEIGHT_RETENTION`` restores ComfyUI's copy-back behaviour.
"""

from __future__ import annotations

import contextlib
import contextvars
import os
import weakref
from collections.abc import Callable, Iterator
from typing import Any

import torch
from loguru import logger

_DISABLE_ENV_VAR = "HORDELIB_DISABLE_CPU_WEIGHT_RETENTION"
_TRUTHY_VALUES = {"1", "true", "yes", "on"}
_ORIGIN_ATTR = "_hordelib_cpu_origin"
_QUANTIZED_ORIGINS_ATTR = "_hordelib_quantized_cpu_origins"


def cpu_weight_retention_disabled() -> bool:
    """Return whether the kill-switch env var disables CPU weight retention (default: enabled)."""
    return os.environ.get(_DISABLE_ENV_VAR, "").strip().lower() in _TRUTHY_VALUES


def _is_tensor_subclass(tensor: torch.Tensor) -> bool:
    """Return whether ``tensor`` is a wrapper subclass, whose ``Parameter`` wrap keeps the subclass type."""
    return type(tensor) not in (torch.Tensor, torch.nn.Parameter)


def _named_weights(module: torch.nn.Module) -> list[tuple[str, torch.Tensor]]:
    return [
        *((name, param) for name, param in module.named_parameters(recurse=True) if not _is_tensor_subclass(param)),
        *((name, buf) for name, buf in module.named_buffers(recurse=True)),
    ]


class QuantizedOrigin:
    """Represents the CPU ``Parameter`` a tensor-subclass weight held before a load, and the device copy of it.

    The device ``Parameter`` is held by weak reference. A patched weight replaces it, and a strong reference
    would keep the replaced device allocation alive.
    """

    __slots__ = ("_device_twin", "parameter")

    def __init__(self, parameter: torch.nn.Parameter) -> None:
        """Record ``parameter`` as the CPU origin, with no device copy made from it yet."""
        self.parameter: torch.nn.Parameter = parameter
        self._device_twin: weakref.ref[torch.Tensor] | None = None

    def note_device_twin(self, twin: torch.Tensor) -> None:
        """Mutate the record so ``twin`` is the device ``Parameter`` a move made from the origin."""
        self._device_twin = weakref.ref(twin)

    def is_device_twin(self, candidate: torch.Tensor) -> bool:
        """Return whether ``candidate`` is the unmodified device ``Parameter`` a move made from the origin.

        A LoRA patch registers a different ``Parameter``, so a patched weight never matches.
        """
        return self._device_twin is not None and self._device_twin() is candidate


def _quantized_origins(module: torch.nn.Module) -> dict[str, QuantizedOrigin] | None:
    table: Any = module.__dict__.get(_QUANTIZED_ORIGINS_ATTR)
    return table


def quantized_origin_parameter(module: torch.nn.Module, key: str) -> torch.nn.Parameter | None:
    """Return the CPU ``Parameter`` recorded as the origin of ``module``'s parameter ``key``, or ``None``."""
    table = _quantized_origins(module)
    entry = table.get(key) if table is not None else None
    return entry.parameter if entry is not None else None


def _stash_quantized_origins(module: torch.nn.Module) -> int:
    """Record each CPU tensor-subclass parameter in its owning module's table and return how many were recorded."""
    recorded = 0
    for owner in module.modules():
        for key, param in owner.named_parameters(recurse=False):
            if not _is_tensor_subclass(param) or param.device.type != "cpu":
                continue
            table = _quantized_origins(owner)
            if table is None:
                table = {}
                owner.__dict__[_QUANTIZED_ORIGINS_ATTR] = table
            existing = table.get(key)
            if existing is not None and existing.parameter is param:
                continue
            table[key] = QuantizedOrigin(param)
            recorded += 1
    return recorded


def stash_cpu_origins(module: torch.nn.Module) -> int:
    """Record the current CPU tensor of every CPU-resident weight and buffer; returns how many were recorded.

    A weight already carrying a record keeps it (its data has not moved since), so repeated loads of a resident
    model cost one attribute read per weight. Tensor-subclass parameters are recorded as ``Parameter`` objects
    in a table on their owning module, since their moves replace the ``Parameter``.
    """
    if cpu_weight_retention_disabled():
        return 0
    recorded = _stash_quantized_origins(module)
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
    next unload copies the weight back into private memory. Only a plain ``Parameter`` takes the record, since
    the restore assigns ``.data``. A tensor subclass such as ``QuantizedTensor`` is restored from its module's
    table instead (see :func:`note_quantized_device_twin`).
    """
    origin: Any = getattr(previous, _ORIGIN_ATTR, None)
    if origin is None or type(replacement) is not torch.nn.Parameter:
        return False
    try:
        setattr(replacement, _ORIGIN_ATTR, origin)
    except AttributeError:
        return False
    return True


def note_quantized_device_twin(
    module: torch.nn.Module,
    *,
    key: str,
    previous: torch.Tensor,
    replacement: torch.Tensor,
) -> bool:
    """Mutate ``module``'s origin table so ``replacement`` is the device copy of the recorded origin ``previous``.

    Returns whether a twin was noted. Only a move of the origin itself off the CPU is noted, so a weight that was
    patched, or moved from anything other than its origin, is never restored to it.
    """
    table = _quantized_origins(module)
    if table is None:
        return False
    entry = table.get(key)
    if entry is None or entry.parameter is not previous or replacement.device.type == "cpu":
        return False
    entry.note_device_twin(replacement)
    return True


_MoveVerdictKey = tuple[Callable[[torch.Tensor], torch.Tensor], torch.device, torch.dtype]


class _RestoreScope:
    """Represents one ``restoring_cpu_origins`` scope: the weights left to the caller and the origins restored so far.

    Skipped weights are held by weak reference keyed by ``id``, so a weight ComfyUI replaces during the scope is
    freed as usual, and a later tensor that reuses its ``id`` is not mistaken for it.
    """

    __slots__ = ("_cpu_move_verdicts", "_skipped", "restored", "restored_bytes")

    def __init__(self, skipped: list[torch.Tensor]) -> None:
        self.restored = 0
        self.restored_bytes = 0
        self._skipped: dict[int, weakref.ref[torch.Tensor]] = {id(tensor): weakref.ref(tensor) for tensor in skipped}
        self._cpu_move_verdicts: dict[_MoveVerdictKey, bool] = {}

    def is_skipped(self, tensor: torch.Tensor) -> bool:
        """Return whether ``tensor`` is a weight the scope leaves to its caller."""
        reference = self._skipped.get(id(tensor))
        return reference is not None and reference() is tensor

    def moves_to_cpu(self, fn: Callable[[torch.Tensor], torch.Tensor], tensor: torch.Tensor) -> bool:
        """Return whether ``fn`` moves ``tensor`` to the CPU without changing its dtype.

        ``fn`` is applied to an empty tensor on ``tensor``'s device, which allocates nothing. The verdict is kept per
        ``fn``, device and dtype, since one ``Module.to`` applies the same ``fn`` to every module it reaches.
        """
        verdict_key = (fn, tensor.device, tensor.dtype)
        verdict = self._cpu_move_verdicts.get(verdict_key)
        if verdict is None:
            moved = fn(torch.empty(0, dtype=tensor.dtype, device=tensor.device))
            verdict = moved.device.type == "cpu" and moved.dtype == tensor.dtype
            self._cpu_move_verdicts[verdict_key] = verdict
        return verdict


_restore_scope: contextvars.ContextVar[_RestoreScope | None] = contextvars.ContextVar(
    "hordelib_cpu_origin_restore_scope",
    default=None,
)


@contextlib.contextmanager
def restoring_cpu_origins(
    module: torch.nn.Module | None = None,
    *,
    skip_keys: set[str] | frozenset[str] | None = None,
) -> Iterator[None]:
    """Context manager inside which a move to the CPU restores recorded origins in place of copying.

    A quantized weight registers its origin ``Parameter`` (see :func:`restore_quantized_origin`), and a plain op
    weight is pointed at its origin tensor (see :func:`restore_plain_origins`). ``skip_keys`` names weights of
    ``module`` that torch moves as usual, typically the keys ComfyUI backed up before patching, since ComfyUI
    restores those from its own backup. Scoping the restore to ComfyUI's unload paths keeps a move made elsewhere
    unchanged.
    """
    if cpu_weight_retention_disabled():
        yield
        return
    skipped: list[torch.Tensor] = []
    if module is not None and skip_keys:
        skipped = [tensor for name, tensor in _named_weights(module) if name in skip_keys]
    scope = _RestoreScope(skipped)
    token = _restore_scope.set(scope)
    try:
        yield
    finally:
        _restore_scope.reset(token)
        if scope.restored:
            logger.debug(
                "Restored {} CPU-origin weight(s) ({} MB) without copying back from the device",
                scope.restored,
                scope.restored_bytes // (1024 * 1024),
            )


def restore_quantized_origin(
    module: torch.nn.Module,
    *,
    key: str,
    param: torch.Tensor,
    fn: Callable[[torch.Tensor], torch.Tensor],
) -> bool:
    """Register ``param``'s recorded CPU origin on ``module`` when ``fn`` moves it to the CPU, and say whether it did.

    The origin ``Parameter`` is registered as it is, so no device-to-host copy runs and no ``Parameter`` wrap can
    detach a non-inference subclass under ``torch.inference_mode``. ``fn`` is applied only to an empty CPU probe:
    it must return the probe unchanged, which holds for a move to the CPU that keeps the dtype.
    """
    scope = _restore_scope.get()
    if scope is None:
        return False
    table = _quantized_origins(module)
    if table is None:
        return False
    entry = table.get(key)
    if entry is None or not entry.is_device_twin(param):
        return False
    origin = entry.parameter
    if origin.device.type != "cpu" or origin.shape != param.shape or origin.dtype != param.dtype:
        return False
    probe = torch.empty(0, dtype=param.dtype)
    if fn(probe) is not probe:
        return False
    module.register_parameter(key, origin)
    scope.restored += 1
    scope.restored_bytes += origin.nbytes
    return True


def restore_plain_origins(module: torch.nn.Module, *, fn: Callable[[torch.Tensor], torch.Tensor]) -> int:
    """Point each of ``module``'s own plain device parameters at its recorded CPU origin when ``fn`` moves it there.

    Applies only inside :func:`restoring_cpu_origins`. A parameter is restored when it has a recorded origin of the
    same shape and dtype, the scope does not skip it, and ``fn`` moves a tensor on its device to the CPU keeping the
    dtype. The caller then runs torch's move, which finds each restored weight already on the CPU and returns it
    unchanged. A ``fn`` that also changes the layout converts the origin as it would have converted a copy. Every other
    parameter, and every buffer, is moved by torch as before. Returns how many parameters were restored.
    """
    scope = _restore_scope.get()
    if scope is None:
        return 0
    restored = 0
    for param in module._parameters.values():
        if param is None or _is_tensor_subclass(param) or scope.is_skipped(param):
            continue
        origin: Any = getattr(param, _ORIGIN_ATTR, None)
        if origin is None:
            continue
        data = param.data
        if data.device.type in ("cpu", "meta") or origin.shape != data.shape or origin.dtype != data.dtype:
            continue
        if not scope.moves_to_cpu(fn, data):
            continue
        scope.restored_bytes += data.nbytes
        param.data = origin
        restored += 1
    scope.restored += restored
    return restored


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

    Runs after ComfyUI's partial unload for the weights whose moves still copied, such as those of modules that are
    not comfy op classes, the keys the restore scope skipped, and buffers. ``unpin`` is called with each weight's key
    before its copy is dropped, since the copy may be registered as pinned host memory and a registration must not
    outlive its allocation. ``skip_keys`` is as for :func:`restore_cpu_origins`. Returns ``(released, bytes)``.
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
