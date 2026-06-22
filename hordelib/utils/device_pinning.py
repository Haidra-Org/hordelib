"""Single source of truth for masking a subprocess to one compute device.

Each inference child process is pinned to exactly one device so that every
single-device assumption in ComfyUI and hordelib (``cuda:0``, module-level
``total_vram``, ``UserSettings._vram_to_leave_free_mb``) stays correct without
modification inside that child. The orchestrator (which is torch-free) calls
:func:`device_pin_env` to get the env-var patch and extra ComfyUI CLI args it
must apply before spawning the child; the child itself applies them before
calling ``hordelib.initialise()``.

:func:`device_pin_env` is the only function here. All callers (worker entry
points, ``hordelib.initialise``'s optional convenience path) must go through
it so that backend-specific masking details stay in one place.
"""

from __future__ import annotations

import os

from hordelib.utils.torch_memory import AcceleratorKind


def _resolve_physical_id(env_var: str, logical_index: int) -> str:
    """Translate a logical device index into the physical device identifier for ``env_var``.

    When the user restricts visible devices via an environment variable (e.g.
    ``CUDA_VISIBLE_DEVICES=0,2``), torch and ComfyUI re-index the visible set as 0, 1, 2, ...
    :func:`~hordelib.utils.torch_memory.enumerate_accelerators` returns these **logical** indices.
    Writing the logical index directly into the child's env var would pin it to the wrong physical
    device; this function resolves back to the physical identifier at ``logical_index``'s position
    in the current visible list so the child's env var is correct.

    When the variable is unset or empty every logical index equals its physical counterpart, so
    ``str(logical_index)`` is returned unchanged. UUID-format entries (``GPU-...``) are passed
    through as-is, as are any special sentinel values (``NoDevFiles``, ``-1``).

    Args:
        env_var: The environment variable name to read (e.g. ``"CUDA_VISIBLE_DEVICES"``).
        logical_index: The 0-based index within the currently visible device set.

    Returns:
        The physical device identifier (integer string or UUID) for the child's env var.

    Raises:
        IndexError: If ``logical_index`` is out of range for the current visible set.
    """
    current = os.environ.get(env_var, "")
    if not current:
        return str(logical_index)
    entries = [e.strip() for e in current.split(",")]
    return entries[logical_index]


def device_pin_env(kind: AcceleratorKind, index: int) -> tuple[dict[str, str], list[str]]:
    """Return the env-var patch and extra ComfyUI args that restrict a process to device ``index``.

    ``index`` is the **logical** index as returned by
    :func:`~hordelib.utils.torch_memory.enumerate_accelerators` (0-based within whatever device
    set is currently visible to this process). When the caller's environment already restricts
    visible devices via ``CUDA_VISIBLE_DEVICES`` or equivalent, the logical index is translated
    back to the physical device identifier so the child's env var pins the correct card.

    After applying both return values, the OS/driver presents only that one device to the process
    and it becomes ``cuda:0`` (or the backend equivalent), so all single-device assumptions inside
    hordelib and the vendored ComfyUI remain correct without any code changes there.

    Args:
        kind: The :class:`~hordelib.utils.torch_memory.AcceleratorKind` of the target device.
        index: The **logical** device index (as reported by
            :func:`~hordelib.utils.torch_memory.enumerate_accelerators`).

    Returns:
        A 2-tuple ``(env_vars, extra_args)`` where ``env_vars`` is a dict to merge into
        ``os.environ`` and ``extra_args`` is a list of CLI arguments to append to
        ``extra_comfyui_args``. Either or both may be empty when the backend needs no masking
        (cpu, mps).
    """
    if kind is AcceleratorKind.cuda:
        physical = _resolve_physical_id("CUDA_VISIBLE_DEVICES", index)
        return {"CUDA_VISIBLE_DEVICES": physical}, []

    if kind is AcceleratorKind.rocm:
        # ROCm presents through torch.cuda, so CUDA_VISIBLE_DEVICES is honoured too; keep both in sync.
        physical = _resolve_physical_id("HIP_VISIBLE_DEVICES", index)
        return {"HIP_VISIBLE_DEVICES": physical, "CUDA_VISIBLE_DEVICES": physical}, []

    if kind is AcceleratorKind.xpu:
        physical = _resolve_physical_id("ZE_AFFINITY_MASK", index)
        return {"ZE_AFFINITY_MASK": physical}, []

    if kind is AcceleratorKind.directml:
        # DirectML device selection happens via a ComfyUI CLI flag, not an env var.
        return {}, ["--directml", str(index)]

    # cpu and mps have no multi-device isolation mechanism; masking is a no-op.
    return {}, []
