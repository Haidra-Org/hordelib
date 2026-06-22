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

from hordelib.utils.torch_memory import AcceleratorKind


def device_pin_env(kind: AcceleratorKind, index: int) -> tuple[dict[str, str], list[str]]:
    """Return the env-var patch and extra ComfyUI args that restrict a process to device ``index``.

    After applying both, the OS/driver presents only that one device to the process and it
    becomes ``cuda:0`` (or the backend equivalent), so all single-device assumptions inside
    hordelib and the vendored ComfyUI remain correct without any code changes there.

    Args:
        kind: The :class:`~hordelib.utils.torch_memory.AcceleratorKind` of the target device.
        index: The **global** device index (as reported by
            :func:`~hordelib.utils.torch_memory.enumerate_accelerators`).

    Returns:
        A 2-tuple ``(env_vars, extra_args)`` where ``env_vars`` is a dict to merge into
        ``os.environ`` and ``extra_args`` is a list of CLI arguments to append to
        ``extra_comfyui_args``. Either or both may be empty when the backend needs no masking
        (cpu, mps).
    """
    if kind is AcceleratorKind.cuda:
        return {"CUDA_VISIBLE_DEVICES": str(index)}, []

    if kind is AcceleratorKind.rocm:
        # ROCm presents through torch.cuda, so CUDA_VISIBLE_DEVICES is honoured too; keep them in sync.
        return {"HIP_VISIBLE_DEVICES": str(index), "CUDA_VISIBLE_DEVICES": str(index)}, []

    if kind is AcceleratorKind.xpu:
        return {"ZE_AFFINITY_MASK": str(index)}, []

    if kind is AcceleratorKind.directml:
        # DirectML device selection happens via a ComfyUI CLI flag, not an env var.
        return {}, ["--directml", str(index)]

    # cpu and mps have no multi-device isolation mechanism; masking is a no-op.
    return {}, []
