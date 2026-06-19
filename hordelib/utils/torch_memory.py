"""Memory and accelerator statistics that work with or without ComfyUI loaded.

Consumers (notably horde-worker-reGen's process monitoring, including its safety process,
which must never trigger a ComfyUI import) need VRAM/RAM numbers and a device inventory in
processes where ``hordelib.initialise()`` may not have run. When ComfyUI *is* loaded in this
process, these delegate to its backend-agnostic memory management for exact agreement with
what the execution backend sees; otherwise they fall back to plain torch device queries that
cover every torch backend (CUDA/ROCm, Intel XPU, Apple MPS, CPU), not just CUDA.

This module is the single source of accelerator truth for the worker: it never assumes
NVIDIA. NVML is treated as optional NVIDIA-only enrichment (see :mod:`hordelib.utils.nvml`),
never as a hard requirement here; :func:`get_accelerator_utilization_percent` is the one place
that consults it, and only for the CUDA/NVIDIA backend.
"""

from __future__ import annotations

import sys
from types import ModuleType

import psutil
from loguru import logger
from pydantic import BaseModel
from strenum import StrEnum

_MB = 1024 * 1024

_COMFY_MODEL_MANAGEMENT = "comfy.model_management"


class AcceleratorKind(StrEnum):
    """The compute backend a device is driven through.

    ``cuda`` and ``rocm`` both present through ``torch.cuda`` in PyTorch; they are
    distinguished by ``torch.version.hip``. ``directml`` cannot be enumerated without ComfyUI
    loaded, so the plain-torch fallback never reports it (ComfyUI does).
    """

    cuda = "cuda"
    rocm = "rocm"
    xpu = "xpu"
    npu = "npu"
    mlu = "mlu"
    mps = "mps"
    directml = "directml"
    cpu = "cpu"


class AcceleratorInfo(BaseModel):
    """A single compute device discovered on this machine."""

    index: int
    name: str
    total_vram_mb: int
    kind: AcceleratorKind


def _comfy_model_management() -> ModuleType | None:
    """Return ComfyUI's ``model_management`` module iff it is already imported here.

    Uses ``sys.modules`` rather than a fresh import so that calling this never triggers the
    heavy (and, in the safety process, forbidden) ComfyUI import.
    """
    return sys.modules.get(_COMFY_MODEL_MANAGEMENT)


def _comfy_memory_funcs() -> tuple | None:
    """Return comfy's (get_total_memory, get_free_memory) if comfy is imported here."""
    mm = _comfy_model_management()
    if mm is None:
        return None
    total = getattr(mm, "get_total_memory", None)
    free = getattr(mm, "get_free_memory", None)
    if total is None or free is None:
        return None
    return total, free


def _active_torch_kind() -> AcceleratorKind:
    """Detect the backend the active torch build talks to, without ComfyUI loaded."""
    import torch

    if getattr(torch.version, "hip", None) is not None and torch.cuda.is_available():
        return AcceleratorKind.rocm
    if getattr(torch.version, "cuda", None) is not None and torch.cuda.is_available():
        return AcceleratorKind.cuda
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return AcceleratorKind.xpu
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return AcceleratorKind.mps
    return AcceleratorKind.cpu


def _torch_fallback_vram_bytes(*, free: bool) -> int:
    """Total or free device memory of the active torch backend, in bytes (0 on failure).

    Mirrors ComfyUI's per-backend handling for the case where ComfyUI is not loaded: CUDA/ROCm
    and XPU expose ``mem_get_info``; MPS and CPU have no device VRAM, so system RAM is the
    honest answer (matching ComfyUI's ``get_total_memory`` for those backends).
    """
    import torch

    kind = _active_torch_kind()
    try:
        if kind in (AcceleratorKind.cuda, AcceleratorKind.rocm):
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            return free_bytes if free else total_bytes
        if kind is AcceleratorKind.xpu and hasattr(torch.xpu, "mem_get_info"):
            free_bytes, total_bytes = torch.xpu.mem_get_info()
            return free_bytes if free else total_bytes
    except Exception as e:
        logger.debug(f"torch VRAM probe failed for {kind}: {e}")
        return 0

    vmem = psutil.virtual_memory()
    return vmem.available if free else vmem.total


def get_torch_total_vram_mb() -> int:
    """Total VRAM of the active torch device, in MB (system RAM for CPU/MPS backends)."""
    comfy_funcs = _comfy_memory_funcs()
    if comfy_funcs is not None:
        return round(comfy_funcs[0]() / _MB)
    return round(_torch_fallback_vram_bytes(free=False) / _MB)


def get_torch_free_vram_mb() -> int:
    """Free VRAM of the active torch device, in MB (available RAM for CPU/MPS backends)."""
    comfy_funcs = _comfy_memory_funcs()
    if comfy_funcs is not None:
        return round(comfy_funcs[1]() / _MB)
    return round(_torch_fallback_vram_bytes(free=True) / _MB)


def get_free_ram_mb() -> int:
    """Available system RAM, in MB."""
    return round(psutil.virtual_memory().available / _MB)


def log_free_ram() -> None:
    logger.debug(f"Free VRAM: {get_torch_free_vram_mb():0.0f} MB, Free RAM: {get_free_ram_mb():0.0f} MB")


def get_accelerator_utilization_percent(index: int = 0) -> int | None:
    """Return device ``index``'s core-utilization percentage (0-100), or None when unavailable.

    Utilization is vendor telemetry with no portable torch/ComfyUI equivalent, so this is best-effort and
    backend-specific: the NVIDIA CUDA backend is read via NVML (:mod:`hordelib.utils.nvml`); every other
    backend (ROCm, XPU, MPS, CPU) returns None until a vendor source is wired, so callers simply report no
    GPU duty cycle there. ROCm presents through ``torch.cuda`` but is not an NVML device, so it returns None.
    """
    if _active_torch_kind() is not AcceleratorKind.cuda:
        return None
    from hordelib.utils import nvml

    return nvml.get_device_utilization_percent(index)


def _enumerate_via_comfy(mm: ModuleType) -> list[AcceleratorInfo]:
    """Enumerate devices through ComfyUI's backend-agnostic device APIs."""
    devices = mm.get_all_torch_devices()
    accelerators: list[AcceleratorInfo] = []
    for index, device in enumerate(devices):
        device_type = getattr(device, "type", "cpu")
        if device_type == "cuda":
            import torch

            kind = AcceleratorKind.rocm if getattr(torch.version, "hip", None) is not None else AcceleratorKind.cuda
        else:
            kind = _COMFY_TYPE_TO_KIND.get(device_type, AcceleratorKind.cpu)

        try:
            total_mb = round(mm.get_total_memory(device) / _MB)
        except Exception as e:
            logger.debug(f"Could not read total memory for device {device}: {e}")
            total_mb = 0

        try:
            name = mm.get_torch_device_name(device)
        except Exception:
            name = str(device)

        accelerators.append(AcceleratorInfo(index=index, name=name, total_vram_mb=total_mb, kind=kind))
    return accelerators


_COMFY_TYPE_TO_KIND: dict[str, AcceleratorKind] = {
    "cuda": AcceleratorKind.cuda,
    "xpu": AcceleratorKind.xpu,
    "npu": AcceleratorKind.npu,
    "mlu": AcceleratorKind.mlu,
    "mps": AcceleratorKind.mps,
    "cpu": AcceleratorKind.cpu,
}


def _enumerate_via_torch() -> list[AcceleratorInfo]:
    """Enumerate devices with plain torch (ComfyUI not loaded in this process).

    Cannot see DirectML (ComfyUI owns that path); reports a single CPU pseudo-device when no
    accelerator is present so callers always get at least one device to reason about.
    """
    import torch

    kind = _active_torch_kind()
    accelerators: list[AcceleratorInfo] = []

    if kind in (AcceleratorKind.cuda, AcceleratorKind.rocm):
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            accelerators.append(
                AcceleratorInfo(
                    index=i,
                    name=getattr(props, "name", f"cuda:{i}"),
                    total_vram_mb=round(getattr(props, "total_memory", 0) / _MB),
                    kind=kind,
                ),
            )
    elif kind is AcceleratorKind.xpu:
        for i in range(torch.xpu.device_count()):
            props = torch.xpu.get_device_properties(i)
            accelerators.append(
                AcceleratorInfo(
                    index=i,
                    name=getattr(props, "name", f"xpu:{i}"),
                    total_vram_mb=round(getattr(props, "total_memory", 0) / _MB),
                    kind=kind,
                ),
            )
    elif kind is AcceleratorKind.mps:
        accelerators.append(
            AcceleratorInfo(
                index=0,
                name="Apple MPS (unified memory)",
                total_vram_mb=round(psutil.virtual_memory().total / _MB),
                kind=kind,
            ),
        )

    if not accelerators:
        accelerators.append(
            AcceleratorInfo(
                index=0,
                name="CPU",
                total_vram_mb=round(psutil.virtual_memory().total / _MB),
                kind=AcceleratorKind.cpu,
            ),
        )
    return accelerators


def enumerate_accelerators() -> list[AcceleratorInfo]:
    """Return every compute device on this machine, across all torch/ComfyUI backends.

    Prefers ComfyUI's detection when it is loaded in this process (it covers DirectML, NPU and
    MLU too); otherwise uses a plain-torch fallback. Always returns at least one device (a CPU
    pseudo-device when no accelerator is found), so callers never see an empty inventory the way
    a bare ``torch.cuda.device_count()`` loop does on non-CUDA backends.
    """
    mm = _comfy_model_management()
    if mm is not None:
        try:
            return _enumerate_via_comfy(mm)
        except Exception as e:
            logger.debug(f"ComfyUI device enumeration failed, falling back to torch: {e}")
    return _enumerate_via_torch()


def clear_accelerator_cache() -> None:
    """Release cached device memory on whatever backend is active (a no-op on CPU).

    Delegates to ComfyUI's ``soft_empty_cache`` when loaded (backend-aware for CUDA/XPU/NPU/
    MLU/MPS); otherwise clears the active torch backend's cache directly. Never imports from
    ``torch.cuda`` unconditionally, so it is safe on non-NVIDIA builds.
    """
    mm = _comfy_model_management()
    if mm is not None:
        soft_empty_cache = getattr(mm, "soft_empty_cache", None)
        if soft_empty_cache is not None:
            soft_empty_cache()
            return

    import torch

    kind = _active_torch_kind()
    try:
        if kind in (AcceleratorKind.cuda, AcceleratorKind.rocm):
            torch.cuda.empty_cache()
        elif kind is AcceleratorKind.xpu and hasattr(torch.xpu, "empty_cache"):
            torch.xpu.empty_cache()
        elif kind is AcceleratorKind.mps and hasattr(torch, "mps"):
            torch.mps.empty_cache()
    except Exception as e:
        logger.debug(f"clear_accelerator_cache failed for {kind}: {e}")
