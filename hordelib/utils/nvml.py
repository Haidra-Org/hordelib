"""Optional NVIDIA-only GPU telemetry via NVML, the single home for direct NVML access in hordelib.

NVML is NVIDIA-specific *enrichment* (core utilization, temperature, power, and the most accurate memory
figures); the backend-agnostic metrics that never assume NVIDIA live in
:mod:`hordelib.utils.torch_memory`. Every function here is best-effort: when NVML is missing or the machine
has no NVIDIA driver, each returns ``None`` so non-NVIDIA backends simply report no enrichment rather than
erroring.

Uses ``nvidia-ml-py`` (imported as ``pynvml``), the maintained official binding. Memory is read through
``nvmlDeviceGetMemoryInfo`` with the ``nvmlMemory_v2`` version, which accounts for driver-reserved memory;
the deprecated standalone ``pynvml`` package and its ``pynvml.smi`` interface under-report *used* VRAM
(over-reporting *free*), which is exactly the wrong direction for a VRAM budget.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from loguru import logger
from pydantic import BaseModel

if TYPE_CHECKING:
    from ctypes import c_void_p

_BYTES_PER_MB = 1024 * 1024


class NvmlMemory(BaseModel):
    """Represents a device's NVML memory figures, in MB (from ``nvmlDeviceGetMemoryInfo`` v2)."""

    total_mb: int
    free_mb: int
    used_mb: int


class NvmlDeviceStats(BaseModel):
    """Represents a single NVIDIA device's NVML telemetry snapshot (enrichment for display/diagnostics)."""

    name: str
    memory: NvmlMemory
    utilization_gpu_percent: int
    temperature_celsius: int
    power_watts: int
    fan_speed_percent: int
    pcie_link_generation: int
    pcie_link_width: int


_nvml_unavailable = False
"""Latches True after the first failed NVML init so the worker does not retry NVML every call."""


def _nvml_module() -> object | None:
    """Return the initialised ``nvidia-ml-py`` module, or None when NVML is unavailable.

    Initialisation is attempted once; any failure (no driver, non-NVIDIA host, missing binding) latches
    NVML off for the process and logs the reason at debug. ``nvmlInit`` is idempotent, so calling this
    repeatedly is cheap once initialised.
    """
    global _nvml_unavailable
    if _nvml_unavailable:
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # silence any deprecation warning from a shim install
            import pynvml
        pynvml.nvmlInit()
        return pynvml
    except Exception as nvml_error:  # noqa: BLE001 - any NVML failure means "no NVIDIA telemetry", not a crash
        logger.debug(f"NVML unavailable ({type(nvml_error).__name__}: {nvml_error}); NVIDIA telemetry disabled")
        _nvml_unavailable = True
        return None


def is_nvml_available() -> bool:
    """Return whether NVML telemetry can be read on this machine."""
    return _nvml_module() is not None


def _device_handle(pynvml: object, index: int) -> c_void_p | None:
    """Return the NVML handle for device ``index``, or None when it cannot be resolved."""
    try:
        return pynvml.nvmlDeviceGetHandleByIndex(index)  # type: ignore[attr-defined]
    except Exception as handle_error:  # noqa: BLE001 - a missing device is "no telemetry", not a crash
        logger.debug(f"NVML handle for device {index} unavailable ({type(handle_error).__name__}: {handle_error})")
        return None


def get_device_memory_mb(index: int = 0) -> NvmlMemory | None:
    """Return device ``index``'s memory (MB) via the accurate ``nvmlMemory_v2`` query, or None.

    The v2 query includes driver-reserved memory in ``used``/``free``; older v1 (and the deprecated
    ``pynvml.smi``) omit it, under-reporting used VRAM. Falls back to v1 only if the binding lacks v2.
    """
    pynvml = _nvml_module()
    if pynvml is None:
        return None
    handle = _device_handle(pynvml, index)
    if handle is None:
        return None
    try:
        memory_v2_version = getattr(pynvml, "nvmlMemory_v2", None)
        if memory_v2_version is not None:
            info = pynvml.nvmlDeviceGetMemoryInfo(handle, version=memory_v2_version)  # type: ignore[attr-defined]
        else:
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)  # type: ignore[attr-defined]
    except Exception as memory_error:  # noqa: BLE001 - best-effort enrichment
        logger.debug(f"NVML memory read failed for device {index} ({type(memory_error).__name__}: {memory_error})")
        return None
    return NvmlMemory(
        total_mb=round(info.total / _BYTES_PER_MB),
        free_mb=round(info.free / _BYTES_PER_MB),
        used_mb=round(info.used / _BYTES_PER_MB),
    )


def get_device_utilization_percent(index: int = 0) -> int | None:
    """Return device ``index``'s core-utilization percentage (0-100), or None when unavailable."""
    pynvml = _nvml_module()
    if pynvml is None:
        return None
    handle = _device_handle(pynvml, index)
    if handle is None:
        return None
    try:
        return int(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)  # type: ignore[attr-defined]
    except Exception as utilization_error:  # noqa: BLE001 - best-effort enrichment
        logger.debug(f"NVML utilization read failed for device {index} ({type(utilization_error).__name__})")
        return None


def _device_name(pynvml: object, handle: c_void_p) -> str:
    """Return the device product name, decoding the bytes older bindings return."""
    try:
        name = pynvml.nvmlDeviceGetName(handle)  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001 - name is cosmetic
        return "unknown"
    return name.decode() if isinstance(name, bytes) else str(name)


def _best_effort_int(read: object, *, field: str, index: int) -> int:
    """Return ``read()`` coerced to int, or 0 when the card does not support that sensor."""
    try:
        return int(read())  # type: ignore[operator]
    except Exception as sensor_error:  # noqa: BLE001 - many sensors are unsupported per-card
        logger.debug(f"NVML {field} unsupported for device {index} ({type(sensor_error).__name__})")
        return 0


def get_device_stats(index: int = 0) -> NvmlDeviceStats | None:
    """Return a full NVML telemetry snapshot for device ``index``, or None when NVML is unavailable.

    Memory uses the accurate v2 query; each enrichment sensor (temperature, power, fan, PCIe) is read
    best-effort and defaults to 0 when the card does not expose it.
    """
    pynvml = _nvml_module()
    if pynvml is None:
        return None
    handle = _device_handle(pynvml, index)
    if handle is None:
        return None
    memory = get_device_memory_mb(index)
    if memory is None:
        return None

    temperature_sensor = getattr(pynvml, "NVML_TEMPERATURE_GPU", 0)
    power_milliwatts = _best_effort_int(lambda: pynvml.nvmlDeviceGetPowerUsage(handle), field="power", index=index)
    return NvmlDeviceStats(
        name=_device_name(pynvml, handle),
        memory=memory,
        utilization_gpu_percent=get_device_utilization_percent(index) or 0,
        temperature_celsius=_best_effort_int(
            lambda: pynvml.nvmlDeviceGetTemperature(handle, temperature_sensor),
            field="temperature",
            index=index,
        ),
        power_watts=round(power_milliwatts / 1000),
        fan_speed_percent=_best_effort_int(lambda: pynvml.nvmlDeviceGetFanSpeed(handle), field="fan", index=index),
        pcie_link_generation=_best_effort_int(
            lambda: pynvml.nvmlDeviceGetCurrPcieLinkGeneration(handle),
            field="pcie_gen",
            index=index,
        ),
        pcie_link_width=_best_effort_int(
            lambda: pynvml.nvmlDeviceGetCurrPcieLinkWidth(handle),
            field="pcie_width",
            index=index,
        ),
    )
