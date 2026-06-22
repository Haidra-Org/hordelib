"""Human-facing GPU information for diagnostics and the benchmark.

VRAM totals/frees come from the backend-agnostic :mod:`hordelib.utils.torch_memory` (correct on every
backend ComfyUI supports); NVIDIA-only enrichment (utilization, temperature, power, fan, PCIe) comes from
:mod:`hordelib.utils.nvml`. On non-NVIDIA backends the rich stats are simply marked unsupported rather than
erroring.
"""

from __future__ import annotations

import os

import torch.version
from loguru import logger
from pydantic import BaseModel
from strenum import StrEnum

from hordelib.utils import nvml
from hordelib.utils.torch_memory import get_torch_free_vram_mb, get_torch_total_vram_mb

_AVERAGE_SAMPLES_PER_SECOND = 10
_AVERAGE_WINDOW_SECONDS = 60 * 5


class GPUInfo:
    """Collects per-device GPU statistics, keeping rolling averages of load, temperature, and power."""

    def __init__(self, device_index: int | None = None) -> None:
        """Initialize the collector for the given device index.

        Args:
            device_index: The global NVML/CUDA device index to query. When ``None`` (default), reads
                ``CUDA_VISIBLE_DEVICES`` (which is ``0`` inside a pinned subprocess, and the real
                device index in the unmasked orchestrator). Pass an explicit index when an unmasked
                caller needs to survey a specific card without masking the process.
        """
        self.avg_load: list[int] = []
        self.avg_temp: list[int] = []
        self.avg_power: list[int] = []
        self._average_window_samples = _AVERAGE_SAMPLES_PER_SECOND * _AVERAGE_WINDOW_SECONDS
        if not self.is_nvidia():
            logger.warning("Detailed GPU info (load/temp/power) is only available on NVIDIA GPUs")
        if device_index is not None:
            self.device = device_index
        else:
            self.device = int(os.getenv("CUDA_VISIBLE_DEVICES", "0"))

    def is_nvidia(self) -> bool:
        """Return whether the active torch build targets NVIDIA CUDA (not ROCm/HIP)."""
        return torch.version.cuda is not None and torch.version.hip is None

    def is_amd(self) -> bool:
        """Return whether the active torch build targets AMD ROCm/HIP."""
        return torch.version.hip is not None

    def get_total_vram_mb(self) -> int:
        """Return the device's total VRAM in MB (backend-agnostic), or 0 when unavailable."""
        return get_torch_total_vram_mb()

    def get_free_vram_mb(self) -> int:
        """Return the device's free VRAM in MB (backend-agnostic), or 0 when unavailable."""
        return get_torch_free_vram_mb()

    def _append_rolling(self, samples: list[int], value: int) -> list[int]:
        """Append ``value`` to ``samples`` and trim it to the averaging window; return the trimmed list."""
        samples.append(value)
        return samples[-self._average_window_samples :]

    def get_info(self) -> GPUInfoResult | None:
        """Return a snapshot of this device's stats, or None when even total VRAM cannot be read.

        On non-NVIDIA backends the snapshot carries backend-agnostic VRAM figures with the rich sensors
        marked unsupported. On NVIDIA, sensors come from NVML (accurate v2 memory).
        """
        total_vram_mb = self.get_total_vram_mb()
        free_vram_mb = self.get_free_vram_mb()

        stats = nvml.get_device_stats(self.device) if self.is_nvidia() else None
        if stats is None:
            return GPUInfoResult.get_empty_info(vram_total_mb=total_vram_mb, vram_free_mb=free_vram_mb)

        avg_load = self._append_rolling(self.avg_load, stats.utilization_gpu_percent)
        avg_temp = self._append_rolling(self.avg_temp, stats.temperature_celsius)
        avg_power = self._append_rolling(self.avg_power, stats.power_watts)

        # NVML's v2 memory is more accurate than the comfy/torch totals here; prefer it when present.
        vram_total = stats.memory.total_mb or total_vram_mb
        vram_free = stats.memory.free_mb or free_vram_mb
        vram_used = stats.memory.used_mb or max(vram_total - vram_free, 0)

        return GPUInfoResult(
            supported=True,
            product=stats.name,
            pci_gen=str(stats.pcie_link_generation),
            pci_width=str(stats.pcie_link_width),
            fan_speed=(str(stats.fan_speed_percent), Unit.percent),
            vram_total=(vram_total, Unit.megabytes),
            vram_used=(vram_used, Unit.megabytes),
            vram_free=(vram_free, Unit.megabytes),
            load=(stats.utilization_gpu_percent, Unit.percent),
            temp=(stats.temperature_celsius, Unit.degrees_celsius),
            power=(stats.power_watts, Unit.watts),
            avg_load=(round(sum(avg_load) / len(avg_load)), Unit.percent),
            avg_temp=(round(sum(avg_temp) / len(avg_temp)), Unit.degrees_celsius),
            avg_power=(round(sum(avg_power) / len(avg_power)), Unit.watts),
        )


class Unit(StrEnum):
    """A display unit for a GPU statistic."""

    unitless = ""
    percent = "%"
    degrees_celsius = "C"
    megabytes = "MiB"
    gigabytes = "GiB"
    watts = "W"


class GPUInfoResult(BaseModel):
    """Represents a single snapshot of a GPU's statistics, each value paired with its display unit."""

    supported: bool
    product: str
    pci_gen: str
    pci_width: str
    fan_speed: tuple[str, Unit]
    vram_total: tuple[int, Unit]
    vram_used: tuple[int, Unit]
    vram_free: tuple[int, Unit]
    load: tuple[int, Unit]
    temp: tuple[int, Unit]
    power: tuple[int, Unit]
    avg_load: tuple[int, Unit]
    avg_temp: tuple[int, Unit]
    avg_power: tuple[int, Unit]

    def __str__(self) -> str:
        """Return a newline-separated ``key: value unit`` rendering of every field."""
        final_string = ""
        for key, value in self.model_dump().items():
            rendered_value = f"{value[0]} {value[1]}" if isinstance(value, tuple) else value
            final_string += f"{key}: {rendered_value}\n"
        return final_string

    @classmethod
    def get_empty_info(cls, vram_total_mb: int = 0, vram_free_mb: int = 0) -> GPUInfoResult:
        """Create a result carrying only backend-agnostic VRAM, with the NVIDIA-only sensors zeroed."""
        vram_used_mb = max(vram_total_mb - vram_free_mb, 0)
        return GPUInfoResult(
            supported=False,
            product="unknown (detailed stats unavailable on this backend)",
            pci_gen="?",
            pci_width="?",
            fan_speed=("0", Unit.percent),
            vram_total=(vram_total_mb, Unit.megabytes),
            vram_used=(vram_used_mb, Unit.megabytes),
            vram_free=(vram_free_mb, Unit.megabytes),
            load=(0, Unit.percent),
            temp=(0, Unit.degrees_celsius),
            power=(0, Unit.watts),
            avg_load=(0, Unit.percent),
            avg_temp=(0, Unit.degrees_celsius),
            avg_power=(0, Unit.watts),
        )
