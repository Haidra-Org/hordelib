import os

import torch.version
from loguru import logger
from pydantic import BaseModel
from strenum import StrEnum


class GPUInfo:
    def __init__(self):
        self.avg_load = []
        self.avg_temp = []
        self.avg_power = []
        # Average period in samples, default 10 samples per second, period 5 minutes
        self.samples_per_second = 10
        if not self.is_nvidia():
            logger.warning("GPU info is only supported on NVIDIA GPUs")
        # Look out for device env var hack
        self.device = int(os.getenv("CUDA_VISIBLE_DEVICES", 0))

    # Return a value from the given dictionary supporting dot notation
    def get(self, data, key, default=""):
        # Handle nested structures
        path = key.split(".")

        if len(path) == 1:
            # Simple case
            return data[key] if key in data else default
        # Nested case
        walkdata = data
        for element in path:
            if element in walkdata:
                walkdata = walkdata[element]
            else:
                walkdata = ""
                break
        return walkdata

    def is_nvidia(self):
        if torch.version.cuda is not None:
            return True

        return False

    def is_amd(self):
        if torch.version.hip is not None:
            return True

        return False

    def _get_gpu_data(self) -> dict | None:
        if self.is_nvidia():
            from pynvml.smi import nvidia_smi

            nvsmi = nvidia_smi.getInstance()
            data = nvsmi.DeviceQuery()
            return data.get("gpu", [None])[self.device]

        return None

    def _mem(self, raw):
        unit = "GB"
        mem = raw / 1024
        if mem < 1:
            unit = "MB"
            raw *= 1024
        return f"{round(mem)} {unit}"

    def get_total_vram_mb(self):
        """Get total VRAM in MB as an integer, or 0"""
        value = 0
        data = self._get_gpu_data()
        if data:
            value = self.get(data, "fb_memory_usage.total", "0")
            try:
                value = int(value)
            except ValueError:
                value = 0
        return value

    def get_free_vram_mb(self):
        """Get free VRAM in MB as an integer, or 0"""
        value = 0
        data = self._get_gpu_data()
        if data:
            value = self.get(data, "fb_memory_usage.free", "0")
            try:
                value = int(value)
            except ValueError:
                value = 0
        return value

    def get_info(self):
        if not self.is_nvidia():
            return GPUInfoResult.get_empty_info()

        data = self._get_gpu_data()
        if not data:
            return None

        # Calculate averages
        try:
            gpu_util = int(self.get(data, "utilization.gpu_util", 0))
        except ValueError:
            gpu_util = 0

        try:
            gpu_temp = int(self.get(data, "temperature.gpu_temp", 0))
        except ValueError:
            gpu_temp = 0

        try:
            gpu_power = int(self.get(data, "power_readings.power_draw", 0))
        except ValueError:
            gpu_power = 0

        self.avg_load.append(gpu_util)
        self.avg_temp.append(gpu_temp)
        self.avg_power.append(gpu_power)
        self.avg_load = self.avg_load[-(self.samples_per_second * 60 * 5) :]
        self.avg_power = self.avg_power[-(self.samples_per_second * 60 * 5) :]
        self.avg_temp = self.avg_temp[-(self.samples_per_second * 60 * 5) :]
        avg_load = int(sum(self.avg_load) / len(self.avg_load))
        avg_power = int(sum(self.avg_power) / len(self.avg_power))
        avg_temp = int(sum(self.avg_temp) / len(self.avg_temp))

        vram_total = self.get_total_vram_mb()
        vram_free = self.get_free_vram_mb()
        vram_used = vram_total - vram_free

        return GPUInfoResult(
            supported=True,
            product=self.get(data, "product_name", "unknown"),
            pci_gen=self.get(data, "pci.pci_gpu_link_info.pcie_gen.current_link_gen", "?"),
            pci_width=self.get(data, "pci.pci_gpu_link_info.link_widths.current_link_width", "?"),
            fan_speed=(str(self.get(data, "fan_speed")), Unit.percent),
            vram_total=(vram_total, Unit.megabytes),
            vram_used=(vram_used, Unit.megabytes),
            vram_free=(vram_free, Unit.megabytes),
            load=(gpu_util, Unit.percent),
            temp=(gpu_temp, Unit.degrees_celsius),
            power=(gpu_power, Unit.watts),
            avg_load=(avg_load, Unit.percent),
            avg_temp=(avg_temp, Unit.degrees_celsius),
            avg_power=(avg_power, Unit.watts),
        )


class Unit(StrEnum):
    unitless = ""
    percent = "%"
    degrees_celsius = "C"
    megabytes = "MiB"
    gigabytes = "GiB"
    watts = "W"


class GPUInfoResult(BaseModel):
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

    def __str__(self):
        final_string = ""
        for key, value in self.model_dump().items():
            if isinstance(value, tuple):
                value = f"{value[0]} {value[1]}"
            final_string += f"{key}: {value}\n"
        return final_string

    @classmethod
    def get_empty_info(cls):
        return GPUInfoResult(
            supported=False,
            product="unknown (stats with AMD not supported)",
            pci_gen="?",
            pci_width="?",
            fan_speed=("0", Unit.percent),
            vram_total=(0, Unit.megabytes),
            vram_used=(0, Unit.megabytes),
            vram_free=(0, Unit.megabytes),
            load=(0, Unit.percent),
            temp=(0, Unit.degrees_celsius),
            power=(0, Unit.watts),
            avg_load=(0, Unit.percent),
            avg_temp=(0, Unit.degrees_celsius),
            avg_power=(0, Unit.watts),
        )
