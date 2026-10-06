"""Measure the system commit a checkpoint mapping charges on Windows, per mapping mechanism.

torch maps a non-shared file copy-on-write on Windows, which charges the whole file size to system commit
at map time. This test pins that behaviour for ``torch.UntypedStorage.from_file`` and reports what
``comfy_aimdo``'s ``ModelMMAP`` (the mapping behind ComfyUI's ``load_safetensors``) and a read-only
``mmap.mmap`` charge for the same file, so a loader change can be judged against measured figures.

The file is the smallest ``.safetensors`` under ``$AIWORKER_CACHE_HOME/vae`` within the size window below,
so the mappings stay a few hundred megabytes and are released before the next one is made.
"""

from __future__ import annotations

import ctypes
import gc
import importlib
import mmap
import os
import sys
from collections.abc import Callable
from pathlib import Path

import psutil
import pytest
import torch

_MINIMUM_FILE_BYTES = 300 * 1024**2
_MAXIMUM_FILE_BYTES = 1536 * 1024**2
"""Upper bound that keeps the measurement off multi-gigabyte checkpoints on a host that is in use."""

_KNOWN_CHARGE_SHARE = 0.9


class _MemoryStatusEx(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


class _CommitCharge:
    """Represents the commit a mapping added, system-wide and for this process."""

    def __init__(self, *, system_bytes: int, process_bytes: int) -> None:
        self.system_bytes = system_bytes
        self.process_bytes = process_bytes

    def describe(self, file_bytes: int) -> str:
        return (
            f"system {self.system_bytes / 2**20:.1f} MiB ({self.system_bytes / file_bytes:.1%}), "
            f"process private {self.process_bytes / 2**20:.1f} MiB ({self.process_bytes / file_bytes:.1%})"
        )


def _system_commit_used_bytes() -> int:
    status = _MemoryStatusEx()
    status.dwLength = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):  # type: ignore[attr-defined]
        raise ctypes.WinError()  # type: ignore[attr-defined]
    return status.ullTotalPageFile - status.ullAvailPageFile


def _process_private_bytes() -> int:
    return psutil.Process().memory_info().private  # type: ignore[attr-defined]


def _smallest_measurable_file() -> Path | None:
    cache_home = os.environ.get("AIWORKER_CACHE_HOME")
    if not cache_home:
        return None
    vae_directory = Path(cache_home) / "vae"
    if not vae_directory.is_dir():
        return None
    candidates = [
        candidate
        for candidate in vae_directory.glob("*.safetensors")
        if _MINIMUM_FILE_BYTES <= candidate.stat().st_size <= _MAXIMUM_FILE_BYTES
    ]
    return min(candidates, key=lambda candidate: candidate.stat().st_size, default=None)


def _measure(map_file: Callable[[], object]) -> _CommitCharge:
    """Return the commit added while the object ``map_file`` returns is alive, released before returning."""
    gc.collect()
    system_before = _system_commit_used_bytes()
    process_before = _process_private_bytes()
    mapping = map_file()
    system_after = _system_commit_used_bytes()
    process_after = _process_private_bytes()
    del mapping
    gc.collect()
    return _CommitCharge(system_bytes=system_after - system_before, process_bytes=process_after - process_before)


@pytest.mark.skipif(sys.platform != "win32", reason="copy-on-write commit charging is Windows behaviour")
def test_mapping_commit_charge() -> None:
    """``from_file`` charges the file size to commit; ``ModelMMAP`` and a read-only mmap are reported."""
    found_file = _smallest_measurable_file()
    if found_file is None:
        pytest.skip("no .safetensors of 300 MB to 1.5 GB under $AIWORKER_CACHE_HOME/vae")
    file_path: Path = found_file
    file_bytes = file_path.stat().st_size

    from comfy_aimdo import control as aimdo_control
    from comfy_aimdo import model_mmap

    # model_mmap binds control.lib and declares the native signatures at import, and hordelib never calls
    # control.init(), so the module is reloaded after init here and again after deinit.
    initialised_here = aimdo_control.lib is None
    if initialised_here:
        if not aimdo_control.init():
            pytest.skip("comfy_aimdo could not load its native library")
        model_mmap = importlib.reload(model_mmap)

    def _map_with_from_file() -> object:
        return torch.UntypedStorage.from_file(str(file_path), False, file_bytes)

    def _map_with_model_mmap() -> object:
        aimdo_mapping = model_mmap.ModelMMAP(str(file_path))
        mapped_view = memoryview((ctypes.c_uint8 * file_bytes).from_address(aimdo_mapping.get()))
        return (aimdo_mapping, mapped_view)

    def _map_read_only() -> object:
        with open(file_path, "rb") as handle:
            return mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)

    try:
        from_file_charge = _measure(_map_with_from_file)
        model_mmap_charge = _measure(_map_with_model_mmap)
        read_only_charge = _measure(_map_read_only)
    finally:
        if initialised_here:
            aimdo_control.deinit()
            importlib.reload(model_mmap)

    report = (
        f"{file_path.name} ({file_bytes / 2**20:.1f} MiB): "
        f"from_file {from_file_charge.describe(file_bytes)}; "
        f"ModelMMAP {model_mmap_charge.describe(file_bytes)}; "
        f"mmap ACCESS_READ {read_only_charge.describe(file_bytes)}"
    )
    print(report)
    assert from_file_charge.system_bytes >= _KNOWN_CHARGE_SHARE * file_bytes, report
