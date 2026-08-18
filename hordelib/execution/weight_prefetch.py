"""Page in a module's CPU-resident weights ahead of the RAM->VRAM transfer.

Checkpoint weights are kept as zero-copy views over the memory-mapped safetensors file (see
``zero_copy_load``), so the bytes are read from disk only when something touches them: the first
``.to(device)`` in ComfyUI's model load. That read is driven by page faults, one small synchronous I/O
at a time, and on the load path it sits inside the sampling window, after the process has been cleared
to load, so a cold model costs several seconds of idle GPU per switch (a 5 GB UNet paged in this way
transfers at well under half the disk's sequential rate).

Asking the OS to prefetch the whole mapped range up front turns those faults into a few large parallel
reads at the disk's full rate, and doing it *before* the transfer (at preload time, or while the process
waits for its sampling clearance) takes the disk read off the GPU's critical path entirely: by the time
the weights move, the pages are already in the file cache and the copy runs at memory speed.

Windows: ``PrefetchVirtualMemory`` on the coalesced address ranges. POSIX: ``madvise(MADV_WILLNEED)`` on
each page-aligned range. Both are hints; a failure to prefetch only means the load pages in lazily as
before. Ranges over private (non-file-backed) memory are harmless: already-resident pages cost nothing.

The prefetch alone lands the pages in the file cache, not in the process; the copy that follows still
soft-faults every page in, which on Windows runs at a fraction of memory speed (a 5 GB UNet took over two
seconds from the standby list against under a second once mapped). So after the prefetch each tensor is
touched, one read per page, on the same background thread: the pages join the working set there, off the
critical path, and stay mapped across the load/unload cycle while the tensor lives.
"""

from __future__ import annotations

import ctypes
import os
import sys
import threading
import time
from collections.abc import Iterable

import torch
from loguru import logger

_DISABLE_ENV_VAR = "HORDELIB_DISABLE_WEIGHT_PREFETCH"
_TRUTHY_VALUES = {"1", "true", "yes", "on"}

# Neighbouring tensors of one file mapping are contiguous or nearly so; merging across small gaps keeps
# a whole component down to a handful of ranges instead of one entry per tensor.
_COALESCE_GAP_BYTES = 4 * 1024 * 1024
_PAGE = 4096

_lock = threading.Lock()
_in_flight: set[int] = set()


def weight_prefetch_disabled() -> bool:
    """Return whether the kill-switch env var disables weight prefetch (default: enabled)."""
    return os.environ.get(_DISABLE_ENV_VAR, "").strip().lower() in _TRUTHY_VALUES


def collect_cpu_weight_ranges(module: torch.nn.Module) -> list[tuple[int, int]]:
    """Return coalesced ``(address, length)`` ranges covering the module's CPU-resident weights and buffers."""
    spans: list[tuple[int, int]] = []
    tensors: Iterable[torch.Tensor] = (
        *(p.data for p in module.parameters(recurse=True)),
        *module.buffers(recurse=True),
    )
    for tensor in tensors:
        if tensor.device.type != "cpu" or tensor.is_meta or tensor.numel() == 0:
            continue
        try:
            storage = tensor.untyped_storage()
            spans.append((storage.data_ptr(), storage.nbytes()))
        except (RuntimeError, AttributeError):
            continue
    if not spans:
        return []
    spans.sort()
    merged: list[list[int]] = [[spans[0][0], spans[0][0] + spans[0][1]]]
    for start, length in spans[1:]:
        end = start + length
        last = merged[-1]
        if start <= last[1] + _COALESCE_GAP_BYTES:
            last[1] = max(last[1], end)
        else:
            merged.append([start, end])
    return [(start, end - start) for start, end in merged]


def _prefetch_ranges_windows(ranges: list[tuple[int, int]]) -> bool:
    import ctypes.wintypes as wt

    class _MemoryRange(ctypes.Structure):
        _fields_ = [("VirtualAddress", ctypes.c_void_p), ("NumberOfBytes", ctypes.c_size_t)]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    prefetch = kernel32.PrefetchVirtualMemory
    prefetch.argtypes = [wt.HANDLE, ctypes.c_size_t, ctypes.POINTER(_MemoryRange), wt.DWORD]
    prefetch.restype = wt.BOOL
    entries = (_MemoryRange * len(ranges))(*(_MemoryRange(addr, length) for addr, length in ranges))
    ok = bool(prefetch(kernel32.GetCurrentProcess(), len(ranges), entries, 0))
    if not ok:
        logger.debug("PrefetchVirtualMemory failed: error={}", ctypes.get_last_error())
    return ok


def _prefetch_ranges_posix(ranges: list[tuple[int, int]]) -> bool:
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        madvise = libc.madvise
    except (OSError, AttributeError):
        return False
    madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    madvise.restype = ctypes.c_int
    madv_willneed = 3
    ok = True
    for addr, length in ranges:
        aligned = addr - (addr % _PAGE)
        if madvise(aligned, length + (addr - aligned), madv_willneed) != 0:
            ok = False
    return ok


def prefetch_ranges(ranges: list[tuple[int, int]]) -> bool:
    """Ask the OS to page in ``ranges``; returns whether the hint was accepted."""
    if not ranges:
        return True
    try:
        if sys.platform == "win32":
            return _prefetch_ranges_windows(ranges)
        return _prefetch_ranges_posix(ranges)
    except Exception as exc:
        logger.debug("Weight prefetch hint failed: {}", exc)
        return False


def _cpu_weight_tensors(module: torch.nn.Module) -> list[torch.Tensor]:
    tensors: list[torch.Tensor] = []
    for tensor in (*(p.data for p in module.parameters(recurse=True)), *module.buffers(recurse=True)):
        if tensor.device.type != "cpu" or tensor.is_meta or tensor.numel() == 0:
            continue
        tensors.append(tensor)
    return tensors


def touch_cpu_weights(tensors: Iterable[torch.Tensor]) -> int:
    """Read one element per page of every tensor so its pages are mapped into the process; returns bytes touched."""
    touched = 0
    with torch.no_grad():
        for tensor in tensors:
            if not tensor.is_contiguous():
                continue
            # A byte view strided by the page size reads one byte per page; the sum is discarded.
            as_bytes = tensor.view(torch.uint8).reshape(-1)
            as_bytes[::_PAGE].sum()
            touched += as_bytes.numel()
    return touched


def prefetch_module_weights(module: torch.nn.Module, *, label: str = "") -> None:
    """Synchronously prefetch the module's CPU-resident weights into the file cache and map them in."""
    started = time.perf_counter()
    ranges = collect_cpu_weight_ranges(module)
    total = sum(length for _, length in ranges)
    ok = prefetch_ranges(ranges)
    prefetched_at = time.perf_counter()
    touched = touch_cpu_weights(_cpu_weight_tensors(module))
    logger.debug(
        "Weight prefetch {}: {} MB in {} range(s), accepted={}, {:.2f}s; touched {} MB in {:.2f}s",
        label or module.__class__.__name__,
        total // (1024 * 1024),
        len(ranges),
        ok,
        prefetched_at - started,
        touched // (1024 * 1024),
        time.perf_counter() - prefetched_at,
    )


def prefetch_module_weights_async(module: torch.nn.Module, *, label: str = "") -> threading.Thread | None:
    """Prefetch the module's CPU-resident weights on a daemon thread; returns the thread, or None if skipped.

    At most one prefetch runs per module at a time; a request that finds one in flight is dropped, since the
    running one already covers the same pages.
    """
    if weight_prefetch_disabled():
        return None
    key = id(module)
    with _lock:
        if key in _in_flight:
            return None
        _in_flight.add(key)

    def _run() -> None:
        try:
            prefetch_module_weights(module, label=label)
        except Exception as exc:
            logger.debug("Weight prefetch {} raised: {}", label, exc)
        finally:
            with _lock:
                _in_flight.discard(key)

    thread = threading.Thread(target=_run, name=f"hordelib-weight-prefetch-{label or key}", daemon=True)
    thread.start()
    return thread
