"""Return freed heap and cold mmap-faulted pages to the operating system.

An allocator (glibc ``malloc``) and the OS working-set manager hold on to pages a process once touched
even after the process frees the underlying objects: freed heap arenas are kept for reuse and cold pages
that were faulted in (for example the mmap'd pages of a zero-copy checkpoint load) stay resident until
something forces them out. Nothing in the ordinary ``gc.collect()`` plus device-cache-clear path asks the
OS to reclaim any of that, so a long-lived inference process ratchets its measured resident set upward
across model hops and per-job weight patching while its live data stays flat.

This module issues the platform's release request so the process's measured residency reflects live data
again: ``malloc_trim`` on glibc (Linux) hands unused arena pages back to the kernel, and
``EmptyWorkingSet`` on Windows trims the process working set so cold pages are unmapped. Reclaimed cold
pages refault on demand the next time they are touched, so a trim is only free at an unload or idle
boundary; calling it mid-inference would just pay to evict pages the very next job faults straight back in.

Best-effort and silent: every failure path (unsupported platform, missing symbol, a refusing OS) returns
False and is debug-logged, never raised, so a caller can trim unconditionally at a boundary without
guarding the call.
"""

from __future__ import annotations

import ctypes
import gc
import sys
import threading
import time

from loguru import logger

COMPONENT_RELEASE_TRIM_MIN_INTERVAL_SECONDS = 90.0
"""Minimum wall-time gap between two component-release trims.

The disaggregated encode workload alternates between a small set of hot models, so a component cache
eviction or single-slot replacement fires many times per minute. Trimming on every one would pay an
``EmptyWorkingSet`` per swap and dump hot pages the very next encode faults straight back in, so
:func:`trim_host_after_component_release` collapses a swap storm to at most one trim per this interval.
"""

_component_release_trim_lock = threading.Lock()
_last_component_release_trim_monotonic: float | None = None


def trim_host_after_component_release() -> bool:
    """Reclaim host pages freed by a component cache eviction, at most once per throttle interval.

    Call this at a component cache eviction or single-slot replacement boundary, after the cache has
    dropped its own reference to the displaced component. It runs a ``gc.collect()`` and then
    :func:`trim_host_memory`, spaced at least :data:`COMPONENT_RELEASE_TRIM_MIN_INTERVAL_SECONDS` apart so
    a swap storm cannot thrash the working set.

    This reclaims only pages of components the comfy model-management layer has already released: while a
    component stays resident in ``comfy.model_management.current_loaded_models`` it keeps a strong reference
    to the ModelPatcher, so dropping the cache entry alone leaves its pages live and this trim skips them.
    Guaranteed reclaim of every held component is the worker-driven RAM unload
    (:func:`hordelib.comfy_horde.unload_all_models_ram`), which frees the comfy loaded set first.

    Returns True when a trim was issued this call, False when the throttle suppressed it or the platform
    release request did not run. Best-effort: never raises.
    """
    now = time.monotonic()
    global _last_component_release_trim_monotonic
    with _component_release_trim_lock:
        last_trim = _last_component_release_trim_monotonic
        if last_trim is not None and (now - last_trim) < COMPONENT_RELEASE_TRIM_MIN_INTERVAL_SECONDS:
            return False
        _last_component_release_trim_monotonic = now
    try:
        gc.collect()
        return trim_host_memory()
    except Exception as trim_error:
        logger.debug(f"Component-release trim was not performed: {trim_error}")
        return False


def trim_host_memory() -> bool:
    """Ask the OS to reclaim this process's freed heap and cold mmap-faulted pages.

    Returns True when the platform's release request was issued successfully, False on any other outcome
    (a platform with no such request, a missing symbol, or an OS that refused). Never raises.

    Callers should invoke this at unload or idle boundaries only: reclaimed cold pages refault on demand,
    so trimming during active inference merely evicts pages the next job pulls back in.
    """
    platform = sys.platform
    try:
        if platform.startswith("linux"):
            libc = ctypes.CDLL("libc.so.6", use_last_error=True)
            libc.malloc_trim.argtypes = [ctypes.c_size_t]
            libc.malloc_trim.restype = ctypes.c_int
            libc.malloc_trim(0)
            return True
        if platform == "win32":
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            psapi = ctypes.WinDLL("psapi", use_last_error=True)
            kernel32.GetCurrentProcess.restype = ctypes.c_void_p
            psapi.EmptyWorkingSet.argtypes = [ctypes.c_void_p]
            psapi.EmptyWorkingSet.restype = ctypes.c_int
            handle = kernel32.GetCurrentProcess()
            return bool(psapi.EmptyWorkingSet(handle))
    except Exception as exc:
        logger.debug(f"Host memory trim was not performed on {platform}: {exc}")
        return False

    # macOS and any other platform: no portable per-process release request, so there is nothing to do.
    logger.debug(f"Host memory trim is a no-op on {platform}")
    return False
