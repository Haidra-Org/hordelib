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
import sys

from loguru import logger


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
