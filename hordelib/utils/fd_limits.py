"""Raise the process's open-file soft limit, the sanctioned guard against descriptor-sharing exhaustion.

Sharing tensors across processes with PyTorch (``torch.multiprocessing``) caches one file descriptor per
shared storage under the default ``file_descriptor`` strategy, so a process that ships many tensors across
a boundary accumulates descriptors and can reach its per-process ``RLIMIT_NOFILE`` ceiling. At the ceiling
every ``open()`` is refused with ``EMFILE`` ("Too many open files"), which in an inference process makes
model/LoRA loads and even routine ``/proc`` reads fail.

PyTorch's documented preference is to keep the (faster) ``file_descriptor`` strategy and raise the limit,
falling back to the ``file_system`` strategy only where the limit cannot be raised (that strategy avoids
caching descriptors but leaks shared-memory files on abnormal exit, needing the ``torch_shm_manager``
daemon to reap them). So this raise is the primary mitigation, not a workaround.

POSIX-only: Windows has no ``RLIMIT_NOFILE`` and an effectively unbounded handle count, so the call is a
no-op there. Best-effort throughout: a platform that refuses the change is logged and left as-is.
"""

from __future__ import annotations

from loguru import logger

try:
    import resource
except ImportError:  # Windows has no ``resource`` module (and no RLIMIT_NOFILE): every path below no-ops.
    resource = None  # type: ignore[assignment]


def raise_open_file_soft_limit() -> tuple[int, int] | None:
    """Raise this process's soft ``RLIMIT_NOFILE`` to its hard ceiling.

    Returns the ``(old_soft, new_soft)`` pair when the limit was raised, or None when there was nothing to
    do (already at the ceiling, or no such limit on this platform). Child processes spawned afterwards
    inherit the raised soft limit, so calling this before spawning the sharing children hardens them all.
    """
    if resource is None:
        return None
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)  # type: ignore[attr-defined]  # POSIX-only, guarded above
    except (ValueError, OSError):
        return None
    if soft == hard:
        return None
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))  # type: ignore[attr-defined]  # POSIX-only, guarded above
    except (ValueError, OSError) as exc:
        logger.debug(f"Could not raise the open-file soft limit from {soft}: {exc}")
        return None
    logger.debug(f"Raised open-file soft limit {soft} -> {hard} to harden cross-process tensor sharing")
    return (soft, hard)
