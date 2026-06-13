"""Native sampling progress via ComfyUI's global progress-bar hook.

Historically hordelib learned about sampling progress by scraping tqdm output from
ComfyUI's stdout (:class:`hordelib.utils.ioredirect.OutputCollector`), which breaks
whenever tqdm formatting shifts. ComfyUI exposes a first-class seam instead:
``comfy.utils.set_progress_bar_global_hook`` — every ``ProgressBar.update_absolute``
call (throttled upstream to 100ms / 0.5% between regular updates, with first and final
updates always delivered) invokes the hook with absolute step and total values.

This module installs that hook once per process (``ProgressBar`` captures the hook at
construction, so installation must precede pipeline execution — the in-process backend
installs it at ``start()``). Each invocation feeds the in-process
:class:`~hordelib.metrics.MetricsCollector` (step/rate accounting plus VRAM/RAM
high-water sampling) and forwards a :class:`~hordelib.utils.ioredirect.ComfyUIProgress`
to the per-run callback registered by ``Comfy_Horde._run_pipeline``. When the native
hook is live, the tqdm parser's callback channel is suppressed so consumers see exactly
one progress stream; the tqdm path remains as a fallback for processes that never
install the hook.
"""

import threading
import time
from collections.abc import Callable
from typing import Any

import psutil
from loguru import logger

from hordelib.metrics import get_metrics_collector
from hordelib.utils.ioredirect import ComfyUIProgress, ComfyUIProgressUnit
from hordelib.utils.torch_memory import get_torch_free_vram_mb, get_torch_total_vram_mb

_MB = 1024 * 1024

_lock = threading.Lock()
_hook_installed = False
_run_callback: Callable[[ComfyUIProgress, str], None] | None = None
_last_value: int | None = None
_last_timestamp: float | None = None


def install_native_progress_hook() -> None:
    """Install the global ComfyUI progress hook (idempotent; requires comfy imported)."""
    global _hook_installed
    with _lock:
        if _hook_installed:
            return
        import comfy.utils

        comfy.utils.set_progress_bar_global_hook(_native_hook)
        _hook_installed = True
    logger.debug("Native ComfyUI progress hook installed")


def is_native_hook_installed() -> bool:
    """Whether the native progress hook has been installed in this process."""
    return _hook_installed


def set_run_progress_callback(callback: Callable[[ComfyUIProgress, str], None] | None) -> None:
    """Register (or clear) the callback receiving native progress for the current run."""
    global _run_callback, _last_value, _last_timestamp
    with _lock:
        _run_callback = callback
        _last_value = None
        _last_timestamp = None


def _native_hook(value: int, total: int, preview: Any = None, node_id: Any = None) -> None:
    """Receive one absolute progress update from any ComfyUI ProgressBar."""
    global _last_value, _last_timestamp
    now = time.time()

    collector = get_metrics_collector()
    collector.record_sampling_step(value, total, now)

    total_vram_mb = get_torch_total_vram_mb()
    if total_vram_mb > 0:
        collector.record_memory_sample(
            vram_used_mb=total_vram_mb - get_torch_free_vram_mb(),
            ram_used_mb=round(psutil.virtual_memory().used / _MB),
        )

    with _lock:
        callback = _run_callback
        # -1.0 means "rate not yet known", matching the tqdm parser's convention for "?".
        rate = -1.0
        if _last_value is not None and _last_timestamp is not None and value > _last_value and now > _last_timestamp:
            rate = (value - _last_value) / (now - _last_timestamp)
        _last_value = value
        _last_timestamp = now

    if callback is None:
        return

    progress = ComfyUIProgress(
        percent=round((value / total) * 100) if total > 0 else 0,
        current_step=value,
        total_steps=total,
        rate=round(rate, 2),
        rate_unit=ComfyUIProgressUnit.ITERATIONS_PER_SECOND,
        source="native",
    )
    try:
        callback(progress, str(progress))
    except Exception:
        logger.exception("Native progress callback failed")
