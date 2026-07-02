"""Always-on per-job phase timing into the metrics collector.

The logfire instrumentation in ``logfire_comfy_internals`` already times model loads, VAE
operations and sampling, but it is only applied when logfire is configured — which a host
worker explicitly disables (``HORDELIB_EXTERNAL_LOGFIRE=1``). That left the in-process
collector blind to RAM->VRAM transfers and VAE decode/encode, so a host could see sampling
time but not the non-sampling phases between jobs.

This module patches the same ComfyUI internals *independently of logfire* and records the
durations into :class:`hordelib.metrics.MetricsCollector`, so any embedder can see where a
job's wall-clock goes. It is installed once from the execution backend's ``start()`` and is
idempotent; the collector recording for model loads lives here (not in the logfire hook) so
it happens exactly once whether or not logfire is active.
"""

from __future__ import annotations

import time

from loguru import logger

from hordelib.metrics import ModelLoadEvent, get_metrics_collector

# load_models_gpu runs on every sampling pass and is a fast no-op when models are resident;
# only transfers slower than this are real RAM->VRAM moves worth recording as load events.
_GPU_LOAD_RECORD_THRESHOLD_SECONDS = 0.05

_installed = False


def install_phase_timing_hooks() -> bool:
    """Patch ComfyUI model-load and VAE ops to record phase durations into the collector.

    Idempotent. Returns True if the hooks are in place (now or already), False if ComfyUI
    internals could not be patched.
    """
    global _installed
    if _installed:
        return True

    patched_any = False
    patched_any |= _patch_model_load()
    patched_any |= _patch_vae()
    patched_any |= _patch_clip()

    if patched_any:
        _installed = True
        logger.debug("Installed always-on phase-timing hooks")
    return patched_any


def _patch_model_load() -> bool:
    try:
        from comfy import model_management as mm
    except Exception as e:
        logger.warning(f"Phase timing: could not patch model_management ({e})")
        return False

    if getattr(mm.load_models_gpu, "_hordelib_phase_timed", False):
        return True

    original = mm.load_models_gpu

    def _timed_load_models_gpu(*args: object, **kwargs: object) -> object:
        start = time.perf_counter()
        result = original(*args, **kwargs)
        duration = time.perf_counter() - start
        if duration >= _GPU_LOAD_RECORD_THRESHOLD_SECONDS:
            get_metrics_collector().record_model_load(
                ModelLoadEvent(
                    model_name="ram_to_vram",
                    phase="ram_to_vram",
                    duration_seconds=duration,
                    timestamp=time.time(),
                ),
            )
        return result

    _timed_load_models_gpu._hordelib_phase_timed = True  # type: ignore[attr-defined]
    mm.load_models_gpu = _timed_load_models_gpu
    return True


def _patch_clip() -> bool:
    try:
        from comfy import sd as sd_module
    except Exception as e:
        logger.warning(f"Phase timing: could not patch CLIP ({e})")
        return False

    clip_cls = getattr(sd_module, "CLIP", None)
    if clip_cls is None:
        return False

    # encode_from_tokens is the core text-encode both CLIPTextEncode paths funnel through; timing
    # it (rather than the node) captures prompt encoding regardless of the scheduled wrapper.
    original = getattr(clip_cls, "encode_from_tokens", None)
    if original is None or getattr(original, "_hordelib_phase_timed", False):
        return original is not None

    def _timed(self: object, *args: object, **kwargs: object) -> object:
        start = time.perf_counter()
        result = original(self, *args, **kwargs)  # type: ignore[operator]
        get_metrics_collector().record_phase("clip_encode", time.perf_counter() - start)
        return result

    _timed._hordelib_phase_timed = True  # type: ignore[attr-defined]
    clip_cls.encode_from_tokens = _timed
    return True


def _patch_vae() -> bool:
    try:
        from comfy import sd as sd_module
    except Exception as e:
        logger.warning(f"Phase timing: could not patch VAE ({e})")
        return False

    vae_cls = getattr(sd_module, "VAE", None)
    if vae_cls is None:
        return False

    for method_name, phase_name in (("decode", "vae_decode"), ("encode", "vae_encode")):
        original = getattr(vae_cls, method_name, None)
        if original is None or getattr(original, "_hordelib_phase_timed", False):
            continue

        def _make_timed(orig: object, phase: str) -> object:
            def _timed(self: object, *args: object, **kwargs: object) -> object:
                start = time.perf_counter()
                result = orig(self, *args, **kwargs)  # type: ignore[operator]
                get_metrics_collector().record_phase(phase, time.perf_counter() - start)
                return result

            _timed._hordelib_phase_timed = True  # type: ignore[attr-defined]
            return _timed

        setattr(vae_cls, method_name, _make_timed(original, phase_name))

    return True
