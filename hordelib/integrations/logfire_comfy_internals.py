# mypy: ignore-errors
# Monkeypatching is inherently type-unsafe
"""
Deep ComfyUI internals instrumentation for Logfire.

This module monkey-patches critical ComfyUI internal functions to provide
comprehensive tracing and metrics with minimal overhead and zero logic duplication.
"""

import time

import logfire
from loguru import logger

from hordelib.metrics import ModelLoadEvent, get_metrics_collector

# load_models_gpu is called on every sampling pass and is usually a fast no-op when the
# models are already resident; only transfers slower than this are real RAM->VRAM moves
# worth recording as per-job model-load events (the logfire histogram still sees all calls).
_GPU_LOAD_RECORD_THRESHOLD_SECONDS = 0.05

# Module-level metrics
comfy_node_execution_histogram = logfire.metric_histogram(
    "comfy.node.execution_duration_ms",
    unit="ms",
    description="Per-node execution duration in ComfyUI graph",
)

comfy_model_load_histogram = logfire.metric_histogram(
    "comfy.model_load.duration_ms",
    unit="ms",
    description="Time to load models to GPU",
)

comfy_ram_free_gauge = logfire.metric_gauge(
    "comfy.ram.free_mb",
    unit="MB",
    description="Current free system RAM",
)

comfy_vram_free_gauge = logfire.metric_gauge(
    "comfy.vram.free_mb",
    unit="MB",
    description="Current free VRAM on GPU device",
)

comfy_memory_pressure_counter = logfire.metric_counter(
    "comfy.memory_pressure.events",
    unit="1",
    description="Number of memory pressure events",
)

comfy_sampling_duration_histogram = logfire.metric_histogram(
    "comfy.sampling.duration_ms",
    unit="ms",
    description="Total sampling duration",
)

comfy_vae_decode_histogram = logfire.metric_histogram(
    "comfy.vae.decode_duration_ms",
    unit="ms",
    description="VAE decode duration",
)

comfy_vae_encode_histogram = logfire.metric_histogram(
    "comfy.vae.encode_duration_ms",
    unit="ms",
    description="VAE encode duration",
)


def instrument_comfy_internals():
    """Apply monkey-patches to ComfyUI internal functions."""
    try:
        instrumented_count = 0

        instrumented_count += _instrument_execution_flow()
        instrumented_count += _instrument_model_management()
        instrumented_count += _instrument_sampling()
        instrumented_count += _instrument_vae_operations()

        logger.info("comfy.instrumentation.applied", functions_instrumented=instrumented_count)
    except Exception as e:
        logger.error("comfy.instrumentation.failed", error=str(e))
        raise


def _instrument_execution_flow() -> int:
    """Instrument execution.py functions."""
    count = 0

    try:
        import execution

        if hasattr(execution, "execute"):
            _original_execute = execution.execute

            async def _instrumented_execute(*args, **kwargs):
                start_time = time.perf_counter()

                with logfire.span("comfy.internal.execute_node"):
                    result = await _original_execute(*args, **kwargs)

                    duration_ms = (time.perf_counter() - start_time) * 1000
                    comfy_node_execution_histogram.record(duration_ms)

                    return result

            execution.execute = _instrumented_execute
            count += 1

    except ImportError:
        logger.warning("comfy.instrumentation.execution_not_found", module="execution")
    except Exception as e:
        logger.warning("comfy.instrumentation.execution_failed", error=str(e))

    return count


def _describe_loaded_models(args, kwargs) -> str:
    """Best-effort summary of which models a load_models_gpu call moved to the GPU."""
    models = kwargs.get("models", args[0] if args else None)
    try:
        names = {type(getattr(m, "model", m)).__name__ for m in models}
        return ",".join(sorted(names)) if names else "unknown"
    except Exception:
        return "unknown"


def _instrument_model_management() -> int:
    """Instrument model_management.py functions."""
    count = 0

    try:
        from comfy import model_management as mm

        # Instrument load_models_gpu()
        if hasattr(mm, "load_models_gpu"):
            _original_load_models_gpu = mm.load_models_gpu

            def _instrumented_load_models_gpu(*args, **kwargs):
                start_time = time.perf_counter()

                with logfire.span("comfy.internal.load_models_gpu"):
                    result = _original_load_models_gpu(*args, **kwargs)

                    duration_seconds = time.perf_counter() - start_time
                    comfy_model_load_histogram.record(duration_seconds * 1000)

                    if duration_seconds >= _GPU_LOAD_RECORD_THRESHOLD_SECONDS:
                        get_metrics_collector().record_model_load(
                            ModelLoadEvent(
                                model_name=_describe_loaded_models(args, kwargs),
                                phase="ram_to_vram",
                                duration_seconds=duration_seconds,
                                timestamp=time.time(),
                            ),
                        )

                    return result

            mm.load_models_gpu = _instrumented_load_models_gpu
            count += 1

        # Instrument free_memory()
        if hasattr(mm, "free_memory"):
            _original_free_memory = mm.free_memory

            def _instrumented_free_memory(*args, **kwargs):
                comfy_memory_pressure_counter.add(1)

                with logfire.span("comfy.internal.free_memory"):
                    return _original_free_memory(*args, **kwargs)

            mm.free_memory = _instrumented_free_memory
            count += 1

        # Instrument get_free_memory() - lightweight gauge only, no span
        if hasattr(mm, "get_free_memory"):
            _original_get_free_memory = mm.get_free_memory

            def _instrumented_get_free_memory(*args, **kwargs):
                result = _original_get_free_memory(*args, **kwargs)

                mem_free_total = None
                mem_free_torch = None

                if not isinstance(mem_free_total, (int, float)) and isinstance(result, (list, tuple)):
                    mem_free_total, mem_free_torch = result

                if mem_free_total is not None:
                    comfy_ram_free_gauge.set(mem_free_total / (1024 * 1024))
                if mem_free_torch is not None:
                    comfy_vram_free_gauge.set(mem_free_torch / (1024 * 1024))

                return result

            mm.get_free_memory = _instrumented_get_free_memory
            count += 1

        # Instrument unload_model_clones()
        if hasattr(mm, "unload_model_clones"):
            _original_unload_model_clones = mm.unload_model_clones

            def _instrumented_unload_model_clones(*args, **kwargs):
                with logfire.span("comfy.internal.unload_model_clones"):
                    return _original_unload_model_clones(*args, **kwargs)

            mm.unload_model_clones = _instrumented_unload_model_clones
            count += 1

    except ImportError:
        logger.warning("comfy.instrumentation.model_management_not_found")
    except Exception as e:
        logger.warning("comfy.instrumentation.model_management_failed", error=str(e))

    return count


def _instrument_sampling() -> int:
    """Instrument sample.py functions."""
    count = 0

    try:
        from comfy import sample as sample_module

        # Instrument sample()
        if hasattr(sample_module, "sample"):
            _original_sample = sample_module.sample

            def _instrumented_sample(*args, **kwargs):
                start_time = time.perf_counter()

                with logfire.span("comfy.internal.sample"):
                    result = _original_sample(*args, **kwargs)

                    duration_ms = (time.perf_counter() - start_time) * 1000
                    comfy_sampling_duration_histogram.record(duration_ms)

                    return result

            sample_module.sample = _instrumented_sample
            count += 1

        # Instrument calc_cond_batch()
        if hasattr(sample_module, "calc_cond_batch"):
            _original_calc_cond_batch = sample_module.calc_cond_batch

            def _instrumented_calc_cond_batch(*args, **kwargs):
                with logfire.span("comfy.internal.calc_cond_batch"):
                    return _original_calc_cond_batch(*args, **kwargs)

            sample_module.calc_cond_batch = _instrumented_calc_cond_batch
            count += 1

    except ImportError:
        logger.warning("comfy.instrumentation.sample_not_found")
    except Exception as e:
        logger.warning("comfy.instrumentation.sample_failed", error=str(e))

    return count


def _instrument_vae_operations() -> int:
    """Instrument VAE encode/decode."""
    count = 0

    try:
        from comfy import sd as sd_module

        if hasattr(sd_module, "VAE"):
            VAE = sd_module.VAE

            if hasattr(VAE, "decode"):
                _original_vae_decode = VAE.decode

                def _instrumented_vae_decode(self, *args, **kwargs):
                    start_time = time.perf_counter()

                    with logfire.span("comfy.internal.vae_decode"):
                        result = _original_vae_decode(self, *args, **kwargs)

                        duration_ms = (time.perf_counter() - start_time) * 1000
                        comfy_vae_decode_histogram.record(duration_ms)

                        return result

                VAE.decode = _instrumented_vae_decode
                count += 1

            if hasattr(VAE, "encode"):
                _original_vae_encode = VAE.encode

                def _instrumented_vae_encode(self, *args, **kwargs):
                    start_time = time.perf_counter()

                    with logfire.span("comfy.internal.vae_encode"):
                        result = _original_vae_encode(self, *args, **kwargs)

                        duration_ms = (time.perf_counter() - start_time) * 1000
                        comfy_vae_encode_histogram.record(duration_ms)

                        return result

                VAE.encode = _instrumented_vae_encode
                count += 1

    except ImportError:
        logger.warning("comfy.instrumentation.vae_not_found")
    except Exception as e:
        logger.warning("comfy.instrumentation.vae_failed", error=str(e))

    return count
