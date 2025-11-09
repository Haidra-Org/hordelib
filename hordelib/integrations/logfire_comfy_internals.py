# type: ignore - Monkeypatching is inherently unsafe
"""
Deep ComfyUI internals instrumentation for Logfire.

This module monkey-patches critical ComfyUI internal functions to provide
comprehensive tracing and metrics with minimal overhead and zero logic duplication.
"""

import time

import logfire

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
        instrumented_count += _instrument_model_patching()

        logfire.info("comfy.instrumentation.applied", functions_instrumented=instrumented_count)
    except Exception as e:
        logfire.error("comfy.instrumentation.failed", error=str(e))
        raise


def _instrument_execution_flow() -> int:
    """Instrument execution.py functions."""
    count = 0

    try:
        from ComfyUI import execution

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
        logfire.warn("comfy.instrumentation.execution_not_found", module="execution")
    except Exception as e:
        logfire.warn("comfy.instrumentation.execution_failed", error=str(e))

    return count


def _instrument_model_management() -> int:
    """Instrument model_management.py functions."""
    count = 0

    try:
        from ComfyUI.comfy import model_management as mm

        # Instrument load_models_gpu()
        if hasattr(mm, "load_models_gpu"):
            _original_load_models_gpu = mm.load_models_gpu

            def _instrumented_load_models_gpu(*args, **kwargs):
                start_time = time.perf_counter()

                with logfire.span("comfy.internal.load_models_gpu"):
                    result = _original_load_models_gpu(*args, **kwargs)

                    duration_ms = (time.perf_counter() - start_time) * 1000
                    comfy_model_load_histogram.record(duration_ms)

                    return result

            mm.load_models_gpu = _instrumented_load_models_gpu
            count += 1

        # Instrument free_memory()
        if hasattr(mm, "free_memory"):
            _original_free_memory = mm.free_memory

            def _instrumented_free_memory(*args, **kwargs):
                comfy_memory_pressure_counter.add(1)

                with logfire.span("comfy.internal.free_memory"):
                    result = _original_free_memory(*args, **kwargs)
                    return result

            mm.free_memory = _instrumented_free_memory
            count += 1

        # Instrument get_free_memory() - lightweight gauge only, no span
        if hasattr(mm, "get_free_memory"):
            _original_get_free_memory = mm.get_free_memory

            def _instrumented_get_free_memory(*args, **kwargs):
                free_mem = _original_get_free_memory(*args, **kwargs)
                comfy_vram_free_gauge.set(free_mem / (1024 * 1024))
                return free_mem

            mm.get_free_memory = _instrumented_get_free_memory
            count += 1

        # Instrument unload_model_clones()
        if hasattr(mm, "unload_model_clones"):
            _original_unload_model_clones = mm.unload_model_clones

            def _instrumented_unload_model_clones(*args, **kwargs):
                with logfire.span("comfy.internal.unload_model_clones"):
                    result = _original_unload_model_clones(*args, **kwargs)
                    return result

            mm.unload_model_clones = _instrumented_unload_model_clones
            count += 1

    except ImportError:
        logfire.warn("comfy.instrumentation.model_management_not_found")
    except Exception as e:
        logfire.warn("comfy.instrumentation.model_management_failed", error=str(e))

    return count


def _instrument_sampling() -> int:
    """Instrument sample.py functions."""
    count = 0

    try:
        from ComfyUI.comfy import sample as sample_module

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
                    result = _original_calc_cond_batch(*args, **kwargs)
                    return result

            sample_module.calc_cond_batch = _instrumented_calc_cond_batch
            count += 1

    except ImportError:
        logfire.warn("comfy.instrumentation.sample_not_found")
    except Exception as e:
        logfire.warn("comfy.instrumentation.sample_failed", error=str(e))

    return count


def _instrument_vae_operations() -> int:
    """Instrument VAE encode/decode."""
    count = 0

    try:
        from ComfyUI.comfy import sd as sd_module

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
        logfire.warn("comfy.instrumentation.vae_not_found")
    except Exception as e:
        logfire.warn("comfy.instrumentation.vae_failed", error=str(e))

    return count


def _instrument_model_patching() -> int:
    """Instrument model_patcher.py functions."""
    count = 0

    try:
        from ComfyUI.comfy import model_patcher

        if hasattr(model_patcher, "ModelPatcher"):
            ModelPatcher = model_patcher.ModelPatcher

            if hasattr(ModelPatcher, "patch_weight_to_device"):
                _original_patch_weight = ModelPatcher.patch_weight_to_device

                def _instrumented_patch_weight(self, *args, **kwargs):
                    with logfire.span("comfy.internal.patch_weight"):
                        result = _original_patch_weight(self, *args, **kwargs)
                        return result

                ModelPatcher.patch_weight_to_device = _instrumented_patch_weight
                count += 1

    except ImportError:
        logfire.warn("comfy.instrumentation.model_patcher_not_found")
    except Exception as e:
        logfire.warn("comfy.instrumentation.model_patcher_failed", error=str(e))

    return count
