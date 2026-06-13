"""Cross-process GPU sampling lease.

A single GPU can only meaningfully run one denoising loop at a time, but the *rest* of a
job's pipeline — loading the model into VRAM, encoding the prompt, building latents — is
GPU-light and can run while another process samples. Left uncoordinated, multiple worker
inference processes tend to fall into lockstep: they all sample together, then all do their
setup/decode together, leaving the GPU idle during the shared setup window.

This module lets the host (e.g. the AI-Horde worker) inject a cross-process semaphore that
hordelib acquires around ``comfy.sample.sample``. In current ComfyUI that call brackets the
diffusion model's load-to-VRAM (``load_models_gpu``) *and* the denoise loop — but not the
upstream checkpoint disk load or prompt encoding, nor the downstream VAE decode. With more
inference processes than lease slots, one process samples while the others stage their next
pipeline (checkpoint load, prompt encode) up to the sampling call; when the sampler finishes,
an already-staged process takes the lease and samples immediately — keeping the GPU busy
back-to-back. Because the VRAM load sits inside the lease, this only helps when models are kept
resident (the host's high-memory / no-unload mode); otherwise it serializes the RAM->VRAM
transfer behind sampling.

When no lease is set (standalone hordelib, or the feature disabled) the wrapper is a no-op.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from loguru import logger


@runtime_checkable
class SamplingLease(Protocol):
    """The subset of ``multiprocessing.Semaphore`` / ``threading.Semaphore`` we rely on."""

    def acquire(self, block: bool = ..., timeout: float | None = ...) -> bool:
        """Acquire the lease, optionally with a timeout; returns whether it was acquired."""
        ...

    def release(self) -> None:
        """Release the lease."""
        ...


_sampling_lease: SamplingLease | None = None
_acquire_timeout_seconds: float = 120.0
_installed: bool = False


def set_gpu_sampling_lease(lease: SamplingLease | None, *, acquire_timeout_seconds: float = 120.0) -> None:
    """Register (or clear) the cross-process lease hordelib holds around the sampling loop.

    Args:
        lease: A semaphore-like object shared across the host's inference processes, or None
            to disable coordination.
        acquire_timeout_seconds: Safety cap on the lease wait. If acquisition times out the
            job samples anyway (degraded coordination beats a hung worker).
    """
    global _sampling_lease, _acquire_timeout_seconds
    _sampling_lease = lease
    _acquire_timeout_seconds = acquire_timeout_seconds
    if lease is not None:
        logger.info(f"GPU sampling lease registered (acquire timeout {acquire_timeout_seconds:.0f}s)")


def install_sampling_lease_hook() -> bool:
    """Monkey-patch ``comfy.sample.sample`` to hold the lease around the denoising loop.

    Idempotent and independent of logfire instrumentation. Returns True if the hook is in
    place (now or already), False if ComfyUI's sample module could not be patched.
    """
    global _installed
    if _installed:
        return True

    try:
        from comfy import sample as sample_module
    except Exception as e:
        logger.warning(f"Could not install GPU sampling lease hook: {e}")
        return False

    original_sample = sample_module.sample

    def _leased_sample(*args: object, **kwargs: object) -> object:
        lease = _sampling_lease
        acquired = False
        if lease is not None:
            try:
                acquired = bool(lease.acquire(True, _acquire_timeout_seconds))
            except Exception as e:
                logger.warning(f"GPU sampling lease acquire failed ({e}); sampling without it")
                acquired = False
            if not acquired:
                logger.warning("GPU sampling lease acquire timed out; sampling without it")
        try:
            return original_sample(*args, **kwargs)
        finally:
            if acquired and lease is not None:
                try:
                    lease.release()
                except Exception as e:
                    logger.error(f"Failed to release GPU sampling lease: {e}")

    sample_module.sample = _leased_sample
    _installed = True
    logger.debug("Installed GPU sampling lease hook on comfy.sample.sample")
    return True
