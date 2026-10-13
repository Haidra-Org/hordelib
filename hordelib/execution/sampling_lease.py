"""Cross-process GPU sampling lease.

A single GPU can only meaningfully run one denoising loop at a time, but the *rest* of a
job's pipeline — loading the model into VRAM, encoding the prompt, building latents — is
GPU-light and can run while another process samples. Left uncoordinated, multiple worker
inference processes tend to fall into lockstep: they all sample together, then all do their
setup/decode together, leaving the GPU idle during the shared setup window.

This module lets the host (e.g. the AI-Horde worker) inject a cross-process semaphore that
hordelib acquires around ``comfy.samplers.CFGGuider.sample``. Every ComfyUI sampling path reaches that
method: ``KSampler``-family nodes through ``comfy.sample.sample``, ``SamplerCustom`` through
``comfy.sample.sample_custom``, and ``SamplerCustomAdvanced`` by calling ``guider.sample`` on a
``CFGGuider`` subclass directly. Hooking either ``comfy.sample`` entry point would leave the last path
sampling outside the lease. ``CFGGuider.sample`` brackets the diffusion model's load-to-VRAM
(``prepare_sampling`` in ``outer_sample``) *and* the denoise loop, but not the upstream checkpoint
disk load or prompt encoding, nor the downstream VAE decode. With more inference processes than lease
slots, one process samples while the others stage their next pipeline (checkpoint load, prompt encode)
up to the sampling call; when the sampler finishes, an already-staged process takes the lease and
samples immediately, keeping the GPU busy back-to-back. Because the VRAM load sits inside the lease,
this only helps when models are kept resident (the host's high-memory / no-unload mode); otherwise it
serializes the RAM->VRAM transfer behind sampling.

A host that also sets a stage lease (``set_gpu_stage_lease``) splits that window in two. The hook takes the
stage lease first and, on its grant, uploads the guider's model with the call ``prepare_sampling`` makes, then
releases it and takes the sampling lease. The later ``prepare_sampling`` finds the model loaded, so the sampling
lease covers only the denoise loop and its onset. The stage permit covers that upload alone. A host never grants a
stage permit without a sampling permit following, and both acquires share one deadline of the acquire timeout, so
the longest wait is unchanged. With no stage lease set the path is the single acquire above.

A host whose grant also carries the activation room it priced for the window (:class:`ClearanceReserveLease`) bounds
ComfyUI's memory estimate for the sampling model's load. ComfyUI sizes a partial load to leave only its own activation
estimate, which can be smaller than the room the host priced. The hook wraps ``comfy.sampler_helpers.estimate_memory``,
so both the staged upload and ``prepare_sampling`` pass the larger of the two figures. The bound applies only on the
thread of a hooked sample after a grant that carried the figure. It applies to every sampling load, so on a forced full
load ComfyUI frees other resident models toward it, leaving the room the host priced. The force-full overflow check
also charges the held figure (:func:`held_sampling_reserve_mb`), so the offloaded remainder of a bounded staged load
is not force-loaded into that room at sampling.

With a sampling lease set, the hook first returns every text encoder on the sampling device to the host, whole,
unless a node of the running prompt that has not finished takes a ``CLIP`` input. By then the encode is done and the
guider holds only the conditioning tensors it produced. An encoder left on the device holds its room through the grant
wait, which is the room a stage grant needs. The diffusion load's ``free_memory`` then unloads it partially, copying
each module back to the host, pinning the copy and releasing it in ``release_copied_back_weights``. The release is
ComfyUI's own full unload (``free_memory`` with every other loaded model kept), whose ``unpatch_model`` to the CPU
points each weight back at its retained origin (``cpu_weight_retention.restore_cpu_origins``) and copies nothing. The
component cache keeps the encoder in host RAM, so the next job's encode uploads it from there. Whether a later node
takes the encoder is read from the running prompt's graph and per-node progress (``comfy_execution.progress``), so a
pipeline that encodes again after sampling keeps it. Outside a prompt run the remaining graph is unknown, and nothing
is released.

A guider subclass that overrides ``sample`` without calling ``super().sample`` bypasses the lease. A
nested hooked call on the same thread passes through without a second acquire.

When no lease is set (standalone hordelib, or the feature disabled) the wrapper is a no-op.
"""

from __future__ import annotations

import importlib
import threading
from collections.abc import Callable
from time import monotonic
from typing import Any, Protocol, runtime_checkable

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


@runtime_checkable
class ClearanceTruthLease(Protocol):
    """An optional extension a lease implements when its grant carries the host's device-free figure.

    A host that grants the lease after measuring the card has a fresher reading than the one the run's
    free-VRAM clamp was entered with. After a successful acquire the hook asks for it and rebases the
    clamp. A lease without this method, or one that returns None, leaves the clamp as it was.
    """

    def take_cleared_device_free_mb(self) -> float | None:
        """Return the device free VRAM (MB) the grant just acquired was admitted against, once, or None."""
        ...


@runtime_checkable
class ClearanceReserveLease(Protocol):
    """An optional extension a lease implements when its grant carries the host's activation reserve for the window.

    The host prices a sampling window as the weights plus a non-weight term, the job's activation. After a successful
    acquire the hook takes that term and bounds ComfyUI's memory estimate for the sampling model's load with it, so a
    partial load leaves the room the host priced. A lease without this method, or one that returns None, leaves
    ComfyUI's estimate as it is.
    """

    def take_cleared_sampling_reserve_mb(self) -> float | None:
        """Return the activation reserve (MB) the grant just acquired was priced with, once, or None."""
        ...


_sampling_lease: SamplingLease | None = None
_stage_lease: SamplingLease | None = None
_acquire_timeout_seconds: float = 120.0
_installed: bool = False
_lease_scope: threading.local = threading.local()
"""Per-thread state of a hooked guider sample.

``active`` marks one already running, so a nested one does not acquire again. ``sampling_reserve_mb`` is the
latest activation reserve a grant of this sample carried, and ``sampling_reserve_logged`` records that its bound was
logged."""


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


def set_gpu_stage_lease(lease: SamplingLease | None) -> None:
    """Register (or clear) the lease hordelib holds while it uploads the sampling model ahead of the sampling lease.

    The stage acquire shares the sampling lease's acquire timeout: both are taken within one deadline. The hook stages
    only while a sampling lease is also set.

    Args:
        lease: A semaphore-like object the host grants when the model's weights fit on the device, or None to load
            the model inside the sampling lease.
    """
    global _stage_lease
    _stage_lease = lease
    if lease is not None:
        logger.info("GPU stage lease registered (shares the sampling lease acquire timeout)")


def _prefetch_sampling_model_weights(guider: object) -> None:
    """Kick a background page-in of the guider's CPU-resident diffusion weights; never raises."""
    try:
        model_patcher = getattr(guider, "model_patcher", None)
        module = getattr(model_patcher, "model", None)
        if module is None:
            return
        from hordelib.execution.weight_prefetch import prefetch_module_weights_async

        prefetch_module_weights_async(module, label="sample")
    except Exception as exc:
        logger.debug(f"Sampling weight prefetch skipped: {exc}")


_FULL_UNLOAD_BYTES = 1e30
"""The requirement ComfyUI's ``unload_all_models`` passes to ``free_memory``. It exceeds any device's free memory."""

_TEXT_ENCODER_OUTPUT_TYPES = frozenset({"CLIP", "*"})
"""Graph output types that can carry a text encoder: ComfyUI's ``CLIP`` and the wildcard a passthrough declares."""


def _is_loaded_text_encoder(loaded_model: Any, device: object) -> bool:
    """Return whether a ComfyUI ``LoadedModel`` is a text encoder holding weights on ``device``."""
    model_patcher = loaded_model.model
    if model_patcher is None or loaded_model.device != device:
        return False
    return getattr(model_patcher, "is_clip", False) is True and loaded_model.model_loaded_memory() > 0


def _output_type(dynprompt: Any, source_node_id: str, output_index: int) -> str | None:
    """Return the declared type of one node output in the running prompt, or None where it cannot be read."""
    comfy_nodes = importlib.import_module("nodes")
    if not dynprompt.has_node(source_node_id):
        return None
    node_class = comfy_nodes.NODE_CLASS_MAPPINGS.get(dynprompt.get_node(source_node_id).get("class_type"))
    return_types = getattr(node_class, "RETURN_TYPES", None)
    if not isinstance(return_types, (tuple, list)) or not 0 <= output_index < len(return_types):
        return None
    return str(return_types[output_index])


def _node_takes_text_encoder(dynprompt: Any, node_id: str) -> bool:
    """Return whether a node of the running prompt has an input linked to an output that can carry a text encoder.

    A link whose source type cannot be read counts as one, so an unreadable graph keeps the encoder.
    """
    graph_utils = importlib.import_module("comfy_execution.graph_utils")
    for input_value in dynprompt.get_node(node_id).get("inputs", {}).values():
        if not graph_utils.is_link(input_value):
            continue
        source_type = _output_type(dynprompt, input_value[0], input_value[1])
        if source_type is None or source_type in _TEXT_ENCODER_OUTPUT_TYPES:
            return True
    return False


def _text_encoder_still_needed() -> bool:
    """Return whether a node of the running prompt that has not finished takes a text encoder.

    Every node ComfyUI ran or served from its cache is marked finished in the prompt's progress registry. The node
    running now is not, so a sampler node that also takes a text encoder keeps it. Outside a prompt run the remaining
    graph is unknown, and the answer is True.
    """
    progress = importlib.import_module("comfy_execution.progress")
    executing = importlib.import_module("comfy_execution.utils").get_executing_context()
    registry = progress.get_progress_state()
    if executing is None or registry.prompt_id != executing.prompt_id:
        return True
    dynprompt = registry.dynprompt
    for node_id in dynprompt.all_node_ids():
        node_progress = registry.nodes.get(node_id)
        if node_progress is not None and node_progress["state"] is progress.NodeState.Finished:
            continue
        if _node_takes_text_encoder(dynprompt, node_id):
            return True
    return False


def _release_text_encoders_before_wait(guider: object) -> None:
    """Unload each text encoder on the sampling device whole, unless a later node of the prompt takes it; never raises.

    The unload is ``free_memory`` with every other loaded model kept, ComfyUI's full unload. Its ``unpatch_model`` to
    the CPU restores each unpatched weight to its recorded CPU origin, so no weight is copied back from the device.
    """
    try:
        device = getattr(getattr(guider, "model_patcher", None), "load_device", None)
        if device is None:
            return
        import comfy.model_management as model_management

        loaded_encoders = [
            loaded_model
            for loaded_model in model_management.current_loaded_models
            if _is_loaded_text_encoder(loaded_model, device)
        ]
        if not loaded_encoders:
            return
        if _text_encoder_still_needed():
            logger.debug("Text encoder kept on the device: a node of this prompt still to run takes it")
            return
        encoder_bytes = {id(loaded_model): loaded_model.model_loaded_memory() for loaded_model in loaded_encoders}
        encoder_names = [type(loaded_model.model.model).__name__ for loaded_model in loaded_encoders]
        kept_models = [
            loaded_model
            for loaded_model in model_management.current_loaded_models
            if id(loaded_model) not in encoder_bytes
        ]
        unloaded_models = model_management.free_memory(_FULL_UNLOAD_BYTES, device, keep_loaded=kept_models)
        released_bytes = sum(encoder_bytes.get(id(loaded_model), 0) for loaded_model in unloaded_models)
        logger.debug(
            "Released text encoder(s) {} ({} MB) to their CPU origins before the sampling lease wait",
            ", ".join(encoder_names),
            released_bytes // 2**20,
        )
    except Exception as e:
        logger.warning(f"Text encoder release skipped ({type(e).__name__}: {e}); the diffusion load frees its room")


def _rebase_clamp_on_grant(lease: SamplingLease) -> None:
    """Rebase the run's free-VRAM clamp on the figure an acquired grant carries; never raises."""
    if not isinstance(lease, ClearanceTruthLease):
        return
    try:
        device_free_mb = lease.take_cleared_device_free_mb()
        if device_free_mb is None:
            return
        from hordelib.execution.comfy_patches import rebase_free_memory_view_clamp

        rebase_free_memory_view_clamp(device_free_mb)
    except Exception as exc:
        logger.warning(f"Could not rebase the free-VRAM clamp on the lease grant ({exc}); keeping the run's figure")


def _take_sampling_reserve_on_grant(lease: SamplingLease) -> None:
    """Hold the activation reserve an acquired grant carries for the rest of the hooked sample; never raises."""
    if not isinstance(lease, ClearanceReserveLease):
        return
    try:
        sampling_reserve_mb = lease.take_cleared_sampling_reserve_mb()
    except Exception as exc:
        logger.warning(
            f"Could not read the activation reserve from the lease grant ({exc}); keeping ComfyUI's estimate"
        )
        return
    if sampling_reserve_mb is None:
        return
    _lease_scope.sampling_reserve_mb = sampling_reserve_mb
    _lease_scope.sampling_reserve_logged = False


def held_sampling_reserve_mb() -> float | None:
    """Return the activation reserve (MB) a grant of this thread's hooked sample carried, or None."""
    sampling_reserve_mb: float | None = getattr(_lease_scope, "sampling_reserve_mb", None)
    return sampling_reserve_mb


def _bound_by_sampling_reserve(memory_required: float, minimum_memory_required: float) -> tuple[float, float]:
    """Return ComfyUI's two memory estimates (bytes), each raised to the held activation reserve when one is held."""
    sampling_reserve_mb = held_sampling_reserve_mb()
    if sampling_reserve_mb is None:
        return memory_required, minimum_memory_required
    sampling_reserve_bytes = int(sampling_reserve_mb * 1024 * 1024)
    if not getattr(_lease_scope, "sampling_reserve_logged", False):
        _lease_scope.sampling_reserve_logged = True
        logger.debug(
            "Sampling model load bounded by the grant's activation reserve: comfy_estimate_mb={:.0f}"
            " comfy_minimum_mb={:.0f} reserve_mb={:.0f}",
            memory_required / 2**20,
            minimum_memory_required / 2**20,
            sampling_reserve_mb,
        )
    return max(memory_required, sampling_reserve_bytes), max(minimum_memory_required, sampling_reserve_bytes)


def _bounded_estimate(
    original_estimate_memory: Callable[[object, object, object], tuple[float, float]],
) -> Callable[[object, object, object], tuple[float, float]]:
    """Return ComfyUI's ``estimate_memory`` wrapped so ``prepare_sampling`` sees the held reserve's bound."""

    def _bounded_estimate_memory(model: object, noise_shape: object, conds: object) -> tuple[float, float]:
        memory_required, minimum_memory_required = original_estimate_memory(model, noise_shape, conds)
        return _bound_by_sampling_reserve(memory_required, minimum_memory_required)

    return _bounded_estimate_memory


def install_sampling_lease_hook() -> bool:
    """Monkey-patch ``comfy.samplers.CFGGuider.sample`` to hold the lease around the model load and denoise loop.

    The patch replaces the base class attribute, so every guider subclass that inherits ``sample`` (or reaches it
    through ``super().sample``) is covered. ``comfy.sampler_helpers.estimate_memory`` is wrapped too, so a grant's
    activation reserve bounds the estimate ``prepare_sampling`` forms. Idempotent and independent of logfire
    instrumentation.

    Returns:
        True if the hook is in place (now or already), False if ComfyUI's samplers module could not be patched.
    """
    global _installed
    if _installed:
        return True

    try:
        from comfy import sampler_helpers as sampler_helpers_module
        from comfy import samplers as samplers_module
    except Exception as e:
        logger.warning(f"Could not install GPU sampling lease hook: {e}")
        return False

    guider_class = samplers_module.CFGGuider
    original_guider_sample = guider_class.sample

    def _leased_guider_sample(self: object, *args: object, **kwargs: object) -> object:
        # A guider whose sampling nests another hooked call already holds the lease (or already gave up waiting
        # for it); a second acquire would wait on its own grant.
        if getattr(_lease_scope, "active", False):
            return original_guider_sample(self, *args, **kwargs)
        _lease_scope.active = True
        try:
            return _sample_under_lease(self, original_guider_sample, args, kwargs)
        finally:
            _lease_scope.active = False
            _lease_scope.sampling_reserve_mb = None

    guider_class.sample = _leased_guider_sample
    sampler_helpers_module.estimate_memory = _bounded_estimate(sampler_helpers_module.estimate_memory)
    _installed = True
    logger.debug("Installed GPU sampling lease hook on comfy.samplers.CFGGuider.sample")
    return True


def _sample_under_lease(
    guider: object,
    original_guider_sample: Callable[..., object],
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> object:
    """Run the original guider sample inside the lease, if one is set; never blocks past the acquire timeout."""
    # The model's RAM->VRAM move runs inside the lease and pages its weights in from the checkpoint mapping on
    # first touch; asking the OS to prefetch them now lets that disk read overlap the wait for clearance instead
    # of sitting in the sampling window.
    _prefetch_sampling_model_weights(guider)
    lease = _sampling_lease
    stage_lease = _stage_lease
    acquired = False
    if lease is not None:
        _release_text_encoders_before_wait(guider)
        try:
            if stage_lease is None:
                acquired = bool(lease.acquire(True, _acquire_timeout_seconds))
            else:
                acquired = _stage_then_acquire_sampling(guider, lease, stage_lease, args, kwargs)
        except Exception as e:
            logger.warning(f"GPU sampling lease acquire failed ({e}); sampling without it")
            acquired = False
        if acquired:
            _rebase_clamp_on_grant(lease)
            _take_sampling_reserve_on_grant(lease)
        else:
            logger.warning("GPU sampling lease acquire timed out; sampling without it")
    try:
        return original_guider_sample(guider, *args, **kwargs)
    finally:
        if acquired and lease is not None:
            try:
                lease.release()
            except Exception as e:
                logger.error(f"Failed to release GPU sampling lease: {e}")


def _stage_then_acquire_sampling(
    guider: object,
    lease: SamplingLease,
    stage_lease: SamplingLease,
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> bool:
    """Upload the guider's model under the stage lease, then acquire the sampling lease within the same deadline.

    Returns:
        Whether the sampling lease was acquired. With nothing left of the deadline the acquire is a non-blocking try.
    """
    deadline = monotonic() + _acquire_timeout_seconds
    _stage_under_lease(guider, stage_lease, deadline, args, kwargs)
    remaining_seconds = deadline - monotonic()
    if remaining_seconds <= 0:
        return bool(lease.acquire(False))
    return bool(lease.acquire(True, remaining_seconds))


def _stage_under_lease(
    guider: object,
    stage_lease: SamplingLease,
    deadline: float,
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> None:
    """Acquire the stage lease up to the deadline and, on grant, upload the guider's model; never raises."""
    try:
        stage_granted = bool(stage_lease.acquire(True, max(0.0, deadline - monotonic())))
    except Exception as e:
        logger.warning(f"GPU stage lease acquire failed ({e}); the model loads at sampling")
        return
    if not stage_granted:
        logger.warning("GPU stage lease acquire timed out; the model loads at sampling")
        return
    try:
        _rebase_clamp_on_grant(stage_lease)
        _take_sampling_reserve_on_grant(stage_lease)
        # ``CFGGuider.sample(noise, latent_image, ...)`` keeps its conditioning in ``original_conds``; ``conds`` is a
        # per-call copy that exists only while ``sample`` runs.
        noise = args[0] if args else kwargs.get("noise")
        _stage_sampling_model(
            guider,
            noise,
            getattr(guider, "original_conds", None),
            getattr(guider, "model_options", None),
        )
    finally:
        try:
            stage_lease.release()
        except Exception as e:
            logger.error(f"Failed to release GPU stage lease: {e}")


def _stage_sampling_model(guider: object, noise: object, conds: object, model_options: object) -> None:
    """Upload the guider's model to the device with the call ``prepare_sampling`` makes, ahead of it; never raises.

    The model list, memory estimate and keyword arguments match ``comfy.sampler_helpers._prepare_sampling``, so the
    sampling-time ``prepare_sampling`` finds the same models loaded and moves nothing, and the ``load_models_gpu``
    hijacks apply as they do there. The estimate is raised to a held activation reserve as the wrapped
    ``estimate_memory`` raises it there. The bound is a maximum, so applying it twice changes nothing. A failure
    leaves the load to sampling. Nested noise is packed by ``sample`` before ``prepare_sampling`` reads its shape, so
    its estimate would differ here and it is not staged.
    """
    try:
        model_patcher = getattr(guider, "model_patcher", None)
        noise_shape = getattr(noise, "shape", None)
        if model_patcher is None or noise_shape is None or not isinstance(conds, dict):
            logger.debug("Early sampling model upload skipped: no model patcher, noise or conditioning")
            return
        if getattr(noise, "is_nested", False):
            logger.debug("Early sampling model upload skipped: nested noise is packed only inside sample")
            return

        import comfy.model_management
        from comfy.sampler_helpers import (
            estimate_memory,
            get_additional_models,
            get_additional_models_from_model_options,
        )

        additional_models, inference_memory = get_additional_models(conds, model_patcher.model_dtype())
        additional_models += get_additional_models_from_model_options(model_options)
        additional_models += model_patcher.get_nested_additional_models()
        memory_required, minimum_memory_required = _bound_by_sampling_reserve(
            *estimate_memory(model_patcher, noise_shape, conds)
        )
        comfy.model_management.load_models_gpu(
            [model_patcher] + additional_models,
            memory_required=memory_required + inference_memory,
            minimum_memory_required=minimum_memory_required + inference_memory,
            force_full_load=False,
        )
    except Exception as e:
        logger.warning(f"Early sampling model upload failed ({type(e).__name__}: {e}); the model loads at sampling")
