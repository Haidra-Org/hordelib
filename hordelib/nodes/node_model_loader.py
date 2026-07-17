# node_model_loader.py
# Simple proof of concept custom node to load models.

import os
import time
from pathlib import Path
from typing import Any

import comfy.model_management
import comfy.sd
import comfy.utils
import folder_paths  # type: ignore
import logfire
import torch
from horde_model_reference.component_hash import ComponentKind
from horde_model_reference.component_identity import read_sidecar
from loguru import logger

from hordelib.comfy_horde import log_free_ram
from hordelib.execution.component_cache import (
    ComponentCache,
    ComponentCacheEntry,
    ComponentCacheKey,
    ComponentSlotKind,
    approx_ram_mb_from_bytes,
    pristine_lora_serving_enabled,
)
from hordelib.execution.standalone_vae import (
    plan_standalone_vae_load,
    standalone_vae_path_disabled,
)
from hordelib.metrics import ModelLoadEvent, get_metrics_collector
from hordelib.shared_model_manager import SharedModelManager

_COMPONENT_FILE_TYPE_KINDS: dict[str, ComponentSlotKind] = {
    "unet": ComponentSlotKind.UNET,
    "text_encoder": ComponentSlotKind.CLIP,
    "vae": ComponentSlotKind.VAE,
}
"""Maps a bare-component ``file_type`` to the cache slot kind it occupies. Kept aligned with
``COMPONENT_FILE_TYPES``; anything absent defaults to the whole-checkpoint kind."""

_sidecar_estimate_warned = False
"""Set once the first sidecar read for RAM estimation fails, so the warning is logged only once per process."""

# Module-level metrics for model loading performance
cache_hits_counter = logfire.metric_counter(
    "model.cache.hits",
    unit="1",
    description="Number of model cache hits",
)

cache_misses_counter = logfire.metric_counter(
    "model.cache.misses",
    unit="1",
    description="Number of model cache misses",
)

disk_load_histogram = logfire.metric_histogram(
    "model.disk_load.duration_ms",
    unit="ms",
    description="Duration to load model from disk",
)


# Don't let the name fool you, this class is trying to load all the files that will be necessary
# for a given comfyUI workflow. That includes loras, etc.
# TODO: Rename to HordeWorkflowModelsLoader ;)
COMPONENT_FILE_TYPES = {"unet", "vae", "text_encoder"}
"""file_type values loaded as bare components via ``comfy.sd.load_diffusion_model``.

Any other file_type (e.g. the stable cascade stages) only selects which file of a
multi-file model entry to load — the file itself is still a full checkpoint and goes
through ``load_checkpoint_guess_config``.
"""


class HordeCheckpointLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "will_load_loras": ("<bool>",),
                "seamless_tiling_enabled": ("<bool>",),
                "horde_model_name": ("<horde model name>",),
                "ckpt_name": ("<ckpt name>",),
                "file_type": ("<file type>",),  # TODO: Make this optional
            },
            "optional": {
                "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"],),  # Unet model type
                # Disaggregated stages load only the component they run: a sample stage loads the UNet
                # (output_clip/output_vae False), a text-encode stage the CLIP (output_model/output_vae
                # False), an image lane the VAE. Absent flags default to a full checkpoint load.
                "output_model": ("<bool>",),
                "output_vae": ("<bool>",),
                "output_clip": ("<bool>",),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load_checkpoint"

    CATEGORY = "loaders"

    @logfire.instrument("model.load_checkpoint", extract_args=False)
    def load_checkpoint(
        self,
        will_load_loras: bool,
        seamless_tiling_enabled: bool,
        horde_model_name: str,
        ckpt_name: str | None = None,
        file_type: str | None = None,
        weight_dtype: str | None = None,
        output_model=True,  # False for a text-encode/decode stage that skips the UNet
        output_vae=True,  # this arg is required by comfyui internals
        output_clip=True,  # this arg is required by comfyui internals
        preloading=False,
    ):
        logger.info(
            "Loading model checkpoint: model={}, file_type={}, will_load_loras={}, seamless={}",
            horde_model_name,
            file_type,
            will_load_loras,
            seamless_tiling_enabled,
        )

        log_free_ram()
        if file_type is not None:
            logger.debug("Loading model: name={}, file_type={}", horde_model_name, file_type)
        else:
            logger.debug("Loading model: name={}", horde_model_name)
        logger.debug("Model options: will_load_loras={}, seamless_tiling={}", will_load_loras, seamless_tiling_enabled)
        if ckpt_name:
            logger.debug("Checkpoint name: name={}", ckpt_name)
            # Check if the checkpoint name is a path
            if Path(ckpt_name).is_absolute():
                logger.debug("Checkpoint name is an absolute path.")

        if preloading:
            logger.debug("Preloading model.")

        if SharedModelManager.manager.compvis is None:
            raise ValueError("CompVisModelManager is not initialised.")

        cache = SharedModelManager.manager._models_in_ram

        # A VAE-only decode of a monolithic checkpoint (file_type is None) can be served from a small
        # pre-extracted standalone VAE and cached by the VAE's content identity, so models that share a
        # byte-identical VAE hit one cache entry instead of each subset-loading their own multi-gigabyte
        # checkpoint. Any absence (no fresh sidecar, no extracted file, kill-switch set) returns None and
        # falls through to the unchanged subset-load path below.
        vae_only_request = bool(output_vae) and not output_model and not output_clip
        standalone_eligible = vae_only_request and file_type is None and not standalone_vae_path_disabled()
        if standalone_eligible:
            standalone_result = self._load_standalone_vae(
                cache,
                horde_model_name,
                ckpt_name,
                seamless_tiling_enabled,
            )
            if standalone_result is not None:
                return standalone_result

        # An entry is stored reusable so a later job can share this pristine base, including a LoRA-bearing
        # one: the graph's LoRA loader clones the base ModelPatcher/CLIP before patching, so the cached
        # weights are never mutated. The rollback knob restores the historical "never share a LoRA job's
        # base" behaviour by marking such entries non-reusable.
        entry_reusable = pristine_lora_serving_enabled() or not will_load_loras

        if file_type in COMPONENT_FILE_TYPES:
            return self._load_bare_component(
                cache,
                horde_model_name,
                ckpt_name,
                file_type,
                weight_dtype,
                seamless_tiling_enabled,
                entry_reusable,
            )

        return self._load_monolithic_checkpoint(
            cache,
            horde_model_name,
            ckpt_name,
            output_model,
            output_vae,
            output_clip,
            seamless_tiling_enabled,
            entry_reusable,
        )

    def _load_monolithic_checkpoint(
        self,
        cache: ComponentCache,
        horde_model_name: str,
        ckpt_name: str | None,
        output_model: bool,
        output_vae: bool,
        output_clip: bool,
        seamless_tiling_enabled: bool,
        entry_reusable: bool,
    ):
        """Serve a full or subset checkpoint through the component cache, cold-loading on a miss.

        A cached tuple satisfies the request only when it carries every component the request asks for; a
        narrower cached tuple (a prior subset load left None in the omitted slots) is a miss that reloads and
        replaces the entry with the broader tuple. Seamless tiling is re-applied on every hit.

        The cache identity is the bare model name: it is stable and free to derive, so a warm hit resolves
        no record or disk path, and every full-or-subset load of a model shares one entry (the property the
        subset-satisfaction check relies on). Checkpoint file resolution happens only on a miss.
        """
        cache_key = ComponentCacheKey(ComponentSlotKind.CHECKPOINT, horde_model_name)
        collector = get_metrics_collector()

        entry = cache.get(cache_key)
        if entry is not None:
            # Indexed access, not unpacking: the cached tuple mirrors load_checkpoint_guess_config's return,
            # which carries trailing elements beyond (model, clip, vae) that this reuse path never touches.
            payload = entry.payload
            cached_model, cached_clip, cached_vae = payload[0], payload[1], payload[2]
            satisfies_request = (
                (not output_model or cached_model is not None)
                and (not output_clip or cached_clip is not None)
                and (not output_vae or cached_vae is not None)
            )
            if satisfies_request:
                cache_hits_counter.add(1)
                collector.record_component_cache_hit()
                collector.record_component_cache_held_mb(cache.held_mb())
                logger.info("Model cache hit: model={}", horde_model_name)
                _apply_model_tiling(cached_model, seamless_tiling_enabled)
                _apply_vae_tiling(cached_vae, seamless_tiling_enabled)
                log_free_ram()
                return payload
            logger.info(
                "Cached load of {} lacks a component this request needs; reloading from disk.",
                horde_model_name,
            )

        cache_misses_counter.add(1)
        collector.record_component_cache_miss()
        logger.info("Model cache miss - loading from disk: model={}", horde_model_name)
        _release_single_slot_before_cold_load(cache)

        resolved_ckpt_name = self._resolve_monolithic_ckpt_name(horde_model_name, ckpt_name)
        ckpt_path = self._resolve_ckpt_path(resolved_ckpt_name)

        from hordelib.execution.zero_copy_load import zero_copy_state_dict_assignment

        with torch.no_grad(), zero_copy_state_dict_assignment():
            load_start_time = time.time()
            with logfire.span("model.load_checkpoint_guess_config"):
                result = comfy.sd.load_checkpoint_guess_config(
                    ckpt_path,
                    output_model=output_model,
                    output_vae=output_vae,
                    output_clip=output_clip,
                    embedding_directory=folder_paths.get_folder_paths("embeddings"),
                )
            load_duration_ms = (time.time() - load_start_time) * 1000
            disk_load_histogram.record(load_duration_ms)
            collector.record_model_load(
                ModelLoadEvent(
                    model_name=horde_model_name,
                    phase="disk_to_ram",
                    duration_seconds=load_duration_ms / 1000,
                    timestamp=time.time(),
                ),
            )
            logger.info(
                "Model loaded from disk: model={}, load_duration_ms={:.2f}",
                horde_model_name,
                load_duration_ms,
            )
            logger.debug(result)

        # A disaggregated stage may have loaded only a subset, so a slot can be None; each helper is a no-op
        # when its component is absent.
        _apply_model_tiling(result[0], seamless_tiling_enabled)
        _apply_vae_tiling(result[2], seamless_tiling_enabled)

        evicted = cache.put(
            ComponentCacheEntry(
                key=cache_key,
                payload=result,
                approx_ram_mb=_estimate_checkpoint_ram_mb(ckpt_path, result),
                reusable=entry_reusable,
                source_ckpt_path=str(ckpt_path),
            ),
        )
        self._record_evictions(evicted, cache, collector)

        log_free_ram()
        logger.debug(result)
        return result

    def _load_bare_component(
        self,
        cache: ComponentCache,
        horde_model_name: str,
        ckpt_name: str | None,
        file_type: str,
        weight_dtype: str | None,
        seamless_tiling_enabled: bool,
        entry_reusable: bool,
    ):
        """Serve a bare single-component load (``comfy.sd.load_diffusion_model``) through the component cache.

        Component file_types (unet, vae, text_encoder) select one file of a multi-file model and load it as a
        single object; the cached tuple is ``(component, None, None)``. A cache hit returns the tuple as-is.
        """
        cache_key = ComponentCacheKey(
            _COMPONENT_FILE_TYPE_KINDS.get(file_type, ComponentSlotKind.CHECKPOINT),
            f"{horde_model_name}:{file_type}",
        )
        collector = get_metrics_collector()

        entry = cache.get(cache_key)
        if entry is not None:
            cache_hits_counter.add(1)
            collector.record_component_cache_hit()
            collector.record_component_cache_held_mb(cache.held_mb())
            logger.info("Model cache hit: model={}, file_type={}", horde_model_name, file_type)
            log_free_ram()
            return entry.payload

        cache_misses_counter.add(1)
        collector.record_component_cache_miss()
        logger.info("Model cache miss - loading from disk: model={}, file_type={}", horde_model_name, file_type)
        _release_single_slot_before_cold_load(cache)

        resolved_ckpt_name = self._resolve_component_file(horde_model_name, ckpt_name, file_type)
        ckpt_path = self._resolve_ckpt_path(resolved_ckpt_name)

        # Keep the checkpoint's mmap-backed tensors as the module weights where byte-identical
        # (dtype-matching) instead of copying them into private memory: sibling processes pinning the
        # same model then share one set of physical pages via the OS page cache. See
        # hordelib.execution.zero_copy_load for the exact adoption rules and fallbacks.
        from hordelib.execution.zero_copy_load import zero_copy_state_dict_assignment

        with torch.no_grad(), zero_copy_state_dict_assignment():
            load_start_time = time.time()
            with logfire.span("model.load_diffusion_model", file_type=file_type):
                model_options: dict[str, Any] = {}
                if weight_dtype == "fp8_e4m3fn":
                    model_options["dtype"] = torch.float8_e4m3fn
                elif weight_dtype == "fp8_e4m3fn_fast":
                    model_options["dtype"] = torch.float8_e4m3fn
                    model_options["fp8_optimizations"] = True
                elif weight_dtype == "fp8_e5m2":
                    model_options["dtype"] = torch.float8_e5m2
                loaded_component = comfy.sd.load_diffusion_model(
                    ckpt_path,
                    model_options=model_options,
                )
                logger.debug(loaded_component)
                # HordeCheckpointLoader declares (MODEL, CLIP, VAE) outputs, but a component
                # load produces a single model object. Pad to the declared arity: ComfyUI calls
                # len() on the returned value to map node outputs (a bare ModelPatcher raises
                # "object of type 'ModelPatcher' has no len()"), and component pipelines (e.g.
                # qwen) wire CLIP/VAE from their own loader nodes, so None here is correct.
                result = (loaded_component, None, None)
            load_duration_ms = (time.time() - load_start_time) * 1000
            disk_load_histogram.record(load_duration_ms)
            collector.record_model_load(
                ModelLoadEvent(
                    model_name=f"{horde_model_name}:{file_type}",
                    phase="disk_to_ram",
                    duration_seconds=load_duration_ms / 1000,
                    timestamp=time.time(),
                ),
            )
            logger.info(
                "Model loaded from disk: model={}, file_type={}, load_duration_ms={:.2f}",
                horde_model_name,
                file_type,
                load_duration_ms,
            )
            logger.debug(result)

        _apply_component_tiling(result[0], file_type, seamless_tiling_enabled)

        evicted = cache.put(
            ComponentCacheEntry(
                key=cache_key,
                payload=result,
                approx_ram_mb=approx_ram_mb_from_bytes(cache_key.kind, _safe_file_size(ckpt_path)),
                reusable=entry_reusable,
                source_ckpt_path=str(ckpt_path),
            ),
        )
        self._record_evictions(evicted, cache, collector)

        log_free_ram()
        logger.debug(result)
        return result

    def _load_standalone_vae(
        self,
        cache: ComponentCache,
        horde_model_name: str,
        ckpt_name: str | None,
        seamless_tiling_enabled: bool,
    ) -> tuple[Any, Any, Any, Any] | None:
        """Serve a VAE-only request from the checkpoint's pre-extracted standalone VAE, or None to fall back.

        Resolves the monolithic checkpoint, consults its component-identity sidecar, and when a fresh sidecar
        records an on-disk extracted VAE, serves that VAE keyed by its content hash so two models embedding
        byte-identical VAE weights share one cached entry. Returns None (fall back to the subset load) whenever
        any of that is unavailable, so the standalone path never alters behaviour it cannot improve.

        The returned tuple mirrors the subset load's ``(model, clip, vae, clipvision)`` shape with only the VAE
        populated, so the cache and downstream node-output mapping are indistinguishable from the subset path.
        """
        ckpt_path = _resolve_monolithic_checkpoint_path(horde_model_name, ckpt_name)
        if ckpt_path is None:
            return None

        plan = plan_standalone_vae_load(ckpt_path, _locate_vae_file)
        if plan is None:
            return None

        cache_key = ComponentCacheKey(ComponentSlotKind.VAE, plan.cache_key)
        collector = get_metrics_collector()

        entry = cache.get(cache_key)
        if entry is not None:
            cache_hits_counter.add(1)
            collector.record_component_cache_hit()
            collector.record_component_cache_held_mb(cache.held_mb())
            logger.info(
                "VAE cache hit by content identity: model={}, key={}",
                horde_model_name,
                plan.cache_key,
            )
            _apply_vae_tiling(entry.payload[2], seamless_tiling_enabled)
            log_free_ram()
            return entry.payload

        cache_misses_counter.add(1)
        collector.record_component_cache_miss()
        logger.info(
            "Loading standalone VAE from disk: model={}, file={}, key={}",
            horde_model_name,
            plan.vae_file_path.name,
            plan.cache_key,
        )
        _release_single_slot_before_cold_load(cache)

        from hordelib.execution.zero_copy_load import zero_copy_state_dict_assignment

        with torch.no_grad(), zero_copy_state_dict_assignment():
            load_start_time = time.time()
            state_dict = comfy.utils.load_torch_file(str(plan.vae_file_path))
            loaded_vae = comfy.sd.VAE(sd=state_dict)
            load_duration_ms = (time.time() - load_start_time) * 1000
            disk_load_histogram.record(load_duration_ms)
            collector.record_model_load(
                ModelLoadEvent(
                    model_name=plan.cache_key,
                    phase="disk_to_ram",
                    duration_seconds=load_duration_ms / 1000,
                    timestamp=time.time(),
                ),
            )

        # Stored always-reusable regardless of the job's LoRA intent: LoRA patches attach to the UNet and
        # text encoders, never the VAE, so a LoRA-bearing job's decode can safely share this entry. Marking
        # it non-reusable would make every LoRA job reload the VAE for no isolation benefit.
        result: tuple[Any, Any, Any, Any] = (None, None, loaded_vae, None)
        evicted = cache.put(
            ComponentCacheEntry(
                key=cache_key,
                payload=result,
                approx_ram_mb=approx_ram_mb_from_bytes(ComponentSlotKind.VAE, plan.vae_tensor_bytes),
                reusable=True,
                source_ckpt_path=str(plan.vae_file_path),
            ),
        )
        self._record_evictions(evicted, cache, collector)
        _apply_vae_tiling(loaded_vae, seamless_tiling_enabled)
        log_free_ram()
        return result

    @staticmethod
    def _record_evictions(
        evicted: list[ComponentCacheEntry],
        cache: ComponentCache,
        collector,
    ) -> None:
        """Log any budget evictions and record their count and the resulting residency into the collector."""
        if evicted:
            logger.debug(
                "Component cache evicted entries to fit budget: identities={}",
                [entry.key.identity for entry in evicted],
            )
            collector.record_component_cache_evictions(len(evicted))
        collector.record_component_cache_held_mb(cache.held_mb())

    def _resolve_monolithic_ckpt_name(
        self,
        horde_model_name: str,
        ckpt_name: str | None,
    ) -> str | None:
        """Resolve the file_type-None checkpoint's on-disk file name (the model's first declared file).

        Returns *ckpt_name* verbatim when given, otherwise the model's first declared file. Raises when the
        model is unavailable. Called only on a cache miss.
        """
        if ckpt_name:
            return ckpt_name

        compvis = SharedModelManager.manager.compvis
        if compvis is None or not compvis.is_model_available(horde_model_name):
            raise ValueError(f"Model {horde_model_name} is not available.")

        file_entries = compvis.get_model_filenames(horde_model_name)
        first_entry = next(iter(file_entries), None)
        if first_entry is None:
            return None
        file_path = first_entry["file_path"]
        return str(file_path) if file_path.is_absolute() else file_path.name

    def _resolve_component_file(
        self,
        horde_model_name: str,
        ckpt_name: str | None,
        file_type: str,
    ) -> str:
        """Resolve the on-disk file name for a bare-component (file_type) load, raising when none matches."""
        if ckpt_name:
            return ckpt_name

        compvis = SharedModelManager.manager.compvis
        if compvis is None or not compvis.is_model_available(horde_model_name):
            raise ValueError(f"Model {horde_model_name} is not available.")

        for file_entry in compvis.get_model_filenames(horde_model_name):
            if file_entry.get("file_type") == file_type:
                return file_entry["file_path"].name
        raise ValueError(f"No {file_type} file for model {horde_model_name}.")

    @staticmethod
    def _resolve_ckpt_path(ckpt_name: str | None) -> str:
        """Turn a resolved checkpoint file name into an absolute on-disk path, raising when it cannot be found."""
        if ckpt_name is None:
            raise ValueError("No model file name provided.")
        if Path(ckpt_name).is_absolute():
            return ckpt_name
        full_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        if full_path is None:
            raise ValueError(f"Model file {ckpt_name} not found.")
        return full_path


def _release_single_slot_before_cold_load(cache: ComponentCache) -> None:
    """Release every resident entry before a cold load when the cache runs in single-slot mode.

    In single-slot mode (budget 0) the resident entry is about to be displaced anyway; releasing it before
    the multi-gigabyte disk read, rather than at insert time, keeps the swap's transient RAM profile at one
    component instead of two, which is the historical single-slot behaviour the zero budget promises. A
    budgeted cache keeps its entries: eviction to fit is the insert's job.
    """
    if cache.budget_mb == 0:
        cache.evict_all()


def _locate_vae_file(file_name: str) -> Path | None:
    """Return the on-disk path of an extracted standalone VAE by name, via ComfyUI's ``vae`` folder search."""
    full_path = folder_paths.get_full_path("vae", file_name)
    return Path(full_path) if full_path else None


def _resolve_monolithic_checkpoint_path(horde_model_name: str, ckpt_name: str | None) -> Path | None:
    """Resolve the monolithic checkpoint path for a VAE-only request, or None when it cannot be found.

    Mirrors the main-flow resolution for the ``file_type is None`` case (the checkpoint is the model's first
    declared file), but returns None on any miss rather than raising, so the standalone path always falls
    back cleanly to the subset load.
    """
    name = ckpt_name
    if not name:
        compvis = SharedModelManager.manager.compvis
        if compvis is None or not compvis.is_model_available(horde_model_name):
            return None
        try:
            file_entries = compvis.get_model_filenames(horde_model_name)
        except ValueError:
            return None
        first_entry = next(iter(file_entries), None)
        if first_entry is None:
            return None
        file_path = first_entry["file_path"]
        name = str(file_path) if file_path.is_absolute() else file_path.name

    if Path(name).is_absolute():
        candidate = Path(name)
        return candidate if candidate.exists() else None

    full_path = folder_paths.get_full_path("checkpoints", name)
    return Path(full_path) if full_path else None


def _safe_file_size(path: str) -> int | None:
    """Return the on-disk byte size of *path*, or None when it cannot be stat-ed."""
    try:
        return os.path.getsize(path)
    except OSError:
        return None


def _safe_read_sidecar(ckpt_path: str):
    """Read a checkpoint's component-identity sidecar for RAM estimation, warning once on failure."""
    global _sidecar_estimate_warned
    try:
        return read_sidecar(Path(ckpt_path))
    except Exception:
        if not _sidecar_estimate_warned:
            logger.warning("Component RAM estimation could not read a sidecar; using conservative constants.")
            _sidecar_estimate_warned = True
        return None


def _estimate_checkpoint_ram_mb(ckpt_path: str, payload: tuple) -> float:
    """Estimate the RAM cost of a loaded checkpoint tuple from its sidecar, or a conservative constant.

    Sums only the components actually present in *payload*: the UNet-ish residual for the model slot, the
    text-encoder tensor bytes for the CLIP slot, and the VAE tensor bytes for the VAE slot. Without a usable
    sidecar it falls back to the per-kind constant, never raising.
    """
    sidecar = _safe_read_sidecar(ckpt_path)
    if sidecar is not None:
        total_bytes = 0
        if payload[0] is not None:
            total_bytes += sidecar.residual_tensor_bytes
        if len(payload) > 1 and payload[1] is not None:
            text_encoders = sidecar.embedded.get(ComponentKind.TEXT_ENCODERS.value)
            if text_encoders is not None:
                total_bytes += text_encoders.tensor_bytes
        if len(payload) > 2 and payload[2] is not None:
            vae = sidecar.embedded.get(ComponentKind.VAE.value)
            if vae is not None:
                total_bytes += vae.tensor_bytes
        if total_bytes > 0:
            return approx_ram_mb_from_bytes(ComponentSlotKind.CHECKPOINT, total_bytes)
    return approx_ram_mb_from_bytes(ComponentSlotKind.CHECKPOINT, None)


def _apply_model_tiling(model: Any, seamless_tiling_enabled: bool) -> None:
    """Retune a model's Conv2d padding for seamless tiling (or reset it); a no-op when *model* is None.

    Also a no-op for objects that are not ModelPatcher-shaped (no ``model`` attribute), such as a bare text
    encoder: the retune only applies to Conv2d-bearing diffusion modules, and the pre-helper code applied
    nothing at all to bare components.
    """
    if model is None or not hasattr(model, "model"):
        return
    model.model.apply(make_circular if seamless_tiling_enabled else make_regular)


def _apply_vae_tiling(vae: Any, seamless_tiling_enabled: bool) -> None:
    """Retune a VAE's Conv2d padding for seamless tiling (or reset it); a no-op when *vae* is None."""
    if vae is None:
        return
    if seamless_tiling_enabled:
        make_circular_vae(vae)
    else:
        make_regular_vae(vae)


def _apply_component_tiling(component: Any, file_type: str | None, seamless_tiling_enabled: bool) -> None:
    """Retune a bare-component load's tiling: the VAE via the VAE helpers, any other component as a model.

    A ``vae`` component is normalized through the VAE helpers; every other component (unet, text encoder) is
    a diffusion ModelPatcher whose Conv2d padding is normalized via the model helpers. Both are no-ops for
    architectures the retune does not apply to.
    """
    if file_type == "vae":
        _apply_vae_tiling(component, seamless_tiling_enabled)
    else:
        _apply_model_tiling(component, seamless_tiling_enabled)


def make_circular(m):
    if isinstance(m, torch.nn.Conv2d):
        m.padding_mode = "circular"


def make_circular_vae(m):
    # Not every VAE exposes a `first_stage_model`: newer architectures (e.g. Z-Image/Lumina2) return a
    # VAE whose `first_stage_model` is None, so reaching into it to retune Conv2d padding would raise and
    # take down the whole load. Seamless tiling simply doesn't apply to those, so skip it rather than fault.
    first_stage_model = getattr(m, "first_stage_model", None)
    if first_stage_model is None:
        logger.debug("VAE has no first_stage_model; skipping circular tiling for this architecture")
        return
    first_stage_model.apply(make_circular)


def make_regular(m):
    if isinstance(m, torch.nn.Conv2d):
        m.padding_mode = "zeros"


def make_regular_vae(m):
    first_stage_model = getattr(m, "first_stage_model", None)
    if first_stage_model is None:
        logger.debug("VAE has no first_stage_model; skipping regular tiling for this architecture")
        return
    first_stage_model.apply(make_regular)


NODE_CLASS_MAPPINGS = {"HordeCheckpointLoader": HordeCheckpointLoader}
