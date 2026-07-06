# node_model_loader.py
# Simple proof of concept custom node to load models.

import time
from pathlib import Path
from typing import Any

import comfy.model_management
import comfy.sd
import folder_paths  # type: ignore
import logfire
import torch
from loguru import logger

from hordelib.comfy_horde import log_free_ram
from hordelib.metrics import ModelLoadEvent, get_metrics_collector
from hordelib.shared_model_manager import SharedModelManager

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

        horde_in_memory_name = horde_model_name
        if file_type is not None:
            horde_in_memory_name = f"{horde_model_name}:{file_type}"
        same_loaded_model = SharedModelManager.manager._models_in_ram.get(horde_in_memory_name)
        logger.debug([horde_in_memory_name, file_type, same_loaded_model])

        # Check cache hit/miss
        cache_hit = same_loaded_model is not None and not same_loaded_model[1]

        if cache_hit:
            cache_hits_counter.add(1)
            logger.info("Model cache hit: model={}, file_type={}", horde_model_name, file_type)
        else:
            cache_misses_counter.add(1)
            logger.info("Model cache miss - loading from disk: model={}, file_type={}", horde_model_name, file_type)

        # Check if the model was previously loaded and if so, not loaded with Loras
        if same_loaded_model and not same_loaded_model[1]:
            if file_type in COMPONENT_FILE_TYPES:
                logger.debug("Model file was previously loaded, returning it: file_type={}", file_type)
                log_free_ram()
                return same_loaded_model[0]
            if seamless_tiling_enabled:
                same_loaded_model[0][0].model.apply(make_circular)
                make_circular_vae(same_loaded_model[0][2])
            else:
                same_loaded_model[0][0].model.apply(make_regular)
                make_regular_vae(same_loaded_model[0][2])

            logger.debug("Model was previously loaded, returning it.")
            log_free_ram()
            return same_loaded_model[0]

        if not ckpt_name:
            if not SharedModelManager.manager.compvis.is_model_available(horde_model_name):
                raise ValueError(f"Model {horde_model_name} is not available.")

            file_entries = SharedModelManager.manager.compvis.get_model_filenames(horde_model_name)
            for file_entry in file_entries:
                if file_type is not None:
                    # if a file_type has been passed, we look at the available files for this model
                    # To find the appropriate type.
                    if file_entry.get("file_type") == file_type:
                        ckpt_name = file_entry["file_path"].name
                        break
                else:
                    # If there's no file_type passed, we follow the previous approach and pick the first file
                    # (There should be only one)
                    if file_entry["file_path"].is_absolute():
                        ckpt_name = str(file_entry["file_path"])
                    else:
                        ckpt_name = file_entry["file_path"].name
                    break

        # Clear references so comfy can free memory as needed
        SharedModelManager.manager._models_in_ram = {}

        # TODO: Currently we don't preload the layer_diffuse tensors which can potentially be big
        # (3G for SDXL). So they will be loaded during runtime, and their memory usage will be
        # handled by comfy as with any lora.
        # Potential improvement here is to preload these models at this point
        # And then just pass their reference to layered_diffusion.py, but that would require
        # Quite a bit of refactoring.

        if ckpt_name is not None and Path(ckpt_name).is_absolute():
            ckpt_path = ckpt_name
        elif ckpt_name is not None:
            full_path = folder_paths.get_full_path("checkpoints", ckpt_name)

            if full_path is None:
                raise ValueError(f"{file_type} file {ckpt_name} not found.")

            ckpt_path = full_path
        else:
            raise ValueError("No model file name provided.")

        # Keep the checkpoint's mmap-backed tensors as the module weights where byte-identical
        # (dtype-matching) instead of copying them into private memory: sibling processes pinning the
        # same model then share one set of physical pages via the OS page cache. See
        # hordelib.execution.zero_copy_load for the exact adoption rules and fallbacks.
        from hordelib.execution.zero_copy_load import zero_copy_state_dict_assignment

        with torch.no_grad(), zero_copy_state_dict_assignment():
            load_start_time = time.time()

            if file_type in COMPONENT_FILE_TYPES:
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
            else:
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
            get_metrics_collector().record_model_load(
                ModelLoadEvent(
                    model_name=horde_in_memory_name,
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
        SharedModelManager.manager._models_in_ram[horde_in_memory_name] = result, will_load_loras

        # Apply tiling settings - handle both checkpoint format (tuple) and component format (single object)
        if file_type in COMPONENT_FILE_TYPES:
            # For individual components, result is already the model patcher/loader
            # No tiling to apply here as these are handled differently
            pass
        elif seamless_tiling_enabled:
            # For full checkpoints, result is a tuple: (model, clip, vae). A disaggregated stage may
            # have loaded only a subset, so a component can be None; tiling only touches what is present.
            if result[0] is not None:
                result[0].model.apply(make_circular)
            if result[2] is not None:
                make_circular_vae(result[2])
        else:
            # For full checkpoints, apply regular tiling
            if result[0] is not None:
                result[0].model.apply(make_regular)
            if result[2] is not None:
                make_regular_vae(result[2])

        log_free_ram()
        logger.debug(result)
        return result


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
