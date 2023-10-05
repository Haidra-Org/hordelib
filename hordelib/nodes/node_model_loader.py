# node_model_loader.py
# Simple proof of concept custom node to load models.

import comfy.model_management
import comfy.sd
import folder_paths  # type: ignore
import torch
from loguru import logger

from hordelib.shared_model_manager import SharedModelManager


class HordeCheckpointLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "will_load_loras": ("<bool>",),
                "seamless_tiling_enabled": ("<bool>",),
                "horde_model_name": ("<horde model name>",),
                "ckpt_name": ("<ckpt name>",),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load_checkpoint"

    CATEGORY = "loaders"

    def load_checkpoint(
        self,
        will_load_loras: bool,
        seamless_tiling_enabled: bool,
        horde_model_name: str,
        ckpt_name: str | None = None,
        output_vae=True,
        output_clip=True,
        preloading=False,
    ):
        logger.debug(f"Loading model {horde_model_name}")
        logger.debug(f"Will load Loras: {will_load_loras}, seamless tiling: {seamless_tiling_enabled}")
        if ckpt_name:
            logger.debug(f"Checkpoint name: {ckpt_name}")

        if preloading:
            logger.debug("Preloading model.")

        if SharedModelManager.manager.compvis is None:
            raise ValueError("CompVisModelManager is not initialised.")

        same_loaded_model = SharedModelManager.manager._models_in_ram.get(horde_model_name)

        # Check if the model was previously loaded and if so, not loaded with Loras
        if same_loaded_model and not same_loaded_model[1]:
            if seamless_tiling_enabled:
                same_loaded_model[0][0].model.apply(make_circular)
                make_circular_vae(same_loaded_model[0][2])
            else:
                same_loaded_model[0][0].model.apply(make_regular)
                make_regular_vae(same_loaded_model[0][2])

            logger.debug("Model was previously loaded, returning it.")

            return same_loaded_model[0]

        if not ckpt_name:
            if not SharedModelManager.manager.compvis.is_model_available(horde_model_name):
                raise ValueError(f"Model {horde_model_name} is not available.")

            ckpt_name = SharedModelManager.manager.compvis.get_model_filename(horde_model_name).name

        # Clear references so comfy can free memory as needed
        SharedModelManager.manager._models_in_ram = {}

        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        result = comfy.sd.load_checkpoint_guess_config(
            ckpt_path,
            output_vae=True,
            output_clip=True,
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
        )

        SharedModelManager.manager._models_in_ram[horde_model_name] = result, will_load_loras

        if seamless_tiling_enabled:
            result[0].model.apply(make_circular)
            make_circular_vae(result[2])
        else:
            result[0].model.apply(make_regular)
            make_regular_vae(result[2])

        return result


def make_circular(m):
    if isinstance(m, torch.nn.Conv2d):
        m.padding_mode = "circular"


def make_circular_vae(m):
    m.first_stage_model.apply(make_circular)


def make_regular(m):
    if isinstance(m, torch.nn.Conv2d):
        m.padding_mode = "zeros"


def make_regular_vae(m):
    m.first_stage_model.apply(make_regular)


NODE_CLASS_MAPPINGS = {"HordeCheckpointLoader": HordeCheckpointLoader}
