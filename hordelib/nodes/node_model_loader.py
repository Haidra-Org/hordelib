# node_model_loader.py
# Simple proof of concept custom node to load models.
# We shall expand this to actually integrate with the horde model manager.
import os

import comfy
from hordelib import horde_model_manager
from loguru import logger


class HordeCheckpointLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": ("<checkpoint file>",),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load_checkpoint"

    CATEGORY = "loaders"

    def load_checkpoint(self, ckpt_name, output_vae=True, output_clip=True):
        logger.info(horde_model_manager)
        logger.info(horde_model_manager.compvis)
        return horde_model_manager.compvis.loaded_models[ckpt_name]

NODE_CLASS_MAPPINGS = {"HordeCheckpointLoader": HordeCheckpointLoader}
