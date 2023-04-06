# node_model_loader.py
# Simple proof of concept custom node to load models.
# We shall expand this to actually integrate with the horde model manager.

from loguru import logger


class HordeCheckpointLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_manager": ("<model manager instance>",),
                "ckpt_name": ("<checkpoint file>",),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load_checkpoint"

    CATEGORY = "loaders"

    def load_checkpoint(self, model_manager, ckpt_name, output_vae=True, output_clip=True):

        if model_manager.manager.compvis is None:
            logger.error("horde_model_manager.compvis appears to be missing!")
            raise RuntimeError()  # XXX better guarantees need to be made

        if ckpt_name not in model_manager.manager.compvis.loaded_models:
            logger.error(f"Model {ckpt_name} is not loaded")
            raise RuntimeError()  # XXX better guarantees need to be made

        return model_manager.manager.compvis.loaded_models[ckpt_name]


NODE_CLASS_MAPPINGS = {"HordeCheckpointLoader": HordeCheckpointLoader}
