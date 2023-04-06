# node_model_loader.py
# Simple proof of concept custom node to load models.
# We shall expand this to actually integrate with the horde model manager.

from loguru import logger

from hordelib.horde import SharedModelManager


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
        if SharedModelManager.manager is None:  # XXX better guarantees need to be made
            raise RuntimeError()  # XXX better guarantees need to be made

        logger.info(SharedModelManager.manager)
        if SharedModelManager.manager.compvis is None:
            logger.error("horde_model_manager.compvis appears to be missing!")
            raise RuntimeError()  # XXX better guarantees need to be made
        logger.info(SharedModelManager.manager.compvis)

        return SharedModelManager.manager.compvis.loaded_models[ckpt_name]


NODE_CLASS_MAPPINGS = {"HordeCheckpointLoader": HordeCheckpointLoader}
