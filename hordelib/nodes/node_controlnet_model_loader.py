import os

import comfy
from loguru import logger


class HordeDiffControlNetLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "control_net_name": ("STRING",),
                "model_manager": ("<model manager instance>",),
            },
        }

    RETURN_TYPES = ("CONTROL_NET",)
    FUNCTION = "load_controlnet"

    CATEGORY = "loaders"

    def load_controlnet(self, model, control_net_name, model_manager):
        logger.debug(f"Loading controlnet {control_net_name} through our custom node")

        if (
            not model_manager
            or not model_manager.manager
            or not model_manager.manager.controlnet
        ):
            logger.error("controlnet model_manager appears to be missing!")
            raise RuntimeError  # XXX better guarantees need to be made

        # FIXME We're not using the model manager here
        # if control_net_name not in model_manager.manager.loaded_models:
        #     logger.error(f"Controlnet {control_net_name} is not loaded")
        #     raise RuntimeError  # XXX better guarantees need to be made

        # XXX This isn't how the model manager is supposed to be used
        controlnet_path = os.path.join(
            model_manager.manager.controlnet.path, control_net_name
        )
        controlnet = comfy.sd.load_controlnet(controlnet_path, model)
        logger.warning(f"{controlnet_path}")
        logger.warning(f"{controlnet}")
        return (controlnet,)


NODE_CLASS_MAPPINGS = {"HordeDiffControlNetLoader": HordeDiffControlNetLoader}

logger.debug("Loaded HordeDiffControlNetLoader")
