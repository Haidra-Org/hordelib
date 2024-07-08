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
        from hordelib.comfy_horde import log_free_ram

        logger.debug(f"Loading controlnet {control_net_name} through our custom node")
        log_free_ram()

        if not model_manager or not model_manager.manager or not model_manager.manager.controlnet:
            logger.error("controlnet model_manager appears to be missing!")
            raise RuntimeError  # XXX better guarantees need to be made

        merge_result = model_manager.manager.controlnet.merge_controlnet(
            control_net_name,
            model,
        )
        log_free_ram()
        return merge_result


NODE_CLASS_MAPPINGS = {"HordeDiffControlNetLoader": HordeDiffControlNetLoader}

logger.debug("Loaded HordeDiffControlNetLoader")
