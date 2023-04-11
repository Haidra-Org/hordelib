from loguru import logger


class HordeUpscaleModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": ("<model name>",),
                "model_manager": ("<model manager instance>",),
            },
        }

    RETURN_TYPES = ("UPSCALE_MODEL",)
    FUNCTION = "load_model"

    CATEGORY = "loaders"

    def load_model(self, model_name, model_manager):
        logger.debug(f"Loading model {model_name} through our custom node")

        if model_manager.manager is None:
            logger.error("horde_model_manager appears to be missing!")
            raise RuntimeError  # XXX better guarantees need to be made

        if model_name not in model_manager.manager.loaded_models:
            logger.error(f"Model {model_name} is not loaded")
            raise RuntimeError  # XXX better guarantees need to be made
        model = model_manager.manager.loaded_models[model_name]["model"]
        # XXX # TODO I would like to revisit this dict->tuple conversion at some point soon
        return (model,)


NODE_CLASS_MAPPINGS = {"HordeUpscaleModelLoader": HordeUpscaleModelLoader}
