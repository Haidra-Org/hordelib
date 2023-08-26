import os

from comfy.sd import load_lora_for_models
from loguru import logger


class HordeLoraLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "lora_name": ("STRING",),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "strength_clip": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "model_manager": ("MODEL_MANAGER",),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "load_lora"

    CATEGORY = "loaders"

    def load_lora(self, model, clip, lora_name, strength_model, strength_clip, model_manager):
        if model_manager.manager is None:
            logger.error("LoraLoader node was not passed a model manager")
            raise RuntimeError

        lora_path = model_manager.manager.get_model_directory("lora")
        lora_path = os.path.join(lora_path, lora_name)
        # XXX This should call back to the hordelib model manager once it has support
        model_lora, clip_lora = load_lora_for_models(model, clip, lora_path, strength_model, strength_clip)
        return (model_lora, clip_lora)


NODE_CLASS_MAPPINGS = {"HordeLoraLoader": HordeLoraLoader}
