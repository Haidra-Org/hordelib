import os

import comfy.utils
import folder_paths
from loguru import logger


class HordeLoraLoader:
    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "lora_name": ("STRING", {"default": ""}),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01}),
                "strength_clip": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "load_lora"

    CATEGORY = "loaders"

    def load_lora(self, model, clip, lora_name, strength_model, strength_clip):
        _test_exception = os.getenv("FAILURE_TEST", False)
        if _test_exception:
            raise Exception("This tests exceptions being thrown from within the pipeline")

        if strength_model == 0 and strength_clip == 0:
            return (model, clip)

        if lora_name is None or lora_name == "" or lora_name == "None":
            logger.warning("No lora name provided, skipping lora loading")
            return (model, clip)

        if not os.path.exists(folder_paths.get_full_path("loras", lora_name)):
            logger.warning(f"Lora file {lora_name} does not exist, skipping lora loading")
            return (model, clip)

        loras_on_disk = folder_paths.get_filename_list("loras")

        if "loras" in folder_paths.filename_list_cache:
            del folder_paths.filename_list_cache["loras"]

        if lora_name not in loras_on_disk:
            logger.warning(f"Lora file {lora_name} does not exist, skipping lora loading")
            return (model, clip)

        lora_path = folder_paths.get_full_path("loras", lora_name)

        lora = None
        if self.loaded_lora is not None:
            if self.loaded_lora[0] == lora_path:
                lora = self.loaded_lora[1]
            else:
                temp = self.loaded_lora
                self.loaded_lora = None
                del temp

        if lora is None:
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
            self.loaded_lora = (lora_path, lora)

        model_lora, clip_lora = comfy.sd.load_lora_for_models(model, clip, lora, strength_model, strength_clip)
        return (model_lora, clip_lora)


NODE_CLASS_MAPPINGS = {"HordeLoraLoader": HordeLoraLoader}
