import os

import comfy.utils
import folder_paths  # type: ignore
import logfire
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

    @logfire.instrument("lora.load_node")
    def load_lora(self, model, clip, lora_name, strength_model, strength_clip):
        from hordelib.comfy_horde import log_free_ram

        log_free_ram()
        logger.info(
            "lora.load_requested: lora_name={}, strength_model={}, strength_clip={}",
            lora_name,
            strength_model,
            strength_clip,
        )

        _test_exception = os.getenv("FAILURE_TEST", False)
        if _test_exception:
            raise Exception("This tests exceptions being thrown from within the pipeline")

        logger.debug("Loading lora through custom node: lora_name={}", lora_name)

        if strength_model == 0 and strength_clip == 0:
            logger.debug("Strengths are 0, skipping lora loading")
            logger.info("lora.load_skipped: reason=zero_strength")
            return (model, clip)

        if lora_name is None or lora_name == "" or lora_name == "None":
            logger.warning("No lora name provided, skipping lora loading")
            logger.warning("lora.load_skipped: reason=no_name")
            return (model, clip)

        if not os.path.exists(folder_paths.get_full_path("loras", lora_name)):
            logger.warning("Lora file does not exist, skipping: lora_name={}", lora_name)
            logger.warning("lora.load_failed: reason=file_not_found, lora_name={}", lora_name)
            return (model, clip)

        loras_on_disk = folder_paths.get_filename_list("loras")

        if "loras" in folder_paths.filename_list_cache:
            del folder_paths.filename_list_cache["loras"]

        if lora_name not in loras_on_disk:
            logger.warning("Lora file does not exist, skipping: lora_name={}", lora_name)
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

        try:
            with logger.catch(reraise=True):
                if lora is None:
                    with logfire.span("lora.load_from_disk", lora_path=lora_path):
                        lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
                        self.loaded_lora = (lora_path, lora)

                with logfire.span("lora.apply_to_models"):
                    model_lora, clip_lora = comfy.sd.load_lora_for_models(
                        model,
                        clip,
                        lora,
                        strength_model,
                        strength_clip,
                    )
                log_free_ram()
                logger.info("lora.loaded_successfully: lora_name={}", lora_name)
                return (model_lora, clip_lora)
        except Exception as e:
            logger.bind(lora_name=lora_name).exception("Error loading lora")
            logger.error("lora.load_exception: lora_name={}, error={}", lora_name, str(e))
            return (model, clip)


NODE_CLASS_MAPPINGS = {"HordeLoraLoader": HordeLoraLoader}
