from loguru import logger

from hordelib.horde import SharedModelManager

class HordePPModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model_name": ("<pp file>",),
                             }}
    RETURN_TYPES = ("UPSCALE_MODEL",)
    FUNCTION = "load_model"

    CATEGORY = "loaders"

    def load_model(self, model_name):
        if SharedModelManager.manager is None:  # XXX
            raise RuntimeError()  # XXX

        if SharedModelManager.manager.esrgan is not None and model_name in SharedModelManager.manager.esrgan.models:
            return SharedModelManager.manager.esrgan.loaded_models[model_name]
        elif SharedModelManager.manager.gfpgan is not None and model_name in SharedModelManager.manager.gfpgan.models:
            return SharedModelManager.manager.gfpgan.loaded_models[model_name]
        else:
            logger.error(f"{model_name} not found in any of the PostProcessor Model Managers")
            raise RuntimeError()  # XXX

        return SharedModelManager.manager.compvis.loaded_models[ckpt_name]