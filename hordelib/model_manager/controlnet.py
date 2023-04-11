import os

from loguru import logger

from hordelib.comfy_horde import horde_load_controlnet
from hordelib.consts import MODEL_CATEGORY_NAMES, MODEL_DB_NAMES
from hordelib.model_manager.base import BaseModelManager


class ControlNetModelManager(BaseModelManager):
    def __init__(self, download_reference=False):
        super().__init__(
            models_db_name=MODEL_DB_NAMES[MODEL_CATEGORY_NAMES.controlnet],
            download_reference=download_reference,
        )
        self.control_nets = {}

    def modelToRam(self, model_name: str):
        raise NotImplementedError(
            "Controlnet requires special handling. Use `ControlNetModelManager.merge_controlnet(...)` instead of `load()`.",
        )  # XXX # TODO There might be way to avoid this.

    def merge_controlnet(
        self,
        control_type,
        model,
        model_baseline="stable diffusion 1",
    ):
        controlnet_name = self.get_controlnet_name(control_type, model_baseline)
        if controlnet_name not in self.model_reference:
            logger.error(f"{controlnet_name} not found")
            return False
        if controlnet_name not in self.available_models:
            logger.error(f"{controlnet_name} not available")
            logger.info(
                f"Downloading {controlnet_name}",
                status="Downloading",
            )  # logger.init_ok
            self.download_control_type(control_type, [model_baseline])
            logger.info(
                f"{controlnet_name} downloaded",
                status="Downloading",
            )  # logger.init_ok

        logger.info(f"{control_type}", status="Merging")  # logger.init
        controlnet_path = os.path.join(
            self.modelFolderPath,
            self.get_controlnet_filename(controlnet_name),
        )
        controlnet = horde_load_controlnet(
            controlnet_path=controlnet_path,
            target_model=model,
        )
        return (controlnet,)

    def download_control_type(
        self,
        control_type,
        sd_baselines=["stable diffusion 1", "stable diffusion 2"],
    ):
        # We need to do a rename, as they're named differently in the model reference
        for bl in sd_baselines:
            controlnet_name = self.get_controlnet_name(control_type, bl)
            if controlnet_name not in self.model_reference:
                logger.warning(
                    f"Could not find {controlnet_name} reference to download",
                )
                continue
            self.download_model(controlnet_name)

    def get_controlnet_name(self, control_type, sd_baseline):
        """We have control nets for both SD and SD2
        So to know which version we need, se use this method to map general control_type (e.g. 'canny')
        to the version stored in our reference based on the SD baseline we need (e.g. control_canny_sd2)
        """
        baseline_appends = {
            "stable diffusion 1": "",
            "stable diffusion 2": "_sd2",
        }
        return f"control_{control_type}{baseline_appends[sd_baseline]}"

    def check_control_type_available(
        self,
        control_type,
        sd_baseline="stable diffusion 1",
    ):
        # We need to do a rename, as they're named differently in the model reference
        controlnet_name = self.get_controlnet_name(control_type, sd_baseline)
        return self.check_model_available(controlnet_name)

    def get_controlnet_filename(self, controlnet_name) -> str | None:
        """Gets the `.safetensors` filename for the model
        so that it can be located on disk
        """
        for f in self.get_model_files(controlnet_name):
            if f["path"].endswith("safetensors"):
                return f["path"]
        return None
