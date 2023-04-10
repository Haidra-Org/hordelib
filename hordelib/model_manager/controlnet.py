import os
from loguru import logger

# from ldm.util import instantiate_from_config
from hordelib.cache import get_cache_directory
from hordelib.consts import REMOTE_MODEL_DB
from hordelib.model_manager.base import BaseModelManager
from hordelib.comfy_horde import load_controlnet


class ControlNetModelManager(BaseModelManager):
    def __init__(self, download_reference=True, compvis=None):
        super().__init__()
        self.download_reference = download_reference
        self.path = f"{get_cache_directory()}/controlnet"
        self.models_db_name = "controlnet"
        self.models_path = self.pkg / f"{self.models_db_name}.json"
        self.remote_db = f"{REMOTE_MODEL_DB}{self.models_db_name}.json"
        self.control_nets = {}
        self.init()

    def merge_controlnet(
        self,
        control_type,
        model,
        model_baseline="stable diffusion 1",
    ):
        controlnet_name = self.get_controlnet_name(control_type, model_baseline)
        if controlnet_name not in self.models:
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
            self.path, self.get_controlnet_filename(controlnet_name)
        )
        controlnet = load_controlnet(controlnet_path, model)
        return (controlnet,)

    def download_control_type(
        self, control_type, sd_baselines=["stable diffusion 1", "stable diffusion 2"]
    ):
        # We need to do a rename, as they're named differently in the model reference
        for bl in sd_baselines:
            controlnet_name = self.get_controlnet_name(control_type, bl)
            if controlnet_name not in self.models:
                logger.warning(
                    f"Could not find {controlnet_name} reference to download"
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
        self, control_type, sd_baseline="stable diffusion 1"
    ):
        # We need to do a rename, as they're named differently in the model reference
        controlnet_name = self.get_controlnet_name(control_type, sd_baseline)
        return self.check_model_available(controlnet_name)

    def get_controlnet_filename(self, controlnet_name):
        """Gets the `.safetensors` filename for the model
        so that it can be located on disk
        """
        for f in self.get_model_files(controlnet_name):
            if f["path"].endswith("safetensors"):
                return f["path"]
