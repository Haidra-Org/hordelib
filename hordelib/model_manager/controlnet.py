import os
import typing
from enum import auto
from typing import Iterable

from loguru import logger
from strenum import StrEnum
from typing_extensions import override

from hordelib.comfy_horde import horde_load_controlnet
from hordelib.consts import MODEL_CATEGORY_NAMES, MODEL_DB_NAMES
from hordelib.model_manager.base import BaseModelManager


class CONTROLNET_BASELINE_NAMES(StrEnum):  # XXX # TODO Move this to consts.py
    stable_diffusion_1 = "stable diffusion 1"
    stable_diffusion_2 = "stable diffusion 2"


CONTROLNET_BASELINE_APPENDS: dict[CONTROLNET_BASELINE_NAMES | str, str] = {
    CONTROLNET_BASELINE_NAMES.stable_diffusion_1: "",
    CONTROLNET_BASELINE_NAMES.stable_diffusion_2: "_sd2",
}


class ControlNetModelManager(BaseModelManager):
    def __init__(self, download_reference=False):
        super().__init__(
            model_category_name=MODEL_CATEGORY_NAMES.controlnet,
            download_reference=download_reference,
        )

    def download_control_type(
        self,
        control_type: str,
        sd_baselines: Iterable[str] | None = None,
    ) -> None:
        if sd_baselines is None:
            sd_baselines = [CONTROLNET_BASELINE_NAMES.stable_diffusion_1, CONTROLNET_BASELINE_NAMES.stable_diffusion_2]
        # We need to do a rename, as they're named differently in the model reference
        for bl in sd_baselines:
            controlnet_name = self.get_controlnet_name(control_type, bl)
            if controlnet_name not in self.model_reference:
                logger.warning(
                    f"Could not find {controlnet_name} reference to download",
                )
                continue
            self.download_model(controlnet_name)

    def get_controlnet_name(self, control_type: str, sd_baseline: str) -> str:
        """We have control nets for both SD and SD2
        So to know which version we need, se use this method to map general control_type (e.g. 'canny')
        to the version stored in our reference based on the SD baseline we need (e.g. control_canny_sd2)
        """

        return f"control_{control_type}{CONTROLNET_BASELINE_APPENDS[sd_baseline]}"

    def get_controlnet_filename(self, controlnet_name: str) -> str | None:
        """Gets the `.safetensors` filename for the model
        so that it can be located on disk
        """
        for f in self.get_model_files(controlnet_name):
            if f["path"].endswith("safetensors"):
                return f["path"]
        logger.error(f"Could not find {controlnet_name}.safetensors on disk.")
        return None
