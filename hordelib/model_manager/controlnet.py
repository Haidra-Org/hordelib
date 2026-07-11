from collections.abc import Iterable

from horde_model_reference import MODEL_REFERENCE_CATEGORY
from horde_model_reference.model_reference_records import GenericModelRecord
from loguru import logger
from strenum import StrEnum

from hordelib.model_manager.base import BaseModelManager


class CONTROLNET_BASELINE_NAMES(StrEnum):  # XXX # TODO Move this to consts.py
    stable_diffusion_1 = "stable diffusion 1"
    stable_diffusion_2 = "stable diffusion 2"


CONTROLNET_BASELINE_APPENDS: dict[CONTROLNET_BASELINE_NAMES | str, str] = {
    CONTROLNET_BASELINE_NAMES.stable_diffusion_1: "",
    CONTROLNET_BASELINE_NAMES.stable_diffusion_2: "_sd2",
}

CONTROLNET_REFERENCE_CONTROL_TYPE_MAP: dict[str, str] = {
    "binary": "scribble",
    "standard_lineart": "lineart",
    "lineart_anime_denoise": "lineart_anime",
    "pidinet": "hed",
    "scribble_xdog": "scribble",
    "scribble_pidinet": "scribble",
    "teed": "hed",
    "pyracanny": "canny",
    "midas_depth": "depth",
    "zoe_depth": "depth",
    "depth_anything": "depth",
    "depth_anything_v2": "depth",
    "oneformer_ade20k": "seg",
    "oneformer_coco": "seg",
    "recolor_luminance": "recolor",
    "recolor_intensity": "recolor",
    "tile_ttplanet_guided": "tile",
    "tile_ttplanet_simple": "tile",
}
"""Map annotator-specific control types to the canonical ControlNet reference they share."""


class ControlNetModelManager(BaseModelManager[GenericModelRecord]):
    def __init__(
        self,
        download_reference=False,
        **kwargs,
    ):
        kwargs.pop("model_category", None)  # consumed by this subclass
        super().__init__(
            model_category=MODEL_REFERENCE_CATEGORY.controlnet,
            download_reference=download_reference,
            **kwargs,
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

        reference_control_type = CONTROLNET_REFERENCE_CONTROL_TYPE_MAP.get(control_type, control_type)
        return f"control_{reference_control_type}{CONTROLNET_BASELINE_APPENDS[sd_baseline]}"
