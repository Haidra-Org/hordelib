# shared_model_manager.py
import builtins
import glob
from pathlib import Path

from loguru import logger

from hordelib.cache import get_cache_directory
from hordelib.model_manager.hyper import BaseModelManager, ModelManager
from hordelib.preload import (
    ANNOTATOR_MODEL_SHA_LOOKUP,
    download_all_controlnet_annotators,
    validate_all_controlnet_annotators,
)


class SharedModelManager:
    _instance = None
    manager: ModelManager | None = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def loadModelManagers(
        cls,
        # aitemplate: bool = False,
        blip: bool = False,
        clip: bool = False,
        codeformer: bool = False,
        compvis: bool = False,
        controlnet: bool = False,
        diffusers: bool = False,
        esrgan: bool = False,
        gfpgan: bool = False,
        safety_checker: bool = False,
    ):
        if cls.manager is None:
            cls.manager = ModelManager()

        args_passed = locals().copy()  # XXX This is temporary
        args_passed.pop("cls")  # XXX This is temporary

        cls.manager.init_model_managers(**args_passed)

    @staticmethod
    def preloadAnnotators() -> bool:
        """Preload all annotators. If they are already downloaded, this will only ensure the SHA256 integrity.

        Returns:
            bool: If the annotators are downloaded and the integrity is OK, this will return True. Otherwise, false.
        """
        annotators_in_legacy_directory = Path(builtins.annotator_ckpts_path).glob("*.pt*")

        for legacy_annotator in annotators_in_legacy_directory:
            logger.init_warn("Annotator found in legacy directory. This file can be safely deleted:", status="Warning")
            logger.init_warn(f"{legacy_annotator}", status="Warning")
            logger.init_warn("", status="Warning")

        builtins.annotator_ckpts_path = (
            Path(get_cache_directory()).joinpath("controlnet").joinpath("annotator").joinpath("ckpts")
        )
        # XXX # FIXME _PLEASE_
        # XXX The hope here is that this hack using a shared package (builtins) will be temporary
        # XXX until comfy officially releases support for controlnet, and the wrangling of the downloads
        # XXX in this way will be a thing of the past.

        logger.debug(f"WORKAROUND: Setting `builtins.annotator_ckpts_path` to: {builtins.annotator_ckpts_path}")

        # If any annotators are downloaded, check those for SHA integrity first.
        # if any get purged (incomplete download), then we will be able to recover
        # by downloading them below.
        validate_all_controlnet_annotators(builtins.annotator_ckpts_path)

        logger.init("Attempting to preload all controlnet annotators.", status="Loading")
        logger.init("This may take several minutes...", status="Loading")

        annotators_downloaded_successfully = download_all_controlnet_annotators()
        if not annotators_downloaded_successfully:
            logger.init_err("Failed to download one or more annotators.", status="Error")
            return False

        annotators_all_validated_successfully = validate_all_controlnet_annotators(builtins.annotator_ckpts_path)
        if not annotators_all_validated_successfully:
            logger.init_err("Failed to validate one or more annotators.", status="Error")
            return False

        return True
