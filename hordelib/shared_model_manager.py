# shared_model_manager.py
import builtins
import glob
from pathlib import Path
from typing import Iterable

from horde_model_reference.legacy.download_live_legacy_dbs import download_all_models as download_live_legacy_dbs
from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORIES
from horde_model_reference.path_consts import get_model_reference_filename
from loguru import logger
from typing_extensions import Self

from hordelib.config_path import get_hordelib_path
from hordelib.consts import MODEL_CATEGORY_NAMES, REMOTE_PROXY
from hordelib.model_manager.hyper import ALL_MODEL_MANAGER_TYPES, BaseModelManager, ModelManager
from hordelib.preload import (
    ANNOTATOR_MODEL_SHA_LOOKUP,
    download_all_controlnet_annotators,
    validate_all_controlnet_annotators,
)
from hordelib.settings import UserSettings


def do_migrations():
    """This function should handle any moving of folders or other restructuring from previous versions."""

    diffusers_dir = Path(UserSettings.get_model_directory()).joinpath("diffusers")
    sd_inpainting_v1_5_ckpt = diffusers_dir.joinpath("sd-v1-5-inpainting.ckpt").resolve()
    sd_inpainting_v1_5_sha256 = diffusers_dir.joinpath("sd-v1-5-inpainting.sha256").resolve()
    if diffusers_dir.exists() and sd_inpainting_v1_5_ckpt.exists():
        logger.warning(
            "stable_diffusion_inpainting found in diffusers folder and is being moved to compvis",
        )

        target_ckpt_path = (
            Path(UserSettings.get_model_directory()).joinpath("compvis").joinpath("sd-v1-5-inpainting.ckpt")
        )
        target_sha_path = (
            Path(UserSettings.get_model_directory()).joinpath("compvis").joinpath("sd-v1-5-inpainting.sha256")
        )

        try:
            sd_inpainting_v1_5_ckpt.rename(target_ckpt_path)
            if sd_inpainting_v1_5_sha256.exists():
                sd_inpainting_v1_5_sha256.rename(target_sha_path)
        except OSError as e:
            logger.error(
                f"Failed to move {sd_inpainting_v1_5_ckpt} to {target_ckpt_path}. {e}",
            )
            logger.error("Please move this file manually and try again.")
            return

        logger.warning(
            "stable_diffusion_inpainting successfully moved to compvis. The diffusers directory can now be deleted.",
        )


class SharedModelManager:
    _instance: Self = None  # type: ignore
    manager: ModelManager

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls.manager = ModelManager()
        return cls._instance

    @classmethod
    def loadModelManagers(
        cls,
        codeformer: bool = False,
        compvis: bool = False,
        controlnet: bool = False,
        # diffusers: bool = False,
        esrgan: bool = False,
        gfpgan: bool = False,
        safety_checker: bool = False,
        lora: bool = False,
        blip: bool = False,
        clip: bool = False,
    ):
        logger.error("This function is deprecated. Please use load_model_managers instead.")
        managers_to_load: list[str] = []
        passed_args = locals().copy()
        passed_args.pop("cls")
        for passed_arg, value in passed_args.items():
            if value and passed_arg in MODEL_CATEGORY_NAMES.__members__.values():
                managers_to_load.append(passed_arg)

        logger.debug(f"Redownloading all model databases to {get_hordelib_path()}.")
        download_live_legacy_dbs(override_existing=True, proxy_url=REMOTE_PROXY)
        do_migrations()
        cls.manager.init_model_managers(managers_to_load)

    @classmethod
    def load_model_managers(
        cls,
        managers_to_load: Iterable[str | MODEL_CATEGORY_NAMES | type[BaseModelManager]],
    ):
        if cls.manager is None:
            cls.manager = ModelManager()

        args_passed = locals().copy()  # XXX This is temporary
        args_passed.pop("cls")  # XXX This is temporary

        logger.debug(f"Redownloading all model databases to {get_hordelib_path()}.")
        download_live_legacy_dbs(override_existing=True, proxy_url=REMOTE_PROXY)
        do_migrations()
        cls.manager.init_model_managers(managers_to_load)

    @classmethod
    def unload_model_managers(
        cls,
        managers_to_unload: Iterable[str | MODEL_CATEGORY_NAMES | type[BaseModelManager]],
    ):
        cls.manager.unload_model_managers(managers_to_unload)

    @staticmethod
    def preloadAnnotators() -> bool:
        """Preload all annotators. If they are already downloaded, this will only ensure the SHA256 integrity.

        Returns:
            bool: If the annotators are downloaded and the integrity is OK, this will return True. Otherwise, false.
        """
        annotators_in_legacy_directory = Path(builtins.annotator_ckpts_path).glob("*.pt*")  # type: ignore

        for legacy_annotator in annotators_in_legacy_directory:
            logger.warning("Annotator found in legacy directory. This file can be safely deleted:")
            logger.warning(f"{legacy_annotator}")

        builtins.annotator_ckpts_path = (  # type: ignore
            Path(UserSettings.get_model_directory()).joinpath("controlnet").joinpath("annotator").joinpath("ckpts")
        )
        # XXX # FIXME _PLEASE_
        # XXX The hope here is that this hack using a shared package (builtins) will be temporary
        # XXX until comfy officially releases support for controlnet, and the wrangling of the downloads
        # XXX in this way will be a thing of the past.

        logger.debug(
            (
                "WORKAROUND: Setting `builtins.annotator_ckpts_path` to: "
                f"{builtins.annotator_ckpts_path}"  # type: ignore
            ),
        )

        # If any annotators are downloaded, check those for SHA integrity first.
        # if any get purged (incomplete download), then we will be able to recover
        # by downloading them below.
        validate_all_controlnet_annotators(builtins.annotator_ckpts_path)  # type: ignore

        logger.info(
            "Attempting to preload all controlnet annotators.",
        )
        logger.info(
            "This may take several minutes...",
        )

        annotators_downloaded_successfully = download_all_controlnet_annotators()
        if not annotators_downloaded_successfully:
            logger.error("Failed to download one or more annotators.")
            return False

        annotators_validated = validate_all_controlnet_annotators(builtins.annotator_ckpts_path)  # type: ignore
        if not annotators_validated:
            logger.error("Failed to validate one or more annotators.")
            return False

        return True
