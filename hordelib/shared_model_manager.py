# shared_model_manager.py
import builtins
import glob
from pathlib import Path

from loguru import logger

from hordelib.cache import get_cache_directory
from hordelib.model_manager.hyper import BaseModelManager, ModelManager
from hordelib.preload import ANNOTATOR_MODEL_SHA_LOOKUP, download_all_controlnet_annotators


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

    @classmethod
    def preloadAnnotators(cls) -> bool:
        """Preload all annotators. If they are already downloaded, this will only ensure the SHA256 integrity.

        Raises:
            RuntimeError: Occurs if loadModelManagers() has not yet been called when invoking this method.

        Returns:
            bool: If the annotators are downloaded and the integrity is OK, this will return True. Otherwise, false.
        """
        annotators_in_legacy_directory = Path(builtins.annotator_ckpts_path).glob("*.pt*")

        for legacy_annotator in annotators_in_legacy_directory:
            logger.warning(f"Annotator found in legacy directory. This file can be safely deleted: {legacy_annotator}")

        builtins.annotator_ckpts_path = (
            Path(get_cache_directory()).joinpath("controlnet").joinpath("annotator").joinpath("ckpts")
        )
        # XXX # FIXME _PLEASE_
        # XXX The hope here is that this hack using a shared package (builtins) will be temporary
        # XXX until comfy officially releases support for controlnet, and the wrangling of the downloads
        # XXX in this way will be a thing of the past.

        logger.debug(f"WORKAROUND: Setting `builtins.annotator_ckpts_path` to: {builtins.annotator_ckpts_path}")

        annotators_downloaded_successfully = download_all_controlnet_annotators()
        if not annotators_downloaded_successfully:
            logger.init_err("Failed to download one or more annotators.", status="Error")
            return False

        annotatorCacheDir = Path(get_cache_directory()).joinpath("controlnet").joinpath("annotator")
        annotators = glob.glob("*.pt*", root_dir=annotatorCacheDir)
        for annotator in annotators:
            annotator_full_path = annotatorCacheDir.joinpath(annotator)
            hash = BaseModelManager.get_file_sha256_hash(annotator_full_path)
            if hash != ANNOTATOR_MODEL_SHA_LOOKUP[annotator]:
                try:
                    annotator_full_path.unlink()
                    logger.init_err(
                        f"Deleted annotator file {annotator} as it was corrupt.",
                        status="Error",
                    )
                except OSError:
                    logger.init_err(
                        f"Annotator file {annotator} is corrupt. Please delete it and try again.",
                        status="Error",
                    )
                    logger.init_err(f"File location: {annotatorCacheDir}", status="Error")
                return False
        return True
