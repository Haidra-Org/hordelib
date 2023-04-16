# shared_model_manager.py
import builtins
import glob
from pathlib import Path

from loguru import logger

from hordelib.cache import get_cache_directory
from hordelib.model_manager.hyper import ModelManager
from hordelib.preload import annotator_model_integrity, download_all_controlnet_annotators


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
        """Attempt to preload all annotators. If they are downloaded, this will only ensure the SHA256 integrity.

        Raises:
            RuntimeError: Occurs if loadModelManagers() has not yet been called when invoking this method.

        Returns:
            bool: If the annotators are downloaded and the integrity is OK, this will return True. Otherwise, false.
        """
        if cls.manager is None:
            raise RuntimeError("ModelManager not initialized. Please call loadModelManagers() first.")
        if cls.manager.controlnet is None:
            raise RuntimeError(
                (
                    "ControlNet not initialized. "
                    " Please call loadModelManagers() first and specify loading ControlNet."
                ),
            )

        builtins.annotator_ckpts_path = (
            Path(get_cache_directory()).joinpath("controlnet").joinpath("annotator").joinpath("ckpts")
        )
        # XXX # FIXME _PLEASE_
        # XXX The hope here is that this will be temporary, until comfy officially releases support for annotators.

        logger.debug(f"WORKAROUND: Setting `builtins.annotator_ckpts_path` to: {builtins.annotator_ckpts_path}")

        annotators_OK = download_all_controlnet_annotators()
        if not annotators_OK:
            logger.init_err("Failed to download one or more annotators.", status="Error")
            return False

        # We're doing this here because the effort involved in getting these annotators into the model managers
        # is non-trivial, and likely will be made obsolete by comfy implementing this functionality natively.
        annotatorCacheDir = Path(get_cache_directory()).joinpath("controlnet").joinpath("annotator")
        annotators = glob.glob("*.pt*", root_dir=annotatorCacheDir)
        for annotator in annotators:
            annotator_full_path = Path(annotatorCacheDir.joinpath(annotator))
            hash = cls.manager.controlnet.get_file_sha256_hash(annotator_full_path)
            if hash != annotator_model_integrity[annotator]:
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
