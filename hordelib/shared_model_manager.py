# shared_model_manager.py
import builtins
from collections.abc import Iterable
from multiprocessing.synchronize import Lock as multiprocessing_lock
from pathlib import Path

import torch
from horde_model_reference import ModelReferenceManager, PrefetchStrategy
from horde_model_reference.model_reference_manager import DeferredPrefetchHandle
from loguru import logger
from typing_extensions import Self

from hordelib.consts import MODEL_CATEGORY_NAMES
from hordelib.model_manager.hyper import (
    ALL_MODEL_MANAGER_TYPES,
    MODEL_MANAGERS_TYPE_LOOKUP,
    BaseModelManager,
    ModelManager,
)
from hordelib.preload import (
    ANNOTATOR_MODEL_SHA_LOOKUP,
    download_all_controlnet_annotators,
    validate_all_controlnet_annotators,
)
from hordelib.settings import UserSettings


class SharedModelManager:
    _instance: Self = None  # type: ignore
    manager: ModelManager
    model_reference_manager: ModelReferenceManager
    cuda_available: bool

    def __new__(cls, do_not_load_model_mangers: bool = True):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls.manager = ModelManager()
            cls.cuda_available = torch.cuda.is_available()
            if not do_not_load_model_mangers:
                cls.load_model_managers()

        return cls._instance

    @classmethod
    def load_model_managers(
        cls,
        managers_to_load: Iterable[str | MODEL_CATEGORY_NAMES | type[BaseModelManager]] = ALL_MODEL_MANAGER_TYPES,
        *,
        multiprocessing_lock: multiprocessing_lock | None = None,
    ):
        """Load the model managers specified.

        Args:
            managers_to_load (Iterable[str  |  MODEL_CATEGORY_NAMES  |  type[BaseModelManager]], optional): \
                The model managers to load. \
                Defaults to ALL_MODEL_MANAGER_TYPES.
            multiprocessing_lock (multiprocessing_lock | None, optional): If you are using multiprocessing, \
                you should pass a lock here. \
                Defaults to None.
        """
        if cls.manager is None:
            cls.manager = ModelManager()

        # ModelReferenceManager is a singleton; subsequent calls return the same instance.
        # The prefetch strategy determines whether reference files are fetched eagerly or lazily.
        try:
            cls.model_reference_manager = ModelReferenceManager(
                prefetch_strategy=PrefetchStrategy.DEFERRED,
            )
            handle = cls.model_reference_manager.deferred_prefetch_handle

            async def download_reference_files(handle: DeferredPrefetchHandle):
                await handle

            import asyncio

            if handle is None:
                raise RuntimeError("ModelReferenceManager's deferred_prefetch_handle is None. This should not happen.")
            asyncio.run(download_reference_files(handle))
        except Exception:
            logger.exception("Failed to initialize model reference manager")
            raise

        cls.manager.init_model_managers(
            managers_to_load,
            multiprocessing_lock=multiprocessing_lock,
        )

    @classmethod
    def unload_model_managers(
        cls,
        managers_to_unload: Iterable[str | MODEL_CATEGORY_NAMES | type[BaseModelManager]],
    ):
        cls.manager.unload_model_managers(managers_to_unload)

    @staticmethod
    def preload_annotators() -> bool:
        """Preload all annotators. If they are already downloaded, this will only ensure the SHA256 integrity.

        Returns:
            bool: If the annotators are downloaded and the integrity is OK, this will return True. Otherwise, false.
        """
        return True  # FIXME
        desired_annotator_path = UserSettings.get_model_directory() / "controlnet" / "annotator" / "ckpts"

        if builtins.annotator_ckpts_path == desired_annotator_path:  # type: ignore
            logger.debug(
                "Controlnet annotators already downloaded and SHA256 integrity validated.",
            )
            return True

        annotators_in_legacy_directory = Path(builtins.annotator_ckpts_path).glob("*.pt*")  # type: ignore

        for legacy_annotator in annotators_in_legacy_directory:
            logger.warning("Annotator found in legacy directory. This file can be safely deleted:")
            logger.warning("Legacy annotator path: path={}", legacy_annotator)

        builtins.annotator_ckpts_path = desired_annotator_path  # type: ignore

        # XXX # FIXME _PLEASE_
        # XXX The hope here is that this hack using a shared package (builtins) will be temporary
        # XXX until comfy officially releases support for controlnet, and the wrangling of the downloads
        # XXX in this way will be a thing of the past.

        logger.debug(
            (
                f"WORKAROUND: Setting `builtins.annotator_ckpts_path` to: {builtins.annotator_ckpts_path}"  # type: ignore
            ),
        )

        # If any annotators are downloaded, check those for SHA integrity first.
        # if any get purged (incomplete download), then we will be able to recover
        # by downloading them below.
        num_validated = validate_all_controlnet_annotators(builtins.annotator_ckpts_path)  # type: ignore
        if num_validated == len(ANNOTATOR_MODEL_SHA_LOOKUP):
            logger.debug(
                "All controlnet annotators SHA256 integrity validated.",
            )
            return True

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
