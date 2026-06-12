# shared_model_manager.py
import asyncio
from collections.abc import Iterable
from multiprocessing.synchronize import Lock as multiprocessing_lock
from typing import Self

import torch
from horde_model_reference import ModelReferenceManager, PrefetchStrategy
from loguru import logger

from hordelib.consts import MODEL_CATEGORY_NAMES
from hordelib.model_manager.hyper import (
    ALL_MODEL_MANAGER_TYPES,
    BaseModelManager,
    ModelManager,
)
from hordelib.preload import download_all_controlnet_annotators


def _await_prefetch(ref_manager: ModelReferenceManager) -> None:
    """Synchronously wait for the reference manager's deferred prefetch to finish.

    The prefetch downloads/refreshes all model reference files; model managers cannot be
    constructed until it completes.
    """
    handle = ref_manager.deferred_prefetch_handle

    if handle is None:
        raise RuntimeError(
            "ModelReferenceManager's deferred_prefetch_handle is None. This is unexpected with "
            "PrefetchStrategy.DEFERRED; the horde_model_reference API may have changed.",
        )

    async def _wait() -> None:
        await handle

    asyncio.run(_wait())


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
            _await_prefetch(cls.model_reference_manager)
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
        """Preload all controlnet annotators (the comfyui_controlnet_aux detector checkpoints).

        Requires hordelib.initialise() to have completed and a HordeLib/Comfy_Horde instance to
        exist (custom nodes must be loaded).

        Returns:
            bool: True if the annotators downloaded (or were already present) and run correctly.
        """
        return download_all_controlnet_annotators()
