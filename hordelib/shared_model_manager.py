# shared_model_manager.py
import asyncio
from collections.abc import Iterable
from multiprocessing.synchronize import Lock as multiprocessing_lock
from typing import Self

import torch
from horde_model_reference import MODEL_REFERENCE_CATEGORY, ModelReferenceManager, PrefetchStrategy
from loguru import logger

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
        managers_to_load: Iterable[str | MODEL_REFERENCE_CATEGORY | type[BaseModelManager]] = ALL_MODEL_MANAGER_TYPES,
        *,
        multiprocessing_lock: multiprocessing_lock | None = None,
        lora_reference_backups: bool | None = None,
    ):
        """Load the model managers specified.

        Args:
            managers_to_load (Iterable[str  |  MODEL_REFERENCE_CATEGORY  |  type[BaseModelManager]], optional): \
                The model managers to load. \
                Defaults to ALL_MODEL_MANAGER_TYPES.
            multiprocessing_lock (multiprocessing_lock | None, optional): If you are using multiprocessing, \
                you should pass a lock here. \
                Defaults to None.
            lora_reference_backups (bool | None, optional): Whether the LoRA manager writes backup \
                copies when saving its reference to disk. Defaults to None (backups on when a \
                multiprocessing lock is in use).
        """
        if cls.manager is None:
            cls.manager = ModelManager()

        # ModelReferenceManager is a singleton; subsequent calls return the same instance.
        # The prefetch strategy determines whether reference files are fetched eagerly or lazily.
        try:
            if not ModelReferenceManager.has_instance():
                cls.model_reference_manager = ModelReferenceManager(
                    prefetch_strategy=PrefetchStrategy.DEFERRED,
                )
                _await_prefetch(cls.model_reference_manager)
        except Exception as e:
            logger.exception("Failed to initialize model reference manager")
            raise RuntimeError("Failed to initialize model reference manager") from e

        # Register the pending/beta provider before constructing managers: each manager
        # loads its database in __init__, and beta_source_for only selects the provider
        # when it is already registered.
        cls._register_pending_provider()

        cls.manager.init_model_managers(
            managers_to_load,
            multiprocessing_lock=multiprocessing_lock,
            lora_reference_backups=lora_reference_backups,
        )

        cls._register_civitai_provider()

    @classmethod
    def _register_pending_provider(cls) -> None:
        """Register the PRIMARY's pending-queue (beta) models under the ``"pending"`` source.

        Beta is opt-in via ``HORDELIB_BETA_MODEL_CATEGORIES`` / ``HORDELIB_BETA_MODELS_API_KEY``;
        when not configured this is a no-op. See :mod:`hordelib.beta_models`.
        """
        from hordelib.beta_models import build_pending_provider

        provider = build_pending_provider()
        if provider is None:
            return

        ModelReferenceManager.get_instance().register_provider(provider, replace=True)

    @classmethod
    def _register_civitai_provider(cls) -> None:
        """Expose the loaded LoRA/TI managers through the reference manager's ``"civitai"`` source.

        Registering a :class:`~hordelib.model_manager.civitai_provider.CivitaiModelProvider` lets
        consumers read LoRA/TI records via ``model_reference_manager.query(category, source="civitai")``
        alongside every other category, instead of reaching into the managers directly. Skipped when
        neither manager was loaded.
        """
        from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORY

        from hordelib.model_manager.civitai_provider import (
            CivitaiModelProvider,
            SupportsCurrentRecords,
        )

        managers_by_category: dict[MODEL_REFERENCE_CATEGORY, SupportsCurrentRecords] = {}
        if cls.manager.lora is not None:
            managers_by_category[MODEL_REFERENCE_CATEGORY.lora] = cls.manager.lora
        if cls.manager.ti is not None:
            managers_by_category[MODEL_REFERENCE_CATEGORY.ti] = cls.manager.ti

        if not managers_by_category:
            return

        ModelReferenceManager.get_instance().register_provider(
            CivitaiModelProvider(managers_by_category),
            replace=True,
        )

    @classmethod
    def unload_model_managers(
        cls,
        managers_to_unload: Iterable[str | MODEL_REFERENCE_CATEGORY | type[BaseModelManager]],
    ):
        cls.manager.unload_model_managers(managers_to_unload)

    @staticmethod
    def preload_annotators() -> bool:
        """Preload all controlnet annotators (the comfyui_controlnet_aux detector checkpoints).

        Requires hordelib.initialise() to have completed. A HordeLib instance is constructed
        on demand if needed, so the custom nodes are guaranteed to be loaded.

        Returns:
            bool: True if the annotators downloaded (or were already present) and run correctly.
        """
        return download_all_controlnet_annotators()
