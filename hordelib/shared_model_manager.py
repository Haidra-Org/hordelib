# shared_model_manager.py
import asyncio
from collections.abc import Iterable
from multiprocessing.synchronize import Lock as multiprocessing_lock
from typing import Self

import torch
from horde_model_reference import (
    MODEL_REFERENCE_CATEGORY,
    ModelReferenceManager,
    PrefetchStrategy,
    horde_model_reference_settings,
)
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

    _reference_offline: bool | None = None
    """Explicit offline override set by :func:`hordelib.initialise`. ``None`` means defer to \
    ``horde_model_reference_settings.offline`` (the ``HORDE_MODEL_REFERENCE_OFFLINE`` env var)."""

    @classmethod
    def _resolve_reference_offline(cls, explicit: bool | None) -> bool:
        """Resolve whether the reference manager should be offline (read-only, never download).

        Precedence: an explicit per-call argument, then the process-wide override set by
        ``initialise(reference_offline=...)``, then ``horde_model_reference_settings.offline``.
        """
        if explicit is not None:
            return explicit
        if cls._reference_offline is not None:
            return cls._reference_offline
        return horde_model_reference_settings.offline

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
        reference_offline: bool | None = None,
        adhoc_read_only: bool = False,
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
            reference_offline (bool | None, optional): If True, construct the reference manager in \
                offline mode (read references from local disk only, never download). If None, defers \
                to the process-wide override / ``HORDE_MODEL_REFERENCE_OFFLINE``. Defaults to None.
            adhoc_read_only (bool, optional): When True, the ad-hoc CivitAI managers (LoRA and TI) are \
                constructed read-only. A read-only ad-hoc manager never writes its reference, downloads \
                weights, or evicts, and construction itself performs no writes; any mutating call raises \
                ReadOnlyModelManagerError. Lets a consumer process that must never write (an inference \
                child) get that enforcement. Non-ad-hoc managers are unaffected. Defaults to False.
        """
        if cls.manager is None:
            cls.manager = ModelManager()

        # ModelReferenceManager is a singleton; subsequent calls return the same instance.
        # hordelib does not own the download policy: when a subprocess (or the worker) has already
        # constructed the singleton, we reuse it as-is. We only create one ourselves when absent, and
        # even then we never force a network download if offline is requested - the parent process is
        # responsible for downloading and writing the reference files to disk.
        try:
            if not ModelReferenceManager.has_instance():
                if cls._resolve_reference_offline(reference_offline):
                    cls.model_reference_manager = ModelReferenceManager(
                        offline=True,
                        prefetch_strategy=PrefetchStrategy.NONE,
                    )
                else:
                    cls.model_reference_manager = ModelReferenceManager(
                        prefetch_strategy=PrefetchStrategy.DEFERRED,
                    )
                    _await_prefetch(cls.model_reference_manager)
            else:
                # Reuse a manager constructed elsewhere (e.g. the worker pre-built an offline one).
                cls.model_reference_manager = ModelReferenceManager.get_instance()
        except Exception as e:
            logger.exception("Failed to initialize model reference manager")
            raise RuntimeError("Failed to initialize model reference manager") from e

        # Register the pending/beta and annotator providers before constructing managers: each manager
        # loads its database in __init__, so a manager whose records come from a provider (the annotator
        # manager reads the comfyui_controlnet_aux source) needs that provider already registered, and
        # beta_source_for only selects the pending provider when it is already registered.
        cls._register_pending_provider()
        cls._register_annotator_provider()

        cls.manager.init_model_managers(
            managers_to_load,
            multiprocessing_lock=multiprocessing_lock,
            lora_reference_backups=lora_reference_backups,
            adhoc_read_only=adhoc_read_only,
        )

        cls._register_civitai_provider()

    @classmethod
    def _register_annotator_provider(cls) -> None:
        """Expose the ControlNet annotators as the queryable ``"comfyui_controlnet_aux"`` source.

        The installed ``comfyui_controlnet_aux`` is the authority on which annotator checkpoints exist; this
        registers them as a first-class provider so consumers can read them via
        ``model_reference_manager.query(controlnet_annotator, source="comfyui_controlnet_aux")`` rather than
        treating the package's internal download as an opaque side-channel. The ``controlnet_annotator``
        category is supplied by the pinned ``horde_model_reference``.
        """
        from hordelib.model_manager.annotator_provider import AnnotatorModelProvider

        ModelReferenceManager.get_instance().register_provider(AnnotatorModelProvider(), replace=True)

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
