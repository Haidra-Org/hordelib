"""Home for the controller class ModelManager, and related meta information."""

import os
import threading
from collections.abc import Callable, Iterable
from multiprocessing.synchronize import Lock as multiprocessing_lock

import torch
from horde_model_reference import MODEL_REFERENCE_CATEGORY, category_folder
from loguru import logger

# from hordelib.model_manager.diffusers import DiffusersModelManager
from hordelib.execution.component_cache import ComponentCache, component_cache_budget_mb
from hordelib.model_manager.base import BaseModelManager
from hordelib.model_manager.civitai_adhoc import CivitaiAdhocModelManager
from hordelib.model_manager.codeformer import CodeFormerModelManager
from hordelib.model_manager.compvis import CompVisModelManager
from hordelib.model_manager.controlnet import ControlNetModelManager
from hordelib.model_manager.controlnet_annotator import ControlNetAnnotatorModelManager
from hordelib.model_manager.esrgan import EsrganModelManager
from hordelib.model_manager.gfpgan import GfpganModelManager
from hordelib.model_manager.lora import LoraModelManager
from hordelib.model_manager.miscellaneous import MiscellaneousModelManager
from hordelib.model_manager.safety_checker import SafetyCheckerModelManager
from hordelib.model_manager.ti import TextualInversionModelManager

MODEL_MANAGERS_TYPE_LOOKUP: dict[MODEL_REFERENCE_CATEGORY | str, type[BaseModelManager]] = {
    MODEL_REFERENCE_CATEGORY.codeformer: CodeFormerModelManager,
    MODEL_REFERENCE_CATEGORY.image_generation: CompVisModelManager,
    MODEL_REFERENCE_CATEGORY.controlnet: ControlNetModelManager,
    MODEL_REFERENCE_CATEGORY.controlnet_annotator: ControlNetAnnotatorModelManager,
    MODEL_REFERENCE_CATEGORY.esrgan: EsrganModelManager,
    MODEL_REFERENCE_CATEGORY.gfpgan: GfpganModelManager,
    MODEL_REFERENCE_CATEGORY.safety_checker: SafetyCheckerModelManager,
    MODEL_REFERENCE_CATEGORY.lora: LoraModelManager,
    MODEL_REFERENCE_CATEGORY.ti: TextualInversionModelManager,
    MODEL_REFERENCE_CATEGORY.miscellaneous: MiscellaneousModelManager,
}
"""A lookup table for the `BaseModelManager` types."""

ALL_MODEL_MANAGER_TYPES: list[type[BaseModelManager]] = list(MODEL_MANAGERS_TYPE_LOOKUP.values())


# Migrating to horde_model_reference v5 renamed some categories (e.g. the old "compvis" category is
# now "image_generation") while keeping the original on-disk folder names for compatibility. Callers
# and existing tests still address managers by those legacy folder names, so map each folder name that
# differs from its category value back to the canonical category.
MODEL_MANAGER_NAME_ALIASES: dict[str, MODEL_REFERENCE_CATEGORY] = {
    folder: category
    for category in MODEL_MANAGERS_TYPE_LOOKUP
    if isinstance(category, MODEL_REFERENCE_CATEGORY)
    and (folder := category_folder(category)) is not None
    and folder != category.value
}
"""Legacy folder-name aliases (e.g. ``"compvis"``) mapped to their `MODEL_REFERENCE_CATEGORY`."""


def resolve_manager_alias(
    identifier: str | MODEL_REFERENCE_CATEGORY | type[BaseModelManager],
) -> str | MODEL_REFERENCE_CATEGORY | type[BaseModelManager]:
    """Map a legacy folder-style manager name (e.g. ``"compvis"``) to its canonical category.

    Identifiers that aren't legacy aliases (categories, manager types, unrecognised strings) are
    returned unchanged.
    """
    if isinstance(identifier, str):
        return MODEL_MANAGER_NAME_ALIASES.get(identifier, identifier)
    return identifier


class ModelManager:
    """Controller class for all managers which extend `BaseModelManager`."""

    disk_read_mutex = threading.Lock()

    # These properties are here with compatibility in mind, but they may survive for a while.
    @property
    def codeformer(self) -> CodeFormerModelManager | None:
        """The CodeFormer model manager instance. Returns `None` if not loaded."""
        found_mm = self.get_model_manager_instance(CodeFormerModelManager)
        return found_mm if isinstance(found_mm, CodeFormerModelManager) else None

    @property
    def compvis(self) -> CompVisModelManager | None:
        """The CompVis model manager instance. Returns `None` if not loaded."""
        found_mm = self.get_model_manager_instance(CompVisModelManager)
        return found_mm if isinstance(found_mm, CompVisModelManager) else None

    @property
    def controlnet(self) -> ControlNetModelManager | None:
        """The ControlNet model manager instance. Returns `None` if not loaded."""
        found_mm = self.get_model_manager_instance(ControlNetModelManager)
        return found_mm if isinstance(found_mm, ControlNetModelManager) else None

    @property
    def controlnet_annotator(self) -> ControlNetAnnotatorModelManager | None:
        """The ControlNet annotator model manager instance. Returns `None` if not loaded."""
        found_mm = self.get_model_manager_instance(ControlNetAnnotatorModelManager)
        return found_mm if isinstance(found_mm, ControlNetAnnotatorModelManager) else None

    @property
    def esrgan(self) -> EsrganModelManager | None:
        """The ESRGAN model manager instance. Returns `None` if not loaded."""
        found_mm = self.get_model_manager_instance(EsrganModelManager)
        return found_mm if isinstance(found_mm, EsrganModelManager) else None

    @property
    def gfpgan(self) -> GfpganModelManager | None:
        """The GFPGAN model manager instance Returns `None` if not loaded.."""
        found_mm = self.get_model_manager_instance(GfpganModelManager)
        return found_mm if isinstance(found_mm, GfpganModelManager) else None

    @property
    def safety_checker(self) -> SafetyCheckerModelManager | None:
        """The SafetyChecker model manager instance. Returns `None` if not loaded."""
        found_mm = self.get_model_manager_instance(SafetyCheckerModelManager)
        return found_mm if isinstance(found_mm, SafetyCheckerModelManager) else None

    @property
    def lora(self) -> LoraModelManager | None:
        """The Lora model manager instance. Returns `None` if not loaded."""
        found_mm = self.get_model_manager_instance(LoraModelManager)
        return found_mm if isinstance(found_mm, LoraModelManager) else None

    @property
    def ti(self) -> TextualInversionModelManager | None:
        """The Textual Inversion model manager instance. Returns `None` if not loaded."""
        found_mm = self.get_model_manager_instance(TextualInversionModelManager)
        return found_mm if isinstance(found_mm, TextualInversionModelManager) else None

    @property
    def miscellaneous(self) -> MiscellaneousModelManager | None:
        """The miscellaneous model manager instance. Returns `None` if not loaded."""
        found_mm = self.get_model_manager_instance(MiscellaneousModelManager)
        return found_mm if isinstance(found_mm, MiscellaneousModelManager) else None

    @property
    def models(self) -> dict:
        """All model manager's internal dictionaries of models, loaded from model database JSON files."""
        _models: dict = {}
        for model_manager in self.active_model_managers:
            _models.update(model_manager.model_reference)
        return _models

    _models_in_ram: ComponentCache
    """The in-RAM component cache: a content-addressed, MB-budgeted LRU of loaded model components.

    Keyed by :class:`~hordelib.execution.component_cache.ComponentCacheKey`. The budget is read once per
    process from ``HORDE_COMPONENT_CACHE_MB`` at construction; a budget of ``0`` keeps a single component
    resident (the historical single-slot behaviour).
    """

    def get_available_models(
        self,
        mm_include: Iterable[str | MODEL_REFERENCE_CATEGORY | type[BaseModelManager]] | None = None,
        mm_exclude: Iterable[str | MODEL_REFERENCE_CATEGORY | type[BaseModelManager]] | None = None,
    ) -> list[str]:
        """All models for which information exists, and for which a download attempt could be made."""
        all_available_models: list[str] = []

        resolved_include_managers = self.get_model_manager_instances(mm_include)
        resolved_exclude_managers = self.get_model_manager_instances(mm_exclude)

        for model_manager in self.active_model_managers:
            if resolved_include_managers and model_manager not in resolved_include_managers:
                continue
            if model_manager in resolved_exclude_managers:
                continue

            all_available_models.extend(model_manager.available_models)

        return all_available_models

    available_models = property(get_available_models)

    def is_model_available(self, model_name: str) -> bool:
        """Checks if the model is available for download.

        Args:
            model_name (str): The model to check.

        Returns:
            bool: Whether the model is available for download.
        """
        for model_manager in self.active_model_managers:
            if model_name in model_manager.available_models:
                return True
        return False

    active_model_managers: list[BaseModelManager]

    def __init__(
        self,
    ):
        """Create a new instance of model manager."""
        self.cuda_available = torch.cuda.is_available()
        """DEPRECATED: Use `torch.cuda.is_available()` instead."""

        # We use this to serialise disk reads as no point in
        # doing more than one at a time as this will slow down the sequential read op.
        self.disk_read_mutex = threading.Lock()
        self.active_model_managers = []

        # The component budget is fixed for the life of the process at construction; a fresh manager (e.g. a
        # re-initialised singleton) re-reads it so a changed environment takes effect on the next init.
        self._models_in_ram = ComponentCache(component_cache_budget_mb())

    def init_model_managers(
        self,
        managers_to_load: Iterable[str | MODEL_REFERENCE_CATEGORY | type[BaseModelManager]],
        multiprocessing_lock: multiprocessing_lock | None = None,
        lora_reference_backups: bool | None = None,
        adhoc_read_only: bool = False,
    ) -> None:
        for raw_manager_to_load in managers_to_load:
            manager_to_load = resolve_manager_alias(raw_manager_to_load)
            resolve_manager_to_load_type: type[BaseModelManager] | None = None
            if isinstance(manager_to_load, type) and issubclass(manager_to_load, BaseModelManager):
                if manager_to_load not in MODEL_MANAGERS_TYPE_LOOKUP.values():
                    logger.debug(
                        "Attempted to load unknown model manager",
                        manager_identifier=str(manager_to_load),
                    )
                    continue
                resolve_manager_to_load_type = manager_to_load
            elif manager_to_load in MODEL_MANAGERS_TYPE_LOOKUP.keys():
                resolve_manager_to_load_type = MODEL_MANAGERS_TYPE_LOOKUP[manager_to_load]
            else:
                logger.warning(
                    "Attempted to load unknown model manager",
                    manager_identifier=str(manager_to_load),
                )
                continue

            if any(mm for mm in self.active_model_managers if isinstance(mm, resolve_manager_to_load_type)):
                logger.debug(
                    "Attempted to load model manager which is already active",
                    manager_type=resolve_manager_to_load_type.__name__,
                )
                continue
            manager_kwargs: dict = {
                "multiprocessing_lock": multiprocessing_lock,
                "civitai_api_token": os.getenv("CIVIT_API_TOKEN", None),
            }
            if resolve_manager_to_load_type is LoraModelManager and lora_reference_backups is not None:
                manager_kwargs["reference_backups"] = lora_reference_backups
            # read_only is an ad-hoc CivitAI concept only; the other managers do not accept it.
            if issubclass(resolve_manager_to_load_type, CivitaiAdhocModelManager):
                manager_kwargs["read_only"] = adhoc_read_only

            self.active_model_managers.append(resolve_manager_to_load_type(**manager_kwargs))

    def unload_model_managers(
        self,
        managers_to_unload: Iterable[str | MODEL_REFERENCE_CATEGORY | type[BaseModelManager]],
    ):
        for raw_manager_to_unload in managers_to_unload:
            manager_to_unload = resolve_manager_alias(raw_manager_to_unload)
            resolved_manager_to_unload_type: type[BaseModelManager] | None = None
            if isinstance(manager_to_unload, type) and issubclass(manager_to_unload, BaseModelManager):
                if manager_to_unload not in MODEL_MANAGERS_TYPE_LOOKUP.values():
                    logger.warning(
                        "Attempted to unload unknown model manager",
                        manager_identifier=str(manager_to_unload),
                    )
                    continue
                resolved_manager_to_unload_type = manager_to_unload
            elif manager_to_unload in MODEL_MANAGERS_TYPE_LOOKUP.keys():
                resolved_manager_to_unload_type = MODEL_MANAGERS_TYPE_LOOKUP[manager_to_unload]
            else:
                logger.warning(
                    "Attempted to unload unknown model manager",
                    manager_identifier=str(manager_to_unload),
                )
                continue

            if not [mm for mm in self.active_model_managers if isinstance(mm, resolved_manager_to_unload_type)]:
                logger.warning(
                    "Attempted to unload model manager which is not active",
                    manager_type=(
                        resolved_manager_to_unload_type.__name__
                        if resolved_manager_to_unload_type is not None
                        else None
                    ),
                )
                continue
            self.active_model_managers = [
                mm for mm in self.active_model_managers if not isinstance(mm, resolved_manager_to_unload_type)
            ]

    def reload_database(self) -> None:
        """Completely resets the `BaseModelManager` classes, and forces each to re-init."""
        for model_manager in self.active_model_managers:
            if model_manager is not None:
                model_manager.load_model_database()

    def download_model(
        self,
        model_name: str,
        callback: Callable[[int, int], None] | None = None,
    ) -> bool | None:
        """Looks across all `BaseModelManager` for the model_name and attempts to download it.

        Args:
            model_name (str): The name of the model to find and attempt to download.

        Returns:
            bool | None: The success of the download. If `None`, the model_name was not found.
        """
        for model_manager_type in MODEL_MANAGERS_TYPE_LOOKUP:
            model_manager: BaseModelManager | None = self.get_model_manager_instance(model_manager_type)
            if model_manager is None:
                continue
            if model_name not in model_manager.model_reference:
                continue

            return model_manager.download_model(model_name, callback=callback)
        logger.warning("Model not found", model_name=model_name)
        return None  # XXX if the download fails, the file causes issues # FIXME

    def download_all(
        self,
        callback: Callable[[int, int], None] | None = None,
    ) -> None:
        """Attempts to download all available models for all `BaseModelManager` types."""
        for model_manager_type in MODEL_MANAGERS_TYPE_LOOKUP:
            model_manager: BaseModelManager | None = self.get_model_manager_instance(model_manager_type)
            if model_manager is None:
                continue

            model_manager.download_all_models(callback=callback)

    def validate_model(
        self,
        model_name: str,
        skip_checksum: bool = False,
    ) -> bool | None:
        """Runs a integrity check against the model specified.

        Args:
            model_name (str): The model to check.
            skip_checksum (bool, optional): Whether to skip the SHA/MD5 check. Defaults to False.

        Returns:
            bool | None: The result of the validation. If `None`, the model was not found.
        """
        for model_manager_type in MODEL_MANAGERS_TYPE_LOOKUP.values():
            model_manager: BaseModelManager | None = self.get_model_manager_instance(model_manager_type)
            if model_manager is None:
                continue
            if model_name in model_manager.model_reference:
                return model_manager.validate_model(model_name, skip_checksum)
        return None

    def taint_models(self, models: list[str]) -> None:
        """Marks a list of models to be unavailable.

        Args:
            models (list[str]): The list of models to mark.
        """
        for model_manager_type in MODEL_MANAGERS_TYPE_LOOKUP:
            model_manager: BaseModelManager = getattr(self, model_manager_type)
            if model_manager is None:
                continue
            if any(model in model_manager.model_reference for model in models):
                model_manager.taint_models(models)

    def get_available_models_by_types(
        self,
        mm_include: Iterable[str | MODEL_REFERENCE_CATEGORY | type[BaseModelManager]] | None = None,
        mm_exclude: Iterable[str | MODEL_REFERENCE_CATEGORY | type[BaseModelManager]] | None = None,
    ) -> list[str]:
        return self.get_available_models(mm_include, mm_exclude)

    def get_model_manager_instance(
        self,
        mm_type: str | MODEL_REFERENCE_CATEGORY | type[BaseModelManager],
    ) -> BaseModelManager | None:
        found_manager = self.get_model_manager_instances([mm_type])
        return found_manager[0] if len(found_manager) > 0 else None

    def get_model_manager_instances(
        self,
        mm_types: Iterable[str | MODEL_REFERENCE_CATEGORY | type[BaseModelManager]] | None = None,
    ) -> list[BaseModelManager]:
        """Returns a set of model managers based on the input Iterable of model manager types.

        Args:
            mm_types (Iterable[str | MODEL_REFERENCE_CATEGORY | type[BaseModelManager]], optional): The Iterable of
            model manager types to resolve. Defaults to None.

        Returns:
            list[BaseModelManager]: A list of the requested model managers.
        """
        if not mm_types:
            return []

        if not isinstance(mm_types, list) and not isinstance(mm_types, tuple) and not isinstance(mm_types, set):
            mm_types = [mm_types]  # type: ignore

        active_model_managers_types = [type(model_manager) for model_manager in self.active_model_managers]

        resolved_types = []
        active_model_managers_types_names = [mm_type.__name__ for mm_type in active_model_managers_types]
        for raw_mm_type in mm_types:  # type: ignore
            mm_type = resolve_manager_alias(raw_mm_type)
            if isinstance(mm_type, type) and mm_type in active_model_managers_types:
                resolved_types.append(mm_type)
                continue

            if isinstance(mm_type, type) and mm_type.__name__ in active_model_managers_types_names:
                resolved_types.append(mm_type)
                logger.debug(
                    "Resolved model manager by name",
                    manager_name=mm_type.__name__,
                    warning="May not match desired manager",
                )
                continue

            if not isinstance(mm_type, str) or mm_type not in MODEL_MANAGERS_TYPE_LOOKUP:
                logger.debug(
                    "Attempted to reference unknown model manager",
                    manager_identifier=str(mm_type),
                )
                continue

            if mm_type in MODEL_MANAGERS_TYPE_LOOKUP:
                if MODEL_MANAGERS_TYPE_LOOKUP[mm_type] not in active_model_managers_types:
                    logger.debug(
                        "Attempted to reference model manager which is not active",
                        manager_identifier=str(mm_type),
                    )
                    continue

                resolved_types.append(MODEL_MANAGERS_TYPE_LOOKUP[mm_type])

        return [mm for mm in self.active_model_managers if type(mm) in resolved_types]
