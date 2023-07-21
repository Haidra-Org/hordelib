"""Home for the controller class ModelManager, and related meta information."""
import copy
import os
import threading
from typing import Iterable

import torch
from loguru import logger

from hordelib.consts import EXCLUDED_MODEL_NAMES, MODEL_CATEGORY_NAMES

# from hordelib.model_manager.diffusers import DiffusersModelManager
from hordelib.model_manager.base import BaseModelManager
from hordelib.model_manager.blip import BlipModelManager
from hordelib.model_manager.clip import ClipModelManager
from hordelib.model_manager.codeformer import CodeFormerModelManager
from hordelib.model_manager.compvis import CompVisModelManager
from hordelib.model_manager.controlnet import ControlNetModelManager
from hordelib.model_manager.esrgan import EsrganModelManager
from hordelib.model_manager.gfpgan import GfpganModelManager
from hordelib.model_manager.lora import LoraModelManager
from hordelib.model_manager.safety_checker import SafetyCheckerModelManager
from hordelib.settings import UserSettings

MODEL_MANAGERS_TYPE_LOOKUP: dict[MODEL_CATEGORY_NAMES | str, type[BaseModelManager]] = {
    MODEL_CATEGORY_NAMES.codeformer: CodeFormerModelManager,
    MODEL_CATEGORY_NAMES.compvis: CompVisModelManager,
    MODEL_CATEGORY_NAMES.controlnet: ControlNetModelManager,
    # MODEL_CATEGORY_NAMES.diffusers: DiffusersModelManager,
    MODEL_CATEGORY_NAMES.esrgan: EsrganModelManager,
    MODEL_CATEGORY_NAMES.gfpgan: GfpganModelManager,
    MODEL_CATEGORY_NAMES.safety_checker: SafetyCheckerModelManager,
    MODEL_CATEGORY_NAMES.lora: LoraModelManager,
    MODEL_CATEGORY_NAMES.blip: BlipModelManager,
    MODEL_CATEGORY_NAMES.clip: ClipModelManager,
}
"""A lookup table for the `BaseModelManager` types."""

ALL_MODEL_MANAGER_TYPES: list[type[BaseModelManager]] = list(MODEL_MANAGERS_TYPE_LOOKUP.values())


class ModelManager:
    """Controller class for all managers which extend `BaseModelManager`."""

    # These properties are here with compatibility in mind, but they may survive for a while.
    @property
    def codeformer(self) -> CodeFormerModelManager | None:
        """The CodeFormer model manager instance. Returns `None` if not loaded."""
        found_mm = self.get_mm_pointer(CodeFormerModelManager)
        return found_mm if isinstance(found_mm, CodeFormerModelManager) else None

    @property
    def compvis(self) -> CompVisModelManager | None:
        """The CompVis model manager instance. Returns `None` if not loaded."""
        found_mm = self.get_mm_pointer(CompVisModelManager)
        return found_mm if isinstance(found_mm, CompVisModelManager) else None

    @property
    def controlnet(self) -> ControlNetModelManager | None:
        """The ControlNet model manager instance. Returns `None` if not loaded."""
        found_mm = self.get_mm_pointer(ControlNetModelManager)
        return found_mm if isinstance(found_mm, ControlNetModelManager) else None

    @property
    def esrgan(self) -> EsrganModelManager | None:
        """The ESRGAN model manager instance. Returns `None` if not loaded."""
        found_mm = self.get_mm_pointer(EsrganModelManager)
        return found_mm if isinstance(found_mm, EsrganModelManager) else None

    @property
    def gfpgan(self) -> GfpganModelManager | None:
        """The GFPGAN model manager instance Returns `None` if not loaded.."""
        found_mm = self.get_mm_pointer(GfpganModelManager)
        return found_mm if isinstance(found_mm, GfpganModelManager) else None

    @property
    def safety_checker(self) -> SafetyCheckerModelManager | None:
        """The SafetyChecker model manager instance. Returns `None` if not loaded."""
        found_mm = self.get_mm_pointer(SafetyCheckerModelManager)
        return found_mm if isinstance(found_mm, SafetyCheckerModelManager) else None

    @property
    def lora(self) -> LoraModelManager | None:
        """The Lora model manager instance. Returns `None` if not loaded."""
        found_mm = self.get_mm_pointer(LoraModelManager)
        return found_mm if isinstance(found_mm, LoraModelManager) else None

    @property
    def blip(self) -> BlipModelManager | None:
        """The Blip model manager instance. Returns `None` if not loaded."""
        found_mm = self.get_mm_pointer(BlipModelManager)
        return found_mm if isinstance(found_mm, BlipModelManager) else None

    @property
    def clip(self) -> ClipModelManager | None:
        """The Clip model manager instance. Returns `None` if not loaded."""
        found_mm = self.get_mm_pointer(ClipModelManager)
        return found_mm if isinstance(found_mm, ClipModelManager) else None

    @property
    def models(self) -> dict:
        """All model manager's internal dictionaries of models, loaded from model database JSON files."""
        _models: dict = {}
        for model_manager in self.active_model_managers:
            model_manager.model_reference
            _models.update(model_manager.model_reference)
        return _models

    def get_model_directory(self, suffix=""):
        model_dir = (
            os.path.join(UserSettings.get_model_directory(), suffix) if suffix else UserSettings.get_model_directory()
        )
        return model_dir

    def get_available_models(
        self,
        mm_include: Iterable[str | MODEL_CATEGORY_NAMES | type[BaseModelManager]] | None = None,
        mm_exclude: Iterable[str | MODEL_CATEGORY_NAMES | type[BaseModelManager]] | None = None,
    ) -> list[str]:
        """All models for which information exists, and for which a download attempt could be made."""
        all_available_models: list[str] = []

        resolved_include_managers = self.get_mm_pointers(mm_include)
        resolved_exclude_managers = self.get_mm_pointers(mm_exclude)

        for model_manager in self.active_model_managers:
            if resolved_include_managers and model_manager not in resolved_include_managers:
                continue
            if model_manager in resolved_exclude_managers:
                continue

            all_available_models.extend(model_manager.available_models)

        return all_available_models

    available_models = property(get_available_models)

    def get_loaded_models(
        self,
        mm_include: Iterable[str | MODEL_CATEGORY_NAMES | type[BaseModelManager]] | None = None,
        mm_exclude: Iterable[str | MODEL_CATEGORY_NAMES | type[BaseModelManager]] | None = None,
    ) -> dict[str, dict]:
        """All models for which have successfully loaded across all `BaseModelManager` types."""
        all_loaded_models: dict[str, dict] = {}

        resolved_include_managers = self.get_mm_pointers(mm_include)
        resolved_exclude_managers = self.get_mm_pointers(mm_exclude)

        for model_manager in self.active_model_managers:
            if resolved_include_managers and model_manager not in resolved_include_managers:
                continue
            if model_manager in resolved_exclude_managers:
                continue

            all_loaded_models.update(model_manager.get_loaded_models())

        return all_loaded_models

    loaded_models = property(get_loaded_models)

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

    def get_model_copy(self, model_name, model_component=None):
        if not model_component:
            return copy.deepcopy(self.loaded_models[model_name])
        else:
            return copy.deepcopy(self.loaded_models[model_name][model_component])

    def init_model_managers(
        self,
        managers_to_load: Iterable[str | MODEL_CATEGORY_NAMES | type[BaseModelManager]],
    ) -> None:
        for manager_to_load in managers_to_load:
            resolve_manager_to_load_type: type[BaseModelManager] | None = None
            if isinstance(manager_to_load, type) and issubclass(manager_to_load, BaseModelManager):
                if manager_to_load not in MODEL_MANAGERS_TYPE_LOOKUP.values():
                    logger.warning(f"Attempted to load a model manager which doesn't exist: '{manager_to_load}'.")
                    continue
                resolve_manager_to_load_type = manager_to_load
            elif manager_to_load in MODEL_MANAGERS_TYPE_LOOKUP.keys():
                resolve_manager_to_load_type = MODEL_MANAGERS_TYPE_LOOKUP[manager_to_load]
            else:
                logger.warning(f"Attempted to load a model manager which doesn't exist: '{manager_to_load}'.")
                continue

            if any(mm for mm in self.active_model_managers if isinstance(mm, resolve_manager_to_load_type)):
                logger.warning(
                    f"Attempted to load a model manager which is already loaded: '{resolve_manager_to_load_type}'.",
                )
                continue
            self.active_model_managers.append(resolve_manager_to_load_type())

    def unload_model_managers(
        self,
        managers_to_unload: Iterable[str | MODEL_CATEGORY_NAMES | type[BaseModelManager]],
    ):
        for manager_to_unload in managers_to_unload:
            resolved_manager_to_unload_type: type[BaseModelManager] | None = None
            if isinstance(manager_to_unload, type) and issubclass(manager_to_unload, BaseModelManager):
                if manager_to_unload not in MODEL_MANAGERS_TYPE_LOOKUP.values():
                    logger.warning(f"Attempted to unload a model manager which doesn't exist: '{manager_to_unload}'.")
                    continue
                resolved_manager_to_unload_type = manager_to_unload
            elif manager_to_unload in MODEL_MANAGERS_TYPE_LOOKUP.keys():
                resolved_manager_to_unload_type = MODEL_MANAGERS_TYPE_LOOKUP[manager_to_unload]
            else:
                logger.warning(f"Attempted to unload a model manager which doesn't exist: '{manager_to_unload}'.")
                continue

            if not [mm for mm in self.active_model_managers if isinstance(mm, resolved_manager_to_unload_type)]:
                logger.warning(
                    f"Attempted to unload a model manager which is not loaded: '{resolved_manager_to_unload_type}'.",
                )
                continue
            self.active_model_managers = [
                mm for mm in self.active_model_managers if not isinstance(mm, resolved_manager_to_unload_type)
            ]

    def reload_database(self) -> None:
        """Completely resets the `BaseModelManager` classes, and forces each to re-init."""
        for model_manager in self.active_model_managers:
            if model_manager is not None:
                model_manager.loadModelDatabase()

    def download_model(self, model_name: str) -> bool | None:
        """Looks across all `BaseModelManager` for the model_name and attempts to download it.

        Args:
            model_name (str): The name of the model to find and attempt to download.

        Returns:
            bool | None: The success of the download. If `None`, the model_name was not found.
        """
        for model_manager_type in MODEL_MANAGERS_TYPE_LOOKUP:
            model_manager: BaseModelManager | None = self.get_mm_pointer(model_manager_type)
            if model_manager is None:
                continue
            if model_name not in model_manager.model_reference:
                continue

            return model_manager.download_model(model_name)
        logger.warning(f"Model '{model_name}' not found!")
        return None  # XXX if the download fails, the file causes issues # FIXME

    def download_all(self) -> None:
        """Attempts to download all available models for all `BaseModelManager` types."""
        for model_manager_type in MODEL_MANAGERS_TYPE_LOOKUP:
            model_manager: BaseModelManager | None = self.get_mm_pointer(model_manager_type)
            if model_manager is None:
                continue

            model_manager.download_all_models()

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
            model_manager: BaseModelManager | None = self.get_mm_pointer(model_manager_type)
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

    def unload_model(self, model_name: str) -> bool | None:
        """Unloads the target model.

        Args:
            model_name (str): The model name to remove.

        Returns:
            bool | None: The result of the unloading. If `None`, the model was not found.
        """
        for model_manager_type in MODEL_MANAGERS_TYPE_LOOKUP.values():
            model_manager: BaseModelManager | None = self.get_mm_pointer(model_manager_type)
            if model_manager is None:
                continue
            if model_name in model_manager.model_reference or model_manager.is_local_model(model_name):
                return model_manager.unload_model(model_name)
        return None

    def move_from_disk_cache(self, model_name, model, clip, vae) -> bool | None:
        """Moves the given model back into ram.

        Args:
            model_name (str): The model name to remove.
            model (model): the model data
            clip (clip): the clip model data
            var (var): the var model data
        """
        for model_manager_type in MODEL_MANAGERS_TYPE_LOOKUP:
            model_manager: BaseModelManager | None = self.get_mm_pointer(model_manager_type)
            if model_manager is None:
                continue
            if model_name in model_manager.model_reference or model_manager.is_local_model(model_name):
                return model_manager.move_from_disk_cache(model_name, model, clip, vae)
        return None

    def get_loaded_models_names(
        self,
        mm_include: Iterable[str | MODEL_CATEGORY_NAMES | type[BaseModelManager]] | None = None,
        mm_exclude: Iterable[str | MODEL_CATEGORY_NAMES | type[BaseModelManager]] | None = None,
    ) -> list:
        """
        Returns:
            list: All currently loaded models.
        """
        loaded_models = self.get_loaded_models(mm_include, mm_exclude)
        return list(loaded_models.keys())

    def is_model_loaded(self, model_name) -> bool:
        # TODO: This function should indicate if the model is even valid
        return model_name in self.loaded_models

    def get_available_models_by_types(
        self,
        mm_include: Iterable[str | MODEL_CATEGORY_NAMES | type[BaseModelManager]] | None = None,
        mm_exclude: Iterable[str | MODEL_CATEGORY_NAMES | type[BaseModelManager]] | None = None,
    ):
        return self.get_available_models(mm_include, mm_exclude)

    def count_available_models_by_types(
        self,
        model_types: Iterable[str | MODEL_CATEGORY_NAMES | type[BaseModelManager]] | None = None,
    ) -> int:
        return len(self.get_available_models_by_types(model_types))

    def ensure_memory_available(self, specific_type: type[BaseModelManager] | None = None) -> None:
        """Asserts minimum amount of RAM is available. Unloads models if necessary."""
        for model_manager_type in MODEL_MANAGERS_TYPE_LOOKUP:
            if specific_type and specific_type != model_manager_type:
                continue
            model_manager: BaseModelManager | None = self.get_mm_pointer(model_manager_type)
            if model_manager is None:
                continue
            model_manager.ensure_ram_available()
        return

    def load(
        self,
        model_name: str,
        half_precision: bool = True,
        gpu_id: int = 0,
        cpu_only: bool = False,
        local: bool = False,
    ) -> bool | None:
        """Loads the target model with the appropriate (loaded) `BaseModelManager` type.

        Args:
            model_name (str): Name of the model to load. See available_models for
            a list of available models.
            half_precision (bool, optional): If the model should be loaded in half precision.
            Defaults to True.
            gpu_id (int, optional): The id of the gpu to use. Defaults to 0.
            cpu_only (bool, optional): If should be loaded on the cpu.
            If True, half_precision will be set to False. Defaults to False.
            local (bool): model_name is a local filesystem filename of a model to load

        Returns:
            bool | None: The success of the load. If `None`, the model was not found.
        """
        if model_name in EXCLUDED_MODEL_NAMES:
            logger.warning(
                f"{model_name} is excluded from loading at this time. If this is unexpected, let us know on discord.",
            )
            return False

        if not self.cuda_available:
            cpu_only = True

        for model_manager in self.active_model_managers:
            if model_name in model_manager.model_reference or model_manager.is_local_model(model_name):
                return model_manager.load(
                    model_name=model_name,
                    half_precision=half_precision,
                    gpu_id=gpu_id,
                    cpu_only=cpu_only,
                )

        logger.error(f"{model_name} not found")
        return None

    def get_mm_pointer(
        self,
        mm_type: str | MODEL_CATEGORY_NAMES | type[BaseModelManager],
    ) -> BaseModelManager | None:
        found_manager = self.get_mm_pointers([mm_type])
        return found_manager[0] if len(found_manager) > 0 else None

    def get_mm_pointers(
        self,
        mm_types: Iterable[str | MODEL_CATEGORY_NAMES | type[BaseModelManager]] | None = None,
    ) -> list[BaseModelManager]:
        """Returns a set of model managers based on the input Iterable of model manager types.

        Args:
            mm_types (Iterable[str | MODEL_CATEGORY_NAMES | type[BaseModelManager]], optional): The Iterable of
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
        for mm_type in mm_types:  # type: ignore
            if isinstance(mm_type, type) and mm_type in active_model_managers_types:
                resolved_types.append(mm_type)
                continue

            if isinstance(mm_type, type) and mm_type.__name__ in active_model_managers_types_names:
                resolved_types.append(mm_type)
                logger.debug(
                    (
                        f"Found model manager by name: {mm_type.__name__}."
                        " This may not be the model manager you are looking for.",
                    ),
                )
                continue

            if not isinstance(mm_type, str) or mm_type not in MODEL_MANAGERS_TYPE_LOOKUP:
                logger.debug(f"Attempted to reference a model manager which doesn't exist: '{mm_type}'.")
                continue

            if mm_type in MODEL_MANAGERS_TYPE_LOOKUP:
                if MODEL_MANAGERS_TYPE_LOOKUP[mm_type] not in active_model_managers_types:
                    logger.debug(f"Attempted to reference a model manager which isn't loaded: '{mm_type}'.")
                    continue

                resolved_types.append(MODEL_MANAGERS_TYPE_LOOKUP[mm_type])

        return [mm for mm in self.active_model_managers if type(mm) in resolved_types]
