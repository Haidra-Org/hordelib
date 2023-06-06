"""Home for the controller class ModelManager, and related meta information."""
import copy
import os
import threading

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

MODEL_MANAGERS_TYPE_LOOKUP: dict[MODEL_CATEGORY_NAMES, type[BaseModelManager]] = {
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
"""Keys are `str` which represent attrs in `ModelManger`. Values are the corresponding `type`."""


class ModelManager:
    """Controller class for all managers which extend `BaseModelManager`."""

    codeformer: CodeFormerModelManager | None = None
    compvis: CompVisModelManager | None = None
    controlnet: ControlNetModelManager | None = None
    # diffusers: DiffusersModelManager | None = None
    esrgan: EsrganModelManager | None = None
    gfpgan: GfpganModelManager | None = None
    safety_checker: SafetyCheckerModelManager | None = None
    lora: LoraModelManager | None = None
    blip: BlipModelManager | None = None
    clip: ClipModelManager | None = None
    # XXX I think this can be reworked into an array of BaseModelManager instances

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
        mm_include: list[str | MODEL_CATEGORY_NAMES | type[BaseModelManager]] = None,
        mm_exclude: list[str | MODEL_CATEGORY_NAMES | type[BaseModelManager]] = None,
    ) -> list[str]:
        """All models for which information exists, and for which a download attempt could be made."""
        all_available_models: list[str] = []
        mm_include = self.get_mm_pointers(mm_include)
        mm_exclude = self.get_mm_pointers(mm_exclude)
        for model_manager in self.active_model_managers:
            if mm_include and model_manager not in mm_include:
                continue
            if mm_exclude and model_manager in mm_exclude:
                continue
            all_available_models.extend(model_manager.available_models)
        return all_available_models

    available_models = property(get_available_models)

    def get_loaded_models(
        self,
        mm_include: list[str | MODEL_CATEGORY_NAMES | type[BaseModelManager]] = None,
        mm_exclude: list[str | MODEL_CATEGORY_NAMES | type[BaseModelManager]] = None,
    ) -> dict[str, dict]:
        """All models for which have successfully loaded across all `BaseModelManager` types."""
        all_loaded_models: dict[str, dict] = {}
        mm_include = self.get_mm_pointers(mm_include)
        mm_exclude = self.get_mm_pointers(mm_exclude)
        for model_manager in self.active_model_managers:
            if mm_include and model_manager not in mm_include:
                continue
            if mm_exclude and model_manager in mm_exclude:
                continue
            all_loaded_models.update(model_manager.get_loaded_models())
        return all_loaded_models

    loaded_models = property(get_loaded_models)

    @property
    def active_model_managers(self) -> list[BaseModelManager]:
        """All loaded model managers."""
        all_model_managers = [
            self.compvis,
            # self.diffusers,
            self.esrgan,
            self.gfpgan,
            self.safety_checker,
            self.codeformer,
            self.controlnet,
            self.lora,
            self.blip,
            self.clip,
        ]
        # reset available models

        _active_model_managers: list[BaseModelManager] = []
        for model_manager in all_model_managers:
            if model_manager is not None:
                _active_model_managers.append(model_manager)

        return _active_model_managers

    def __init__(
        self,
    ):
        """Create a new instance of model manager."""
        self.cuda_available = torch.cuda.is_available()
        """DEPRECATED: Use `torch.cuda.is_available()` instead."""

        # We use this to serialise disk reads as no point in
        # doing more than one at a time as this will slow down the sequential read op.
        self.disk_read_mutex = threading.Lock()

    def get_model_copy(self, model_name, model_component=None):
        if not model_component:
            return copy.deepcopy(self.loaded_models[model_name])
        else:
            return copy.deepcopy(self.loaded_models[model_name][model_component])

    def init_model_managers(
        self,
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
    ):  # XXX are we married to the name and/or the idea behind this function
        """For each arg which is true, attempt to load that `BaseModelManager` type."""
        args_passed: dict = locals().copy()  # XXX This is temporary
        args_passed.pop("self")  # XXX This is temporary

        allModelMangerTypeKeys = MODEL_MANAGERS_TYPE_LOOKUP.keys()
        # e.g. `MODEL_MANAGERS_TYPE_LOOKUP["compvis"]`` returns type `CompVisModelManager`

        for argName, argValue in args_passed.items():
            if not (argName in allModelMangerTypeKeys and hasattr(self, argName)):
                raise Exception(f"{argName} is not a valid model manager type!")
                # XXX better guarantees need to be made
            if not argValue:
                continue
            if getattr(self, argName) is not None:
                continue

            modelmanager = MODEL_MANAGERS_TYPE_LOOKUP[argName]

            # at runtime modelmanager() will be CompVisModelManager(), ClipModelManager(), etc
            setattr(
                self,
                argName,
                modelmanager(download_reference=False),
            )  # XXX # FIXME # HACK

    def reload_database(self) -> None:
        """Completely resets the `BaseModelManager` classes, and forces each to re-init."""
        model_managers: list[BaseModelManager] = []
        for model_manager_type in MODEL_MANAGERS_TYPE_LOOKUP:
            model_managers.append(getattr(self, model_manager_type))

        for model_manager in model_managers:
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
            model_manager: BaseModelManager = getattr(self, model_manager_type)
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
            model_manager: BaseModelManager = getattr(self, model_manager_type)
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
        for model_manager_type in MODEL_MANAGERS_TYPE_LOOKUP:
            model_manager: BaseModelManager = getattr(self, model_manager_type)
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
        for model_manager_type in MODEL_MANAGERS_TYPE_LOOKUP:
            model_manager: BaseModelManager = getattr(self, model_manager_type)
            if model_manager is None:
                continue
            if model_name in model_manager.model_reference or model_manager.is_local_model(model_name):
                return model_manager.unload_model(model_name)
        return None

    def move_from_disk_cache(self, model_name, model, clip, vae):
        """Moves the given model back into ram.

        Args:
            model_name (str): The model name to remove.
            model (model): the model data
            clip (clip): the clip model data
            var (var): the var model data
        """
        for model_manager_type in MODEL_MANAGERS_TYPE_LOOKUP:
            model_manager: BaseModelManager = getattr(self, model_manager_type)
            if model_manager is None:
                continue
            if model_name in model_manager.model_reference or model_manager.is_local_model(model_name):
                return model_manager.move_from_disk_cache(model_name, model, clip, vae)
        return None

    def get_loaded_models_names(
        self,
        mm_include: list[str | MODEL_CATEGORY_NAMES | type[BaseModelManager]] | None = None,
        mm_exclude: list[str | MODEL_CATEGORY_NAMES | type[BaseModelManager]] | None = None,
    ) -> list:
        """
        Returns:
            list: All currently loaded models.
        """
        loaded_models = self.get_loaded_models(mm_include, mm_exclude)
        return list(loaded_models.keys())

    def is_model_loaded(self, model_name) -> bool:
        return model_name in self.loaded_models

    def get_available_models_by_types(
        self,
        mm_include: list[str | MODEL_CATEGORY_NAMES | type[BaseModelManager]] | None = None,
        mm_exclude: list[str | MODEL_CATEGORY_NAMES | type[BaseModelManager]] | None = None,
    ):
        return self.get_available_models(mm_include, mm_exclude)

    def count_available_models_by_types(
        self,
        model_types: list[str] | None = None,
    ) -> int:
        return len(self.get_available_models_by_types(model_types))

    def ensure_memory_available(self, specific_type: type[BaseModelManager] | None = None) -> None:
        """Asserts minimum amount of RAM is available. Unloads models if necessary."""
        for model_manager_type in MODEL_MANAGERS_TYPE_LOOKUP:
            if specific_type and specific_type != model_manager_type:
                continue
            model_manager: BaseModelManager = getattr(self, model_manager_type)
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
        """_summary_

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
            logger.init_warn(
                f"{model_name} is excluded from loading at this time. If this is unexpected, let us know on discord.",
                status="Skipping",
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

    def get_mm_pointers(self, mm_types: list[str | MODEL_CATEGORY_NAMES | type[BaseModelManager]] | None = None):
        if not mm_types:
            return set()

        active_model_managers_types = [type(model_manager) for model_manager in self.active_model_managers]

        resolved_types = []
        for mm_type in mm_types:
            if mm_type in active_model_managers_types:
                resolved_types.append(mm_type)
                continue

            if mm_type in MODEL_MANAGERS_TYPE_LOOKUP:
                if MODEL_MANAGERS_TYPE_LOOKUP[mm_type] not in active_model_managers_types:
                    logger.warning(f"Attempted to reference a model manager which isn't loaded: '{mm_type}'.")
                else:
                    resolved_types.append(MODEL_MANAGERS_TYPE_LOOKUP[mm_type])

        return {mm for mm in self.active_model_managers if type(mm) in resolved_types}
