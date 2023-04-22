"""Home for the controller class ModelManager, and related meta information."""
import copy

import torch
from loguru import logger

from hordelib.consts import MODEL_CATEGORY_NAMES

# from hordelib.model_manager.aitemplate import AITemplateModelManager
from hordelib.model_manager.base import BaseModelManager
from hordelib.model_manager.blip import BlipModelManager
from hordelib.model_manager.clip import ClipModelManager
from hordelib.model_manager.codeformer import CodeFormerModelManager
from hordelib.model_manager.compvis import CompVisModelManager
from hordelib.model_manager.controlnet import ControlNetModelManager
from hordelib.model_manager.diffusers import DiffusersModelManager
from hordelib.model_manager.esrgan import EsrganModelManager
from hordelib.model_manager.gfpgan import GfpganModelManager
from hordelib.model_manager.safety_checker import SafetyCheckerModelManager

# from worker.util.voodoo import initialise_voodoo


MODEL_MANAGERS_TYPE_LOOKUP: dict[MODEL_CATEGORY_NAMES, type[BaseModelManager]] = {
    # ModelCategoryNames.aitemplate: AITemplateModelManager,
    MODEL_CATEGORY_NAMES.blip: BlipModelManager,
    MODEL_CATEGORY_NAMES.clip: ClipModelManager,
    MODEL_CATEGORY_NAMES.codeformer: CodeFormerModelManager,
    MODEL_CATEGORY_NAMES.compvis: CompVisModelManager,
    MODEL_CATEGORY_NAMES.controlnet: ControlNetModelManager,
    MODEL_CATEGORY_NAMES.diffusers: DiffusersModelManager,
    MODEL_CATEGORY_NAMES.esrgan: EsrganModelManager,
    MODEL_CATEGORY_NAMES.gfpgan: GfpganModelManager,
    MODEL_CATEGORY_NAMES.safety_checker: SafetyCheckerModelManager,
}
"""Keys are `str` which represent attrs in `ModelManger`. Values are the corresponding `type`."""


class ModelManager:
    """Controller class for all managers which extend `BaseModelManager`."""

    # aitemplate: AITemplateModelManager | None = None
    blip: BlipModelManager | None = None
    clip: ClipModelManager | None = None
    codeformer: CodeFormerModelManager | None = None
    compvis: CompVisModelManager | None = None
    controlnet: ControlNetModelManager | None = None
    diffusers: DiffusersModelManager | None = None
    esrgan: EsrganModelManager | None = None
    gfpgan: GfpganModelManager | None = None
    safety_checker: SafetyCheckerModelManager | None = None
    # XXX I think this can be reworked into an array of BaseModelManager instances

    @property
    def models(self) -> dict:
        """All model manager's internal dictionaries of models, loaded from model database JSON files."""
        _models: dict = {}
        for model_manager in self.active_model_managers:
            model_manager.model_reference
            _models.update(model_manager.model_reference)
        return _models

    @property
    def available_models(self) -> list[str]:
        """All models for which information exists, and for which a download attempt could be made."""
        all_available_models: list[str] = []
        for model_manager in self.active_model_managers:
            all_available_models.extend(model_manager.available_models)
        return all_available_models

    @property
    def loaded_models(self) -> dict[str, dict]:
        """All models for which have successfully loaded across all `BaseModelManager` types."""
        all_loaded_models: dict[str, dict] = {}
        for model_manager in self.active_model_managers:
            all_loaded_models.update(model_manager.get_loaded_models())
        return all_loaded_models

    @property
    def active_model_managers(self) -> list[BaseModelManager]:
        """All loaded model managers."""
        all_model_managers = [
            # self.aitemplate, # XXX TODO
            self.blip,
            self.clip,
            self.compvis,
            self.diffusers,
            self.esrgan,
            self.gfpgan,
            self.safety_checker,
            self.codeformer,
            self.controlnet,
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
        # logger.initialise_voodoo()
        self.cuda_available = torch.cuda.is_available()
        """DEPRECATED: Use `torch.cuda.is_available()` instead."""

    def get_model_copy(self, model_name, model_component=None):
        if not model_component:
            return copy.deepcopy(self.loaded_models[model_name])
        else:
            return copy.deepcopy(self.loaded_models[model_name][model_component])

    def init_model_managers(
        self,
        # aitemplate: bool = False, # XXX
        blip: bool = False,
        clip: bool = False,
        codeformer: bool = False,
        compvis: bool = False,
        controlnet: bool = False,
        diffusers: bool = False,
        esrgan: bool = False,
        gfpgan: bool = False,
        safety_checker: bool = False,
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
            # if isinstance(model_manager, AITemplateModelManager):
            #    model_manager.download_ait("cuda")
            #    # XXX this special handling predates me (@tazlin)
            #    continue

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
            if model_manager_type == "aitemplate":
                continue
                # XXX the special handling here predates me (@tazlin), unknown if needed
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
            if model_name not in model_manager.model_reference:
                continue
            return model_manager.unload_model(model_name)
        return None

    def get_loaded_models_names(self) -> list:
        """DEPRECATED: Use property self.loaded_models. Returns a list of all the currently loaded models.

        Returns:
            list: All currently loaded models.
        """
        return list(self.loaded_models.keys())

    def is_model_loaded(self, model_name) -> bool:
        return model_name in self.loaded_models

    def get_available_models_by_types(self, model_types: list[str] | None = None):
        if not model_types:
            model_types = ["ckpt", "diffusers"]
        models_available = []
        for model_type in model_types:
            if model_type == "ckpt" and self.compvis is not None:
                for model in self.compvis.model_reference:
                    # We don't want to check the .yaml file as those exist in this repo instead
                    model_files = [
                        filename
                        for filename in self.compvis.get_model_files(model)
                        if not filename["path"].endswith(".yaml")
                    ]
                    if self.compvis.check_available(model_files):
                        models_available.append(model)
            if model_type == "diffusers" and self.diffusers is not None:
                for model in self.diffusers.model_reference:
                    if self.diffusers.check_available(
                        self.diffusers.get_model_files(model),
                    ):
                        models_available.append(model)
        return models_available

    def count_available_models_by_types(
        self,
        model_types: list[str] | None = None,
    ) -> int:
        return len(self.get_available_models_by_types(model_types))

    def get_available_models(self) -> list:
        """Returns a list of all available models.

        Returns:
            list: All available models.
        """
        return self.available_models

    def ensure_memory_available(self, specific_type: type[BaseModelManager] | None = None) -> None:
        """Asserts minimum amount of RAM is available. Unloads models if necessary."""
        for model_manager_type in MODEL_MANAGERS_TYPE_LOOKUP:
            if specific_type and specific_type != model_manager_type:
                continue
            model_manager: BaseModelManager = getattr(self, model_manager_type)
            if model_manager is None:
                continue
            model_manager.ensure_memory_available()
        return

    def load(
        self,
        model_name: str,
        half_precision: bool = True,
        gpu_id: int = 0,
        cpu_only: bool = False,
        voodoo: bool = False,
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
            voodoo (bool, optional): (compvis only) Voodoo ray. Defaults to False.

        Returns:
            bool | None: The success of the load. If `None`, the model was not found.
        """
        if not self.cuda_available:
            cpu_only = True

        for model_manager in self.active_model_managers:
            if model_name in model_manager.model_reference:
                return model_manager.load(
                    model_name=model_name,
                    half_precision=half_precision,
                    gpu_id=gpu_id,
                    cpu_only=cpu_only,
                    voodoo=voodoo,
                )

        logger.error(f"{model_name} not found")
        return None
