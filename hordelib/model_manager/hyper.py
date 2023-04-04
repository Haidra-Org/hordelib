from enum import Enum

import torch
from loguru import logger

from hordelib.model_manager.aitemplate import AITemplateModelManager
from hordelib.model_manager.base import BaseModelManager
from hordelib.model_manager.blip import BlipModelManager
from hordelib.model_manager.clip import ClipModelManager
from hordelib.model_manager.codeformer import CodeFormerModelManager
from hordelib.model_manager.compvis import CompVisModelManager
from hordelib.model_manager.controlnet import ControlNetModelManager
from hordelib.model_manager.diffusers import DiffusersModelManager

# from hordelib.model_manager.esrgan import EsrganModelManager
# from hordelib.model_manager.gfpgan import GfpganModelManager
from hordelib.model_manager.safety_checker import SafetyCheckerModelManager

# from worker.util.voodoo import initialise_voodoo


class EsrganModelManager:  # XXX # FIXME
    pass


class GfpganModelManager:  # XXX # FIXME
    pass


MODEL_MANAGERS_TYPE_LOOKUP = {
    "aitemplate": AITemplateModelManager,
    "blip": BlipModelManager,
    "clip": ClipModelManager,
    "codeformer": CodeFormerModelManager,
    "compvis": CompVisModelManager,
    "controlnet": ControlNetModelManager,
    "diffusers": DiffusersModelManager,
    # "esrgan": EsrganModelManager,
    # "gfpgan": GfpganModelManager,
    "safety_checker": SafetyCheckerModelManager,
}


class ModelManager:
    """
    Contains links to all the other MM classes
    """

    aitemplate: AITemplateModelManager | None = None
    blip: BlipModelManager | None = None
    clip: ClipModelManager | None = None
    codeformer: CodeFormerModelManager | None = None
    compvis: CompVisModelManager | None = None
    controlnet: ControlNetModelManager | None = None
    diffusers: DiffusersModelManager | None = None
    esrgan: EsrganModelManager | None = None
    gfpgan: GfpganModelManager | None = None
    safety_checker: SafetyCheckerModelManager | None = None

    def __init__(
        self,
    ):
        # logger.initialise_voodoo()
        self.cuda_available = torch.cuda.is_available()
        self.models = {}
        self.available_models = []
        self.loaded_models = {}

    def init_model_managers(  # XXX
        self,
        aitemplate: bool = False,
        blip: bool = False,
        clip: bool = False,
        codeformer: bool = False,
        compvis: bool = False,
        controlnet: bool = False,
        diffusers: bool = False,
        # esrgan: bool = False,
        # gfpgan: bool = False,
        safety_checker: bool = False,
    ):
        args_passed: dict = locals().copy()
        args_passed.pop("self")

        allModelMangerTypeKeys = MODEL_MANAGERS_TYPE_LOOKUP.keys()
        # e.g. `MODEL_MANAGERS_TYPE_LOOKUP["compvis"]`` returns type `CompVisModelManager`

        for argName, argValue in args_passed.items():
            if not (argName in allModelMangerTypeKeys and hasattr(self, argName)):
                raise Exception()  # XXX
            if not argValue:
                continue

            modelmanager = MODEL_MANAGERS_TYPE_LOOKUP[argName]
            # at runtime modelmanager() will be CompVisModelManager(), ClipModelManager(), etc

            setattr(self, argName, modelmanager())

        self.refreshManagers()

    def refreshManagers(self):
        """
        Initialize SuperModelManager's models and available_models from
        the models and available_models of the model types.
        Individual model types are already initialized in their own init() functions
        which are called when the individual model manager is created in __init__.
        """
        model_types = [
            self.aitemplate,
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
        self.available_models = []
        for model_type in model_types:
            if model_type is not None:
                self.models.update(model_type.models)
                self.available_models.extend(model_type.available_models)

    def reload_database(self):
        """
        Horde-specific function to reload the database of available models.
        Note: It is not appropriate to place `model_type.init()` in `init()`
        because individual model types are already initialized after being created
        i.e. if `model_type.init()` is placed in `self.init()`, the database will be
        loaded twice. # XXX rewrite
        """
        model_managers = []
        for model_manager_type in MODEL_MANAGERS_TYPE_LOOKUP.keys():
            model_managers.append(getattr(self, model_manager_type))

        self.available_models = []  # reset available models
        for model_manager in model_managers:
            if model_manager is not None:
                model_manager.init()
                self.models.update(model_manager.models)
                self.available_models.extend(model_manager.available_models)

    def download_model(self, model_name) -> bool | None:
        for model_manager_type in MODEL_MANAGERS_TYPE_LOOKUP.keys():
            model_manager: BaseModelManager = getattr(self, model_manager_type)
            if model_manager is None:
                continue
            if model_name not in model_manager.models:
                continue

            return model_manager.download_model(model_name)
        return None  # XXX

    def download_all(self):
        for model_manager_type in MODEL_MANAGERS_TYPE_LOOKUP.keys():
            model_manager: BaseModelManager = getattr(self, model_manager_type)
            if model_manager is None:
                continue
            if isinstance(model_manager, AITemplateModelManager):
                model_manager.download_ait("cuda")  # XXX
                continue

            model_manager.download_all_models()

    def validate_model(self, model_name, skip_checksum=False):
        for model_manager_type in MODEL_MANAGERS_TYPE_LOOKUP.keys():
            model_manager: BaseModelManager = getattr(self, model_manager_type)
            if model_manager is None:
                continue
            if model_manager_type == "aitemplate":  # XXX
                continue
            if model_name in model_manager.models:
                model_manager.validate_model(model_name, skip_checksum)

    def taint_models(self, models):
        for model_manager_type in MODEL_MANAGERS_TYPE_LOOKUP.keys():
            model_manager: BaseModelManager = getattr(self, model_manager_type)
            if model_manager is None:
                continue
            if any(model in model_manager.models for model in models):
                model_manager.taint_models(models)

    def unload_model(self, model_name) -> bool:
        for model_manager_type in MODEL_MANAGERS_TYPE_LOOKUP.keys():
            model_manager: BaseModelManager = getattr(self, model_manager_type)
            if model_manager is None:
                continue
            if model_name not in model_manager.models:
                continue
            model_unloaded = model_manager.unload_model(model_name)
            return model_unloaded
        return False

    def get_loaded_models_names(self, string=False):
        """
        :param string: If True, returns concatenated string of model names
        Returns a list of the loaded model names
        """
        # return ["Deliberate"] # Debug
        if string:
            return ", ".join(self.loaded_models.keys())
        return list(self.loaded_models.keys())

    def get_available_models_by_types(self, model_types=None):
        if not model_types:
            model_types = ["ckpt", "diffusers"]
        models_available = []
        for model_type in model_types:
            if model_type == "ckpt":
                if self.compvis is not None:
                    for model in self.compvis.models:
                        # We don't want to check the .yaml file as those exist in this repo instead
                        model_files = [
                            filename
                            for filename in self.compvis.get_model_files(model)
                            if not filename["path"].endswith(".yaml")
                        ]
                        if self.compvis.check_available(model_files):
                            models_available.append(model)
            if model_type == "diffusers":
                if self.diffusers is not None:
                    for model in self.diffusers.models:
                        if self.diffusers.check_available(
                            self.diffusers.get_model_files(model)
                        ):
                            models_available.append(model)
        return models_available

    def count_available_models_by_types(self, model_types=None):
        return len(self.get_available_models_by_types(model_types))

    def get_available_models(self):
        """
        Returns the available models
        """
        return self.available_models

    def load(
        self,
        model_name,
        half_precision=True,
        gpu_id=0,
        cpu_only=False,
        voodoo=False,
    ):  # XXX # TODO
        """
        model_name: str. Name of the model to load. See available_models for a list of available models.
        half_precision: bool. If True, the model will be loaded in half precision.
        gpu_id: int. The id of the gpu to use. If the gpu is not available, the model will be loaded on the cpu.
        cpu_only: bool. If True, the model will be loaded on the cpu. If True, half_precision will be set to False.
        voodoo: bool. (compvis only) Voodoo ray.
        """
        if not self.cuda_available:
            cpu_only = True
        if self.aitemplate is not None and model_name in self.aitemplate.models:
            return self.aitemplate.load(model_name, gpu_id)
        if self.blip is not None and model_name in self.blip.models:
            success = self.blip.load(
                model_name=model_name,
                half_precision=half_precision,
                gpu_id=gpu_id,
                cpu_only=cpu_only,
            )
            if success:
                self.loaded_models.update(
                    {model_name: self.blip.loaded_models[model_name]}
                )
            return success
        if self.clip is not None and model_name in self.clip.models:
            success = self.clip.load(
                model_name=model_name,
                half_precision=half_precision,
                gpu_id=gpu_id,
                cpu_only=cpu_only,
            )
            if success:
                self.loaded_models.update(
                    {model_name: self.clip.loaded_models[model_name]}
                )
            return success
        if self.codeformer is not None and model_name in self.codeformer.models:
            success = self.codeformer.load(
                model_name=model_name,
                half_precision=half_precision,
                gpu_id=gpu_id,
                cpu_only=cpu_only,
            )
            if success:
                self.loaded_models.update(
                    {model_name: self.codeformer.loaded_models[model_name]}
                )
            return success
        if self.compvis is not None and model_name in self.compvis.models:
            success = self.compvis.load(
                model_name=model_name,
                output_vae=True,
                output_clip=True,
            )
            if success:
                self.loaded_models.update(
                    {model_name: self.compvis.loaded_models[model_name]}
                )
            return success
        if self.diffusers is not None and model_name in self.diffusers.models:
            success = self.diffusers.load(
                model_name=model_name,
                half_precision=half_precision,
                gpu_id=gpu_id,
                cpu_only=cpu_only,
                voodoo=voodoo,
            )
            if success:
                self.loaded_models.update(
                    {model_name: self.diffusers.loaded_models[model_name]}
                )
            return success
        if self.esrgan is not None and model_name in self.esrgan.models:
            success = self.esrgan.load(
                model_name=model_name,
                half_precision=half_precision,
                gpu_id=gpu_id,
                cpu_only=cpu_only,
            )
            if success:
                self.loaded_models.update(
                    {model_name: self.esrgan.loaded_models[model_name]}
                )
            return success
        if self.gfpgan is not None and model_name in self.gfpgan.models:
            success = self.gfpgan.load(
                model_name=model_name, gpu_id=gpu_id, cpu_only=cpu_only
            )
            if success:
                self.loaded_models.update(
                    {model_name: self.gfpgan.loaded_models[model_name]}
                )
            return success
        if self.safety_checker is not None and model_name in self.safety_checker.models:
            success = self.safety_checker.load(
                model_name=model_name,
                half_precision=half_precision,
                gpu_id=gpu_id,
                cpu_only=True,  # for the horde
            )
            if success:
                self.loaded_models.update(
                    {model_name: self.safety_checker.loaded_models[model_name]}
                )
            return success
        logger.error(f"{model_name} not found")
        return False
