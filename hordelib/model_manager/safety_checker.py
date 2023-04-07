import time

import torch
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)
from loguru import logger

from hordelib.cache import get_cache_directory
from hordelib.consts import REMOTE_MODEL_DB
from hordelib.model_manager.base import BaseModelManager


class SafetyCheckerModelManager(BaseModelManager):
    def __init__(self, download_reference=True):
        super().__init__()
        self.download_reference = download_reference
        self.path = f"{get_cache_directory()}/safety_checker"
        self.models_db_name = "safety_checker"
        self.models_path = self.pkg / f"{self.models_db_name}.json"
        self.remote_db = f"{REMOTE_MODEL_DB}{self.models_db_name}.json"
        self.init()

    def load(
        self,
        model_name,
        half_precision=True,
        gpu_id=0,
        cpu_only=False,
    ):
        """
        model_name: str. Name of the model to load. See available_models for a list of available models.
        half_precision: bool. If True, the model will be loaded in half precision.
        gpu_id: int. The id of the gpu to use. If the gpu is not available, the model will be loaded on the cpu.
        cpu_only: bool. If True, the model will be loaded on the cpu. If True, half_precision will be set to False.
        """
        if not self.cuda_available:
            cpu_only = True
        if model_name not in self.models:
            logger.error(f"{model_name} not found")
            return False
        if model_name not in self.available_models:
            logger.error(f"{model_name} not available")
            logger.info(
                f"Downloading {model_name}",
                status="Downloading",
            )  # logger.init_ok
            self.download_model(model_name)
            logger.info(
                f"{model_name} downloaded",
                status="Downloading",
            )  # logger.init_ok
        if model_name not in self.loaded_models:
            tic = time.time()
            logger.info(f"{model_name}", status="Loading")  # logger.init
            self.loaded_models[model_name] = self.load_safety_checker(
                model_name,
                half_precision=half_precision,
                gpu_id=gpu_id,
                cpu_only=cpu_only,
            )
            logger.info(f"Loading {model_name}", status="Success")  # logger.init_ok
            toc = time.time()
            logger.info(
                f"Loading {model_name}: Took {toc-tic} seconds",
                status="Success",
            )  # logger.init_ok
            return True
        return None

    def load_safety_checker(
        self,
        model_name,
        half_precision=True,
        gpu_id=0,
        cpu_only=True,  # True by default for easy Horde compatibility
    ):
        if not self.cuda_available:
            cpu_only = True
        if cpu_only:
            device = torch.device("cpu")
            half_precision = False
        else:
            device = torch.device(f"cuda:{gpu_id}" if self.cuda_available else "cpu")
        logger.info(f"Loading model {model_name} on {device}")
        logger.info(f"Model path: {self.path}")
        model = StableDiffusionSafetyChecker.from_pretrained(self.path)
        model = model.eval()
        model.to(device)
        if half_precision:
            model = model.half()
        return {"model": model, "device": device, "half_precision": half_precision}
