import os
import time

from loguru import logger

from hordelib import comfy_horde
from hordelib.cache import get_cache_directory
from hordelib.consts import REMOTE_MODEL_DB
from hordelib.model_manager.base import BaseModelManager


class GfpganModelManager(BaseModelManager):
    def __init__(self, download_reference=True):
        super().__init__()
        self.download_reference = download_reference
        self.path = f"{get_cache_directory()}/gfpgan"
        self.models_db_name = "gfpgan"
        self.models_path = self.pkg / f"{self.models_db_name}.json"
        self.remote_db = f"{REMOTE_MODEL_DB}{self.models_db_name}.json"
        self.init()

    def load(
        self,
        model_name: str,
    ):
        """
        model_name: str. Name of the model to load. See available_models for a list of available models.
        """
        # if not self.cuda_available:
        #     cpu_only = True
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
            self.loaded_models[model_name] = self.load_gfpgan(
                model_name,
            )
            logger.info(f"Loading {model_name}", status="Success")
            toc = time.time()
            logger.info(
                f"Loading {model_name}: Took {toc-tic} seconds",
                status="Success",
            )  # logger.init_ok
            return True
        return None

    def load_gfpgan(
        self,
        model_name,
    ):
        model_path = self.get_model_files(model_name)[0]["path"]
        model_path = f"{self.path}/{model_path}"
        sd = comfy_horde.load_torch_file(model_path)
        out = comfy_horde.model_loading.load_state_dict(sd).eval()
        return (out,)
