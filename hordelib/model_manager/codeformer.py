import os
import time

from loguru import logger


from hordelib.cache import get_cache_directory
from hordelib.model_manager.base import BaseModelManager
from hordelib import comfy_horde



class CodeFormerModelManager(BaseModelManager):
    def __init__(self, download_reference=True):
        super().__init__()
        self.download_reference = download_reference
        self.path = f"{get_cache_directory()}/codeformer"
        self.models_db_name = "codeformer"
        # self.gfpgan = GfpganModelManager()
        # self.esrgan = EsrganModelManager()
        self.models_path = self.pkg / f"{self.models_db_name}.json"
        self.remote_db = f"https://raw.githubusercontent.com/db0/AI-Horde-image-model-reference/main/{self.models_db_name}.json"
        self.init()

    def load(
        self,
        model_name: str,
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
        if model_name not in self.models:
            logger.error(f"{model_name} not found")
            return False
        if model_name not in self.available_models:
            logger.error(f"{model_name} not available")
            logger.info(
                f"Downloading {model_name}", status="Downloading"
            )  # logger.init_ok
            self.download_model(model_name)
            logger.info(
                f"{model_name} downloaded", status="Downloading"
            )  # logger.init_ok
        if model_name not in self.loaded_models:
            tic = time.time()
            logger.info(f"{model_name}", status="Loading")  # logger.init
            self.loaded_models[model_name] = self.load_codeformer(
                model_name,
                half_precision=half_precision,
                gpu_id=gpu_id,
                cpu_only=cpu_only,
            )
            logger.info(f"Loading {model_name}", status="Success")  # logger.init_ok
            toc = time.time()
            logger.info(
                f"Loading {model_name}: Took {toc-tic} seconds", status="Success"
            )  # logger.init_ok
            return True

    def load_codeformer(
        self,
        model_name,
    ):
        model_path = self.get_model_files(model_name)[0]["path"]
        model_path = f"{self.path}/{model_path}"
        sd = comfy_horde.load_torch_file(model_path)
        out = comfy_horde.model_loading.load_state_dict(sd).eval()
        return (out, )
