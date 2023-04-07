import os
import time

from loguru import logger

from hordelib.cache import get_cache_directory
from hordelib.comfy_horde import load_checkpoint_guess_config
from hordelib.consts import REMOTE_MODEL_DB
from hordelib.model_manager.base import BaseModelManager


class CompVisModelManager(BaseModelManager):
    def __init__(self, download_reference=True, custom_path="models/custom"):
        super().__init__()
        self.download_reference = download_reference
        self.path = f"{get_cache_directory()}/compvis"
        self.custom_path = custom_path
        self.models_db_name = "stable_diffusion"
        self.models_path = self.pkg / f"{self.models_db_name}.json"
        self.remote_db = f"{REMOTE_MODEL_DB}{self.models_db_name}.json"
        self.init()

    def load(
        self,
        model_name: str,
        output_vae=True,
        output_clip=True,
    ):
        """
        model_name: str. Name of the model to load. See available_models for a list of available models.
        """
        if model_name not in self.models:
            logger.error(f"{model_name} not found")
            return False
        if model_name not in self.available_models:
            logger.error(f"{model_name} not available")
            self.download_model(model_name)
            logger.info(f"{model_name}", status="Downloaded")  # logger.init_ok
        if model_name not in self.loaded_models:
            tic = time.time()
            logger.info(f"{model_name}", status="Loading")  # logger.init
            embeddings_path = os.getenv("HORDE_MODEL_DIR_EMBEDDINGS", "./")
            ckpt_path = self.get_model_files(model_name)[0]["path"]
            ckpt_path = f"{self.path}/{ckpt_path}"
            self.loaded_models[model_name] = load_checkpoint_guess_config(
                ckpt_path,
                output_vae=True,
                output_clip=True,
                embedding_directory=embeddings_path,
            )
            toc = time.time()
            logger.info(
                f"{model_name}: {round(toc-tic,2)} seconds",
                status="Loaded",
            )  # logger.init_ok
            return True
        return None
