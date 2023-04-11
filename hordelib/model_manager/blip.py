import importlib.resources as importlib_resources
import time

import torch
from loguru import logger

from hordelib.consts import MODEL_CATEGORY_NAMES, MODEL_DB_NAMES
from hordelib.model_manager.base import BaseModelManager


class BlipModelManager(BaseModelManager):
    def __init__(self, download_reference=False):
        super().__init__(
            models_db_name=MODEL_DB_NAMES[MODEL_CATEGORY_NAMES.blip],
            download_reference=download_reference,
        )

    def load(
        self,
        model_name: str,
        half_precision=True,
        gpu_id=0,
        cpu_only=False,
        blip_image_eval_size=512,
    ):
        """
        model_name: str. Name of the model to load. See available_models for a list of available models.
        half_precision: bool. If True, the model will be loaded in half precision.
        gpu_id: int. The id of the gpu to use. If the gpu is not available, the model will be loaded on the cpu.
        cpu_only: bool. If True, the model will be loaded on the cpu. If True, half_precision will be set to False.
        blip_image_eval_size: int. The size of the image to use for the blip model.
        """
        if model_name not in self.model_reference:
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
            self.loaded_models[model_name] = self.load_blip(  # XXX # FIXME
                model_name,
                half_precision=half_precision,
                gpu_id=gpu_id,
                cpu_only=cpu_only,
                blip_image_eval_size=blip_image_eval_size,
            )
            logger.info(f"Loading {model_name}", status="Success")  # logger.init_ok
            toc = time.time()
            logger.info(
                f"Loading {model_name}: Took {toc-tic} seconds",
                status="Success",
            )  # logger.init_ok
            return True
        return None

    def modelToRam(
        self,
        model_name,
        half_precision=True,
        gpu_id=0,
        cpu_only=False,
        blip_image_eval_size=512,
    ):
        raise NotImplementedError("BLIP is not currently implemented!")
        # if not self.cuda_available:
        #     cpu_only = True
        # vit = "base" if model_name == "BLIP" else "large"
        # model_path = self.get_model_files(model_name)[0]["path"]
        # model_path = f"{self.modelFolderPath}/{model_path}"
        # if cpu_only:
        #     device = torch.device("cpu")
        #     half_precision = False
        # else:
        #     device = torch.device(f"cuda:{gpu_id}" if self.cuda_available else "cpu")
        # logger.info(f"Loading model {model_name} on {device}")
        # logger.info(f"Model path: {model_path}")
        # with importlib_resources.as_file(self.pkg / "med_config.json") as med_config:
        #     logger.info(f"Med config path: {med_config}")
        #     model = blip_decoder(
        #         # XXX # FIXME
        #         pretrained=model_path,
        #         med_config=med_config,
        #         image_size=blip_image_eval_size,
        #         vit=vit,
        #     )
        # model = model.eval()
        # model.to(device)
        # if half_precision:
        #     model = model.half()
        # return {"model": model, "device": device, "half_precision": half_precision}
