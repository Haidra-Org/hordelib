import importlib.resources as importlib_resources
import time
import typing
from pathlib import Path

import torch
from loguru import logger
from typing_extensions import override

from hordelib.config_path import get_hordelib_path
from hordelib.consts import MODEL_CATEGORY_NAMES, MODEL_DB_NAMES
from hordelib.model_manager.base import BaseModelManager
from hordelib.utils.blip.blip import blip_decoder


class BlipModelManager(BaseModelManager):
    def __init__(self, download_reference=False):
        super().__init__(
            models_db_name=MODEL_DB_NAMES[MODEL_CATEGORY_NAMES.blip],
            download_reference=download_reference,
        )

    @override
    def modelToRam(
        self,
        model_name,
        half_precision=True,
        gpu_id=0,
        cpu_only=False,
        blip_image_eval_size=512,
        **kwargs,
    ) -> dict[str, typing.Any]:
        if not self.cuda_available:
            cpu_only = True
        vit = "base" if model_name == "BLIP" else "large"
        model_path = self.get_model_files(model_name)[0]["path"]
        model_path = f"{self.modelFolderPath}/{model_path}"
        if cpu_only:
            device = torch.device("cpu")
            half_precision = False
        else:
            device = torch.device(f"cuda:{gpu_id}" if self.cuda_available else "cpu")
        logger.info(f"Loading model {model_name} on {device}")
        logger.info(f"Model path: {model_path}")
        med_path = Path(get_hordelib_path()).joinpath(
            "model_database/",
            "med_config.json",
        )
        with importlib_resources.as_file(med_path.resolve()) as med_config:
            logger.info(f"Med config path: {med_config}")
            model = blip_decoder(
                pretrained=model_path,
                med_config=med_config,
                image_size=blip_image_eval_size,
                vit=vit,
            )
        model = model.eval()
        model.to(device)
        if half_precision:
            model = model.half()
        return {"model": model, "device": device, "half_precision": half_precision}
