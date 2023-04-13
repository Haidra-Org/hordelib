import typing

import torch
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)
from loguru import logger
from typing_extensions import override

from hordelib.consts import MODEL_CATEGORY_NAMES, MODEL_DB_NAMES
from hordelib.model_manager.base import BaseModelManager


class SafetyCheckerModelManager(BaseModelManager):
    def __init__(self, download_reference=False):
        super().__init__(
            models_db_name=MODEL_DB_NAMES[MODEL_CATEGORY_NAMES.safety_checker],
            download_reference=download_reference,
        )

    @override
    def modelToRam(
        self,
        model_name: str,
        half_precision=True,
        gpu_id=0,
        cpu_only=True,
        **kwargs,
    ) -> dict[str, typing.Any]:
        if not self.cuda_available:
            cpu_only = True
        if cpu_only:
            device = torch.device("cpu")
            half_precision = False
        else:
            device = torch.device(f"cuda:{gpu_id}" if self.cuda_available else "cpu")
        logger.info(f"Loading model {model_name} on {device}")
        logger.info(f"Model path: {self.modelFolderPath}")
        model = StableDiffusionSafetyChecker.from_pretrained(self.modelFolderPath)
        model = model.eval()
        model.to(device)
        if half_precision:
            model = model.half()
        return {"model": model, "device": device, "half_precision": half_precision}
