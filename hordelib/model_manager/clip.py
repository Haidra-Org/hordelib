import time
import typing
from pathlib import Path

import clip
import open_clip
import torch
from loguru import logger
from typing_extensions import override

from hordelib.config_path import get_hordelib_path
from hordelib.consts import MODEL_CATEGORY_NAMES, MODEL_DB_NAMES
from hordelib.model_manager.base import BaseModelManager


class ClipModelManager(BaseModelManager):
    def __init__(self, download_reference=False):
        super().__init__(
            models_db_name=MODEL_DB_NAMES[MODEL_CATEGORY_NAMES.clip],
            download_reference=download_reference,
        )

    def load_ranking_lists(self):
        ranking_lists = {}
        ranking_lists_path = Path(get_hordelib_path()).joinpath(
            "clip/",
            "ranking_lists/",
        )
        for file in ranking_lists_path.glob("*.txt"):
            ranking_lists[file.stem] = load_list(file)
        return ranking_lists

    def load_coca(self, model_name, half_precision=True, gpu_id=0, cpu_only=False):
        model_path = self.get_model_files(model_name)[0]["path"]
        model_path = f"{self.modelFolderPath}/{model_path}"
        if cpu_only:
            device = torch.device("cpu")
            half_precision = False
        else:
            device = torch.device(f"cuda:{gpu_id}" if self.cuda_available else "cpu")
        model, _, transform = open_clip.create_model_and_transforms(
            "coca_ViT-L-14",
            pretrained=model_path,
            device=device,
            precision="fp16" if half_precision else "fp32",
        )
        model = model.eval()
        model.to(device)
        if half_precision:
            model = model.half()
        return {
            "model": model,
            "device": device,
            "transform": transform,
            "half_precision": half_precision,
            "cache_name": model_name.replace("/", "_"),
        }

    def load_open_clip(self, model_name, half_precision=True, gpu_id=0, cpu_only=False):
        pretrained = self.get_model(model_name)["pretrained_name"]
        if cpu_only:
            device = torch.device("cpu")
            half_precision = False
        else:
            device = torch.device(f"cuda:{gpu_id}" if self.cuda_available else "cpu")
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            cache_dir=self.modelFolderPath,
            device=device,
            precision="fp16" if half_precision else "fp32",
        )
        model = model.eval()
        model.to(device)
        if half_precision:
            model = model.half()
        ranking_lists = self.load_ranking_lists()
        return {
            "model": model,
            "device": device,
            "preprocess": preprocess,
            "ranking_lists": ranking_lists,
            "half_precision": half_precision,
            "cache_name": model_name.replace("/", "_"),
        }

    def load_clip(self, model_name, half_precision=True, gpu_id=0, cpu_only=False):
        if cpu_only:
            device = torch.device("cpu")
            half_precision = False
        else:
            device = torch.device(f"cuda:{gpu_id}" if self.cuda_available else "cpu")
        model, preprocess = clip.load(
            model_name,
            device=device,
            download_root=self.modelFolderPath,
        )
        model = model.eval()
        if half_precision:
            model = model.half()
        ranking_lists = self.load_ranking_lists()
        return {
            "model": model,
            "device": device,
            "preprocess": preprocess,
            "ranking_lists": ranking_lists,
            "half_precision": half_precision,
            "cache_name": model_name.replace("/", "_"),
        }

    @override
    def modelToRam(
        self,
        model_name: str,
        half_precision=True,
        gpu_id=0,
        cpu_only=False,
        **kwargs,
    ) -> dict[str, typing.Any]:
        """
        model_name: str. Name of the model to load. See available_models for a list of available models.
        half_precision: bool. If True, the model will be loaded in half precision.
        gpu_id: int. The id of the gpu to use. If the gpu is not available, the model will be loaded on the cpu.
        cpu_only: bool. If True, the model will be loaded on the cpu. If True, half_precision will be set to False.
        """
        loaded_model_info = None
        if not self.cuda_available:
            cpu_only = True
        tic = time.time()
        logger.init(f"{model_name}", status="Loading")  # logger.init
        if self.model_reference[model_name]["type"] == "open_clip":
            loaded_model_info = self.load_open_clip(
                model_name,
                half_precision,
                gpu_id,
                cpu_only,
            )
            self.add_loaded_model(model_name, loaded_model_info)
        elif self.model_reference[model_name]["type"] == "clip":
            loaded_model_info = self.load_clip(
                model_name,
                half_precision,
                gpu_id,
                cpu_only,
            )
            self.add_loaded_model(model_name, loaded_model_info)
        elif self.model_reference[model_name]["type"] == "coca":
            loaded_model_info = self.load_coca(
                model_name,
                half_precision,
                gpu_id,
                cpu_only,
            )
            self.add_loaded_models(model_name, loaded_model_info)
        else:
            logger.error(
                f"Unknown model type: {self.model_reference[model_name]['type']}",
            )
            return {}  # XXX # FIXME
        if not loaded_model_info:
            logger.init_error(f"Failed to load {model_name}", status="Error")
            return {}  # XXX # FIXME

        logger.init_ok(f"Loading {model_name}", status="Success")  # logger.init_ok
        toc = time.time()
        logger.init_ok(
            f"Loading {model_name}: Took {toc-tic} seconds",
            status="Success",
        )  # logger.init_ok
        return loaded_model_info


def load_list(filename):
    with open(filename, encoding="utf-8", errors="replace") as f:
        return [line.strip() for line in f.readlines()]
