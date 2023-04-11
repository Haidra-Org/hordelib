import time

import clip
import open_clip
import torch
from loguru import logger

from hordelib.consts import MODEL_CATEGORY_NAMES, MODEL_DB_NAMES
from hordelib.model_manager.base import BaseModelManager


class ClipModelManager(BaseModelManager):
    def __init__(self, download_reference=False):
        super().__init__(
            models_db_name=MODEL_DB_NAMES[MODEL_CATEGORY_NAMES.clip],
            download_reference=download_reference,
        )

    def load_data_lists(self):
        data_lists = {}
        # data_lists["artist"] = load_list(self.pkg / "artists.txt")
        # data_lists["flavors"] = load_list(self.pkg / "flavors.txt")
        # data_lists["medium"] = load_list(self.pkg / "mediums.txt")
        # data_lists["movement"] = load_list(self.pkg / "movements.txt")
        # data_lists["trending"] = load_list(self.pkg / "sites.txt")
        # data_lists["techniques"] = load_list(self.pkg / "techniques.txt")
        # data_lists["tags"] = load_list(self.pkg / "tags.txt")
        return data_lists

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
        data_lists = self.load_data_lists()
        return {
            "model": model,
            "device": device,
            "preprocess": preprocess,
            "data_lists": data_lists,
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
        data_lists = self.load_data_lists()
        return {
            "model": model,
            "device": device,
            "preprocess": preprocess,
            "data_lists": data_lists,
            "half_precision": half_precision,
            "cache_name": model_name.replace("/", "_"),
        }

    def modelToRam(
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
        loaded_model_info = None
        if not self.cuda_available:
            cpu_only = True
        tic = time.time()
        logger.info(f"{model_name}", status="Loading")  # logger.init
        if self.model_reference[model_name]["type"] == "open_clip":
            loaded_model_info = self.load_open_clip(
                model_name,
                half_precision,
                gpu_id,
                cpu_only,
            )
            self.loaded_models[model_name] = loaded_model_info
        elif self.model_reference[model_name]["type"] == "clip":
            loaded_model_info = self.load_clip(
                model_name,
                half_precision,
                gpu_id,
                cpu_only,
            )
            self.loaded_models[model_name] = loaded_model_info
        elif self.model_reference[model_name]["type"] == "coca":
            loaded_model_info = self.load_coca(
                model_name,
                half_precision,
                gpu_id,
                cpu_only,
            )
            self.loaded_models[model_name] = loaded_model_info
        else:
            logger.error(
                f"Unknown model type: {self.model_reference[model_name]['type']}"
            )
            return None
        if not loaded_model_info:
            logger.init_error(f"Failed to load {model_name}", status="Error")
            return None

        logger.info(f"Loading {model_name}", status="Success")  # logger.init_ok
        toc = time.time()
        logger.info(
            f"Loading {model_name}: Took {toc-tic} seconds",
            status="Success",
        )  # logger.init_ok
        return loaded_model_info
