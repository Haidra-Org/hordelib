import copy
import os
import pickle
import typing

from loguru import logger
from typing_extensions import override

from hordelib import UserSettings
from hordelib.comfy_horde import horde_load_checkpoint
from hordelib.consts import MODEL_CATEGORY_NAMES
from hordelib.model_manager.base import BaseModelManager


class CompVisModelManager(BaseModelManager):
    def __init__(
        self,
        download_reference=False,
        # custom_path="models/custom",  # XXX Remove this and any others like it?
    ):
        super().__init__(
            model_category_name=MODEL_CATEGORY_NAMES.compvis,
            download_reference=download_reference,
        )

    @override
    def is_local_model(self, model_name):
        parts = os.path.splitext(model_name.lower())
        if parts[-1] in [".safetensors", ".ckpt"]:
            return True
        return False

    @override
    def modelToRam(
        self,
        model_name: str,
        **kwargs,
    ) -> dict[str, typing.Any]:
        return {}
        embeddings_path = os.path.join(UserSettings.get_model_directory(), "ti")
        if not embeddings_path:
            logger.debug("No embeddings path found, disabling embeddings")

        if not kwargs.get("local", False):
            ckpt_path = self.getFullModelPath(model_name)
        else:
            ckpt_path = os.path.join(self.modelFolderPath, model_name)
        return horde_load_checkpoint(
            ckpt_path=ckpt_path,
            embeddings_path=embeddings_path if embeddings_path else None,
        )

    def can_move_to_vram(self):
        # Allow moving directly to vram to save ram
        return True
