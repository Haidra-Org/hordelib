import os
import typing

from typing_extensions import override

from hordelib.comfy_horde import horde_load_checkpoint
from hordelib.consts import (
    MODEL_CATEGORY_NAMES,
    MODEL_DB_NAMES,
    MODEL_FOLDER_NAMES,
)
from hordelib.model_manager.base import BaseModelManager


class CompVisModelManager(BaseModelManager):
    def __init__(
        self,
        download_reference=False,
        # custom_path="models/custom",  # XXX Remove this and any others like it?
    ):
        super().__init__(
            modelFolder=MODEL_FOLDER_NAMES[MODEL_CATEGORY_NAMES.compvis],
            models_db_name=MODEL_DB_NAMES[MODEL_CATEGORY_NAMES.compvis],
            download_reference=download_reference,
        )

    @override
    def modelToRam(
        self,
        model_name: str,
        **kwargs,
    ) -> dict[str, typing.Any]:

        embeddings_path = os.getenv("HORDE_MODEL_DIR_EMBEDDINGS", "./")

        return horde_load_checkpoint(
            ckpt_path=self.getFullModelPath(model_name),
            embeddings_path=embeddings_path,
        )
