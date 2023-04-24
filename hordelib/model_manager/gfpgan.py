import os
import typing

from typing_extensions import override

from hordelib import comfy_horde
from hordelib.consts import MODEL_CATEGORY_NAMES, MODEL_DB_NAMES
from hordelib.model_manager.base import BaseModelManager


class GfpganModelManager(BaseModelManager):
    def __init__(self, download_reference=False):
        super().__init__(
            models_db_name=MODEL_DB_NAMES[MODEL_CATEGORY_NAMES.gfpgan],
            download_reference=download_reference,
        )

    @override
    def is_local_model(self, model_name):
        parts = os.path.splitext(model_name.lower())
        if parts[-1] in [".pth"]:
            return True
        return False

    @override
    def modelToRam(
        self,
        model_name: str,
        **kwargs,
    ) -> dict[str, typing.Any]:
        model_path = self.getFullModelPath(model_name)
        sd = comfy_horde.load_torch_file(model_path)
        out = comfy_horde.load_state_dict(sd).eval()
        return {"model": out}
