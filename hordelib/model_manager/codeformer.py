import typing

from typing_extensions import override

from hordelib import comfy_horde
from hordelib.consts import MODEL_CATEGORY_NAMES, MODEL_DB_NAMES
from hordelib.model_manager.base import BaseModelManager


class CodeFormerModelManager(BaseModelManager):
    def __init__(self, download_reference=False):
        super().__init__(
            model_category_name=MODEL_CATEGORY_NAMES.codeformer,
            download_reference=download_reference,
        )
