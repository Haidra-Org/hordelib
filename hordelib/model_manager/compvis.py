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
