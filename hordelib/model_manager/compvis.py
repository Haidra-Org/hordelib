import json
import os
from pathlib import Path

from loguru import logger

from hordelib.consts import MODEL_CATEGORY_NAMES
from hordelib.model_manager.base import BaseModelManager


class CompVisModelManager(BaseModelManager):
    def __init__(
        self,
        download_reference=False,
        **kwargs,
        # custom_path="models/custom",  # XXX Remove this and any others like it?
    ):
        super().__init__(
            model_category_name=MODEL_CATEGORY_NAMES.compvis,
            download_reference=download_reference,
            **kwargs,
        )

    def load_model_database(self) -> None:
        super().load_model_database()

        num_custom_models = 0

        try:
            extra_models_path_str = os.getenv("HORDELIB_CUSTOM_MODELS")
            if extra_models_path_str:
                extra_models_path = Path(extra_models_path_str)
                if extra_models_path.exists():
                    extra_models = json.loads((extra_models_path).read_text())
                    for mname in extra_models:
                        # Avoid cloberring
                        if mname in self.model_reference:
                            continue
                        # Merge all custom models into our new model reference
                        self.model_reference[mname] = extra_models[mname]
                        if self.is_model_available(mname):
                            self.available_models.append(mname)

                    num_custom_models += len(extra_models)

        except json.decoder.JSONDecodeError as e:
            logger.error(f"Custom model database {self.models_db_path} is not valid JSON: {e}")
            raise

        logger.info(f"Loaded {num_custom_models} models from {self.models_db_path}")
