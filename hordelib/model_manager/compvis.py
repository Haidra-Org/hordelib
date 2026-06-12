import json
import os
from pathlib import Path
from typing import Any, override

from horde_model_reference.meta_consts import (
    MODEL_DOMAIN,
    MODEL_PURPOSE,
    MODEL_REFERENCE_CATEGORY,
    ModelClassification,
)
from horde_model_reference.model_reference_records import (
    DownloadRecord,
    GenericModelRecordConfig,
    ImageGenerationModelRecord,
)
from loguru import logger

from hordelib.consts import MODEL_CATEGORY_NAMES
from hordelib.model_manager.base import BaseModelManager


def _custom_model_entry_to_record(model_name: str, entry: dict[str, Any]) -> ImageGenerationModelRecord:
    """Convert a HORDELIB_CUSTOM_MODELS JSON entry into a pydantic record.

    Custom model entries use the legacy flat-dict shape::

        {"name": ..., "baseline": ..., "type": "ckpt", "config": {"files": [{"path": "<abs path>"}]}}

    Normalising them here means the rest of hordelib only ever sees pydantic records.
    """
    download_entries: list[DownloadRecord] = []
    for file_entry in entry.get("config", {}).get("files", []):
        file_path = file_entry.get("path")
        if not file_path:
            continue
        download_entries.append(
            DownloadRecord(
                file_name=file_path,
                file_url=file_entry.get("url", ""),
                sha256sum=file_entry.get("sha256sum", "FIXME"),
            ),
        )

    if not download_entries:
        raise ValueError(f"Custom model {model_name} has no usable file entries")

    return ImageGenerationModelRecord(
        record_type=MODEL_REFERENCE_CATEGORY.image_generation,
        name=entry.get("name", model_name),
        description=entry.get("description", "Custom model (HORDELIB_CUSTOM_MODELS)"),
        baseline=entry.get("baseline", "stable_diffusion_1"),
        nsfw=bool(entry.get("nsfw", False)),
        inpainting=bool(entry.get("inpainting", False)),
        model_classification=ModelClassification(domain=MODEL_DOMAIN.image, purpose=MODEL_PURPOSE.generation),
        config=GenericModelRecordConfig(download=download_entries),
    )


class CompVisModelManager(BaseModelManager[ImageGenerationModelRecord]):
    def __init__(
        self,
        download_reference=False,
        **kwargs,
    ):
        kwargs.pop("model_category_name", None)  # consumed by this subclass
        super().__init__(
            model_category_name=MODEL_CATEGORY_NAMES.compvis,
            download_reference=download_reference,
            **kwargs,
        )

    @override
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
                        # Avoid clobbering
                        if mname in self.model_reference:
                            continue
                        try:
                            self.model_reference[mname] = _custom_model_entry_to_record(mname, extra_models[mname])
                        except Exception as e:
                            logger.error("Skipping invalid custom model: model={}, error={}", mname, e)
                            continue
                        if self.is_model_available(mname):
                            self.available_models.append(mname)
                        num_custom_models += 1

        except json.decoder.JSONDecodeError as e:
            logger.error("Custom model database is not valid JSON: path={}, error={}", self.models_db_path, e)
            raise

        logger.info("Loaded custom models: count={}, path={}", num_custom_models, os.getenv("HORDELIB_CUSTOM_MODELS"))
