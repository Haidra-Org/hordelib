import glob
from pathlib import Path

from loguru import logger

from hordelib.comfy_horde import ANNOTATOR_MODEL_SHA_LOOKUP, download_all_controlnet_annotators
from hordelib.model_manager.base import BaseModelManager


def validate_all_controlnet_annotators(annotatorPath: Path) -> bool:
    annotators = glob.glob("*.pt*", root_dir=annotatorPath)

    for annotator in annotators:
        annotator_full_path = annotatorPath.joinpath(annotator)

        if annotator not in ANNOTATOR_MODEL_SHA_LOOKUP:
            logger.warning(
                f"Annotator file {annotator} is not in the model database. Ignoring...",
            )
            logger.warning(f"File location: {annotator_full_path}")
            continue

        hash = BaseModelManager.get_file_sha256_hash(annotator_full_path)
        if hash != ANNOTATOR_MODEL_SHA_LOOKUP[annotator]:
            try:
                annotator_full_path.unlink()
                logger.error(
                    f"Deleted annotator file {annotator} as it was corrupt.",
                )
            except OSError:
                logger.error(
                    f"Annotator file {annotator} is corrupt. Please delete it and try again.",
                )
                logger.error(f"File location: {annotator_full_path}")
            return False
    return True


__all__ = [
    "download_all_controlnet_annotators",
    "validate_all_controlnet_annotators",
]
