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
            logger.init_warn(
                f"Annotator file {annotator} is not in the model database. Ignoring...",
                status="Warning",
            )
            logger.init_warn(f"File location: {annotator_full_path}", status="Warning")
            continue

        hash = BaseModelManager.get_file_sha256_hash(annotator_full_path)
        if hash != ANNOTATOR_MODEL_SHA_LOOKUP[annotator]:
            try:
                annotator_full_path.unlink()
                logger.init_err(
                    f"Deleted annotator file {annotator} as it was corrupt.",
                    status="Error",
                )
            except OSError:
                logger.init_err(
                    f"Annotator file {annotator} is corrupt. Please delete it and try again.",
                    status="Error",
                )
                logger.init_err(f"File location: {annotator_full_path}", status="Error")
            return False
    return True


__all__ = [
    "download_all_controlnet_annotators",
    "validate_all_controlnet_annotators",
]
