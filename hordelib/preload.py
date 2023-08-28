import glob
from pathlib import Path

from loguru import logger

from hordelib.comfy_horde import ANNOTATOR_MODEL_SHA_LOOKUP, download_all_controlnet_annotators
from hordelib.model_manager.base import BaseModelManager


def validate_all_controlnet_annotators(annotatorPath: Path) -> int:
    # See if the file `.controlnet_annotators` exists
    if annotatorPath.joinpath(".controlnet_annotators").exists():
        return len(ANNOTATOR_MODEL_SHA_LOOKUP)

    annotatorPath.mkdir(parents=True, exist_ok=True)

    validated_file_num = 0
    annotators = glob.glob("*.pt*", root_dir=annotatorPath)
    for annotator in annotators:
        annotator_full_path = annotatorPath.joinpath(annotator)

        if annotator not in ANNOTATOR_MODEL_SHA_LOOKUP:
            logger.warning(
                f"Annotator file {annotator} is not in the model database. Ignoring...",
            )
            logger.warning(f"File location: {annotator_full_path}")
            validated_file_num += 1
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
            return validated_file_num

    # Create a file called `.controlnet_annotators` to indicate that all annotators are valid
    annotatorPath.joinpath(".controlnet_annotators").touch()

    return validated_file_num


__all__ = [
    "download_all_controlnet_annotators",
    "validate_all_controlnet_annotators",
]
