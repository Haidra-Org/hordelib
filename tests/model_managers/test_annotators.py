import glob
import pathlib

import pytest
from PIL import Image


class TestHordePreloadAnnotators:
    def test_preload_annotators(self):
        import hordelib

        hordelib.initialise({}, True)

        from hordelib.shared_model_manager import SharedModelManager

        SharedModelManager.preloadAnnotators()

    def test_check_sha_annotators(self):
        from hordelib.cache import get_cache_directory
        from hordelib.model_manager.base import BaseModelManager

        annotatorCacheDir = pathlib.Path(get_cache_directory()).joinpath("controlnet").joinpath("annotator")
        annotators = glob.glob("*.pt*", root_dir=annotatorCacheDir)
        for annotator in annotators:
            hash = BaseModelManager.get_file_sha256_hash(annotatorCacheDir.joinpath(annotator))
            print(f"{annotator}: {hash}")  # XXX # TODO Validate hashes
