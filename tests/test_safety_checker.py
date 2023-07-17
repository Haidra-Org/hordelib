# test_horde.py
import pytest
from PIL import Image

from hordelib.horde import HordeLib
from hordelib.safety_checker import is_image_nsfw
from hordelib.shared_model_manager import SharedModelManager

# XXX Should find a way to test for a positive NSFW result without something in the repo?


class TestHordeSaftyChecker:
    def test_safety_checker_with_preload(
        self,
        shared_model_manager: type[SharedModelManager],
        db0_test_image: Image.Image,
    ):
        assert shared_model_manager.manager.load("safety_checker", cpu_only=True)
        assert shared_model_manager.manager.safety_checker
        assert shared_model_manager.manager.safety_checker.is_model_loaded("safety_checker") is True
        assert is_image_nsfw(db0_test_image) is False
        assert shared_model_manager.manager.unload_model("safety_checker")

    def test_safety_checker_without_preload(
        self,
        shared_model_manager: type[SharedModelManager],
        db0_test_image: Image.Image,
    ):
        assert shared_model_manager.manager.safety_checker
        assert shared_model_manager.manager.safety_checker.is_model_loaded("safety_checker") is False
        assert is_image_nsfw(db0_test_image) is False
        assert shared_model_manager.manager.unload_model("safety_checker")
