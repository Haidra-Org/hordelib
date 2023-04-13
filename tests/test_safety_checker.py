# test_horde.py
import pytest
from PIL import Image

from hordelib.horde import HordeLib
from hordelib.safety_checker import is_image_nsfw
from hordelib.shared_model_manager import SharedModelManager

# XXX Should find a way to test for a positive NSFW result without something in the repo?


class TestHordeSaftyChecker:
    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        self.horde = HordeLib()

        self.default_model_manager_args = {
            # aitemplate
            # "blip": True,
            # "clip": True,
            # "codeformer": True,
            # "compvis": True,
            # "controlnet": True,
            # "diffusers": True,
            # "esrgan": True,
            # "gfpgan": True,
            "safety_checker": True,
        }
        self.image = Image.open("images/test_db0.jpg")
        SharedModelManager.loadModelManagers(**self.default_model_manager_args)
        assert SharedModelManager.manager is not None
        yield
        del self.horde
        SharedModelManager._instance = None
        SharedModelManager.manager = None

    def test_safety_checker_with_preload(self):
        SharedModelManager.manager.load("safety_checker", cpu_only=True)
        assert SharedModelManager.manager.safety_checker.is_model_loaded("safety_checker") is True
        assert is_image_nsfw(self.image) is False

    def test_safety_checker_without_preload(self):
        assert SharedModelManager.manager.safety_checker.is_model_loaded("safety_checker") is False
        assert is_image_nsfw(self.image) is False
