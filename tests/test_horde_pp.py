# test_horde.py
import pytest

from hordelib.horde import HordeLib
from hordelib.shared_model_manager import SharedModelManager


class TestHordePostProcessing:
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
            "esrgan": True,
            "gfpgan": True,
            # "safety_checker": True,
        }
        SharedModelManager.loadModelManagers(**self.default_model_manager_args)
        assert SharedModelManager.manager is not None
        SharedModelManager.manager.load("RealESRGAN_x4plus")
        yield
        del self.horde
        SharedModelManager._instance = None
        SharedModelManager.manager = None

    def test_load(self):
        assert (
            SharedModelManager.manager.esrgan.is_model_loaded("RealESRGAN_x4plus")
            is True
        )
