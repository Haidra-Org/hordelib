# test_horde.py
import pytest
from PIL import Image

from hordelib.clip.interrogate import Interrogator
from hordelib.horde import HordeLib
from hordelib.shared_model_manager import SharedModelManager


class TestHordeClip:
    @pytest.fixture(autouse=True, scope="class")
    def setup_and_teardown(self):
        TestHordeClip.horde = HordeLib()

        TestHordeClip.default_model_manager_args = {
            # aitemplate
            # "blip": True,
            "clip": True,
            # "codeformer": True,
            # "compvis": True,
            # "controlnet": True,
            # "diffusers": True,
            # "esrgan": True,
            # "gfpgan": True,
            # "safety_checker": True,
        }
        TestHordeClip.image = Image.open("images/test_db0.jpg")
        SharedModelManager.loadModelManagers(**self.default_model_manager_args)
        assert SharedModelManager.manager is not None
        SharedModelManager.manager.load("ViT-L/14")
        yield
        del TestHordeClip.horde
        SharedModelManager._instance = None
        SharedModelManager.manager = None

    def test_clip_similarities(self):
        assert SharedModelManager.manager.is_model_loaded("ViT-L/14") is True
        word_list = ["outlaw", "explosion", "underwater"]
        model_info = SharedModelManager.manager.loaded_models["ViT-L/14"]
        interrogator = Interrogator(model_info)
        similarity_result = interrogator(
            image=self.image,
            text_array=word_list,
            similarity=True,
        )
        assert "default" in similarity_result
        assert similarity_result["default"]["outlaw"] > 0.15
        assert similarity_result["default"]["explosion"] > 0.15
        assert similarity_result["default"]["underwater"] < 0.15

    def test_clip_rankings(self):
        assert SharedModelManager.manager.is_model_loaded("ViT-L/14") is True
        model_info = SharedModelManager.manager.loaded_models["ViT-L/14"]
        interrogator = Interrogator(model_info)
        ranking_result = interrogator(
            image=self.image,
            rank=True,
        )
        assert type(ranking_result) is dict
        assert "artists" in ranking_result
        assert "mediums" in ranking_result
