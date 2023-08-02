# test_horde.py
import pytest
from PIL.Image import Image

from hordelib.clip.interrogate import Interrogator
from hordelib.horde import HordeLib
from hordelib.shared_model_manager import SharedModelManager


class TestHordeClip:
    @pytest.fixture(scope="class")
    def interrogator(self) -> type[Interrogator]:
        return Interrogator

    @pytest.fixture(autouse=True, scope="class")
    def setup_and_teardown(self, shared_model_manager: type[SharedModelManager]):
        shared_model_manager.load_model_managers(["clip"])
        shared_model_manager.manager.load("ViT-L/14")
        assert shared_model_manager.manager.is_model_loaded("ViT-L/14") is True
        yield
        shared_model_manager.manager.unload_model("ViT-L/14")
        assert not shared_model_manager.manager.is_model_loaded("ViT-L/14")

    def test_clip_similarities(
        self,
        shared_model_manager: type[SharedModelManager],
        db0_test_image: Image,
    ):
        word_list = ["outlaw", "explosion", "underwater"]
        model_info = shared_model_manager.manager.loaded_models["ViT-L/14"]
        interrogator = Interrogator(model_info)
        similarity_result = interrogator(
            image=db0_test_image,
            text_array=word_list,
            similarity=True,
        )

        assert type(similarity_result) is dict
        assert "default" in similarity_result
        assert similarity_result["default"]["outlaw"] > 0.15
        assert similarity_result["default"]["explosion"] > 0.15
        assert similarity_result["default"]["underwater"] < 0.15

    def test_clip_rankings(
        self,
        shared_model_manager: type[SharedModelManager],
        db0_test_image: Image,
    ):
        model_info = shared_model_manager.manager.loaded_models["ViT-L/14"]
        interrogator = Interrogator(model_info)
        ranking_result = interrogator(
            image=db0_test_image,
            rank=True,
        )
        assert type(ranking_result) is dict
        assert "artists" in ranking_result
        assert "mediums" in ranking_result
