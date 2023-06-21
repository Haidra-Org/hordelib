# test_horde.py
import pytest
from PIL.Image import Image

from hordelib.blip.caption import Caption
from hordelib.shared_model_manager import SharedModelManager


class TestHordeBlip:
    @pytest.fixture(autouse=True)
    def setup_and_teardown(self, shared_model_manager: type[SharedModelManager]):
        shared_model_manager.load_model_managers(["blip"])
        shared_model_manager.manager.load("BLIP_Large")
        assert shared_model_manager.manager.is_model_loaded("BLIP_Large") is True
        yield
        shared_model_manager.manager.unload_model("BLIP_Large")
        assert not shared_model_manager.manager.is_model_loaded("BLIP_Large")

    def test_blip_large_caption(
        self,
        shared_model_manager: type[SharedModelManager],
        db0_test_image: Image,
    ):
        model = shared_model_manager.manager.loaded_models["BLIP_Large"]
        caption_class = Caption(model)
        caption = caption_class(
            image=db0_test_image,
            sample=True,
            num_beams=7,
            min_length=20,
            max_length=50,
            top_p=0.9,
            repetition_penalty=1.4,
        )
        assert caption is not None
        assert len(caption) > 20
