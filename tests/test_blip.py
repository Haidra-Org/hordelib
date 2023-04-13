# test_horde.py
import pytest
from PIL import Image

from hordelib.blip.caption import Caption
from hordelib.horde import HordeLib
from hordelib.shared_model_manager import SharedModelManager


class TestHordeBlip:
    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        self.horde = HordeLib()

        self.default_model_manager_args = {
            # aitemplate
            "blip": True,
            # "clip": True,
            # "codeformer": True,
            # "compvis": True,
            # "controlnet": True,
            # "diffusers": True,
            # "esrgan": True,
            # "gfpgan": True,
            # "safety_checker": True,
        }
        self.image = Image.open("images/test_db0.jpg")
        SharedModelManager.loadModelManagers(**self.default_model_manager_args)
        assert SharedModelManager.manager is not None
        SharedModelManager.manager.load("BLIP_Large")
        yield
        del self.horde
        SharedModelManager._instance = None
        SharedModelManager.manager = None

    def test_blip_large_caption(self):
        assert SharedModelManager.manager.is_model_loaded("BLIP_Large") is True
        model = SharedModelManager.manager.loaded_models["BLIP_Large"]
        caption_class = Caption(model)
        caption = caption_class(
            image=self.image,
            sample=True,
            num_beams=7,
            min_length=20,
            max_length=50,
            top_p=0.9,
            repetition_penalty=1.4,
        )
        assert caption is not None
        assert len(caption) > 20
