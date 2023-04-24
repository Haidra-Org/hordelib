# test_horde.py
import json

import pytest
from PIL import Image

from hordelib.horde import HordeLib
from hordelib.shared_model_manager import SharedModelManager


class TestHordeInference:
    @pytest.fixture(autouse=True, scope="class")
    def setup_and_teardown(self):
        TestHordeInference.horde = HordeLib()

        TestHordeInference.default_model_manager_args = {
            "compvis": True,
        }
        SharedModelManager.loadModelManagers(**self.default_model_manager_args)
        assert SharedModelManager.manager is not None
        SharedModelManager.manager.load("Deliberate")
        yield
        del TestHordeInference.horde
        SharedModelManager._instance = None
        SharedModelManager.manager = None

    def test_text_to_image(self):
        data = {
            "sampler_name": "k_dpmpp_2m",
            "cfg_scale": 7.5,
            "denoising_strength": 1.0,
            "seed": 123456789,
            "height": 512,
            "width": 512,
            "karras": True,
            "tiling": False,
            "hires_fix": False,
            "clip_skip": 1,
            "control_type": None,
            "image_is_control": False,
            "return_control_map": False,
            "prompt": "a secret metadata store",
            "ddim_steps": 25,
            "n_iter": 1,
            "model": "Deliberate",
        }
        assert self.horde is not None
        png_data = self.horde.basic_inference(data, rawpng=True)
        assert png_data is not None
        image = Image.open(png_data)
        metadata = image.info
        assert "prompt" in metadata
        info = json.loads(metadata["prompt"])
        assert info["prompt"]["inputs"]["text"] == "a secret metadata store"
