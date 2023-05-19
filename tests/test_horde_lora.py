# test_horde_lora.py
import pytest
from PIL import Image

from hordelib.horde import HordeLib
from hordelib.shared_model_manager import SharedModelManager


class TestHordeLora:
    @pytest.fixture(autouse=True, scope="class")
    def setup_and_teardown(self):
        TestHordeLora.horde = HordeLib()

        TestHordeLora.default_model_manager_args = {
            "compvis": True,
        }
        SharedModelManager.loadModelManagers(**self.default_model_manager_args)
        assert SharedModelManager.manager is not None
        SharedModelManager.manager.load("Deliberate")
        yield
        del TestHordeLora.horde
        SharedModelManager._instance = None
        SharedModelManager.manager = None

    def test_validate_payload(self):
        data = {
            "sampler_name": "k_lms",
            "cfg_scale": 5,
            "denoising_strength": 0.75,
            "seed": "23113",
            "height": 512,
            "width": 512,
            "karras": True,
            "tiling": False,
            "hires_fix": False,
            "clip_skip": 1,
            "control_type": None,
            "image_is_control": False,
            "return_control_map": False,
            "prompt": "a dog ### cat, mouse, lion",
            "ddim_steps": 30,
            "n_iter": 1,
            "model": "Deliberate",
        }

        assert self.horde is not None

        # Missing key
        result = data.copy()
        self.horde._check_payload(result)
        assert "loras" in result, "Failed to fix missing lora attribute in payload"

        # Bad and good lora
        data["loras"] = [
            {"clip": "this is bad"},
            {
                "name": "briscou's gingers",
                "model": 0.5,
                "clip": 0.4,
            },
        ]
        result = data.copy()
        self.horde._check_payload(result)
        assert "loras" in result, "Lost the lora attribute in our payload"
        assert len(result["loras"]) == 1, "Unexpected number of loras in payload"
        assert result["loras"][0]["name"] == "briscou's gingers", "We lost Briscou's gingers"
        assert result["loras"][0]["model"] == 0.5, "Unexpected lora model weight"
        assert result["loras"][0]["clip"] == 0.4, "Unexpected lora model clip"

    def test_text_to_image_lora_red(self):

        # Red
        data = {
            "sampler_name": "k_euler",
            "cfg_scale": 8.0,
            "denoising_strength": 1.0,
            "seed": 304886399544324,
            "height": 512,
            "width": 512,
            "karras": True,
            "tiling": False,
            "hires_fix": False,
            "clip_skip": 1,
            "control_type": None,
            "image_is_control": False,
            "return_control_map": False,
            "prompt": "a dark magical crystal, GlowingRunesAIV2_red",
            "loras": [{"name": "GlowingRunesAIV6", "model": 1.0, "clip": 1.0}],
            "ddim_steps": 20,
            "n_iter": 1,
            "model": "Deliberate",
        }
        assert self.horde is not None
        pil_image = self.horde.basic_inference(data)
        assert pil_image is not None
        pil_image.save("images/horde_lora_red.webp", quality=90)

    def test_text_to_image_lora_blue(self):

        # Blue
        data = {
            "sampler_name": "k_euler",
            "cfg_scale": 8.0,
            "denoising_strength": 1.0,
            "seed": 851616030078638,
            "height": 512,
            "width": 512,
            "karras": True,
            "tiling": False,
            "hires_fix": False,
            "clip_skip": 1,
            "control_type": None,
            "image_is_control": False,
            "return_control_map": False,
            "prompt": "a dark magical crystal, GlowingRunesAIV2_paleblue",
            "loras": [{"name": "GlowingRunesAIV6", "model": 1.0, "clip": 1.0}],
            "ddim_steps": 20,
            "n_iter": 1,
            "model": "Deliberate",
        }
        assert self.horde is not None
        pil_image = self.horde.basic_inference(data)
        assert pil_image is not None
        pil_image.save("images/horde_lora_blue.webp", quality=90)

    def test_text_to_image_lora_chained(self):

        data = {
            "sampler_name": "k_euler",
            "cfg_scale": 8.0,
            "denoising_strength": 1.0,
            "seed": 304886399544324,
            "height": 512,
            "width": 512,
            "karras": True,
            "tiling": False,
            "hires_fix": False,
            "clip_skip": 1,
            "control_type": None,
            "image_is_control": False,
            "return_control_map": False,
            "prompt": "a dark magical crystal, GlowingRunesAIV2_red, Dr490nSc4leAI",
            "loras": [
                {"name": "GlowingRunesAIV6", "model": 1.0, "clip": 1.0},
                {"name": "Dra9onScaleAI", "model": 1.0, "clip": 1.0},
            ],
            "ddim_steps": 20,
            "n_iter": 1,
            "model": "Deliberate",
        }
        assert self.horde is not None
        pil_image = self.horde.basic_inference(data)
        assert pil_image is not None
        pil_image.save("images/horde_lora_multiple.webp", quality=90)
