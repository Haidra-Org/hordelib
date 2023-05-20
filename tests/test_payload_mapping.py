import pytest
from PIL import Image

from hordelib.horde import HordeLib
from hordelib.shared_model_manager import SharedModelManager


class TestPayloadMapping:
    @pytest.fixture(autouse=True, scope="class")
    def setup_and_teardown(self):
        TestPayloadMapping.horde = HordeLib()

        yield
        del TestPayloadMapping.horde

    def test_validate_payload_with_lora(self):
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
        result = self.horde._validate_data_structure(result)
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
        result = self.horde._validate_data_structure(result)
        assert "loras" in result, "Lost the lora attribute in our payload"
        assert len(result["loras"]) == 1, "Unexpected number of loras in payload"
        assert result["loras"][0]["name"] == "briscou's gingers", "We lost Briscou's gingers"
        assert result["loras"][0]["model"] == 0.5, "Unexpected lora model weight"
        assert result["loras"][0]["clip"] == 0.4, "Unexpected lora model clip"
