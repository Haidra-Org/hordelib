from hordelib.horde import HordeLib


class TestPayloadMapping:
    def test_validate_payload_with_lora(self, hordelib_instance: HordeLib):
        data = {
            "sampler_name": "k_lms",
            "cfg_scale": 5,
            "denoising_strength": 0.75,
            "seed": "23113",
            "height": 512,
            "width": 512,
            "karras": False,
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

        # Missing key
        result = data.copy()
        result = hordelib_instance._validate_data_structure(result)
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
        result = hordelib_instance._validate_data_structure(result)
        assert "loras" in result, "Lost the lora attribute in our payload"
        assert isinstance(result["loras"], list), "Lora attribute is not a list"
        assert len(result["loras"]) == 1, "Unexpected number of loras in payload"
        assert result["loras"][0]["name"] == "briscou's gingers", "We lost Briscou's gingers"
        assert result["loras"][0]["model"] == 0.5, "Unexpected lora model weight"
        assert result["loras"][0]["clip"] == 0.4, "Unexpected lora model clip"
