# test_horde.py
import json

from PIL import Image

from hordelib.horde import HordeLib


class TestHordeInferenceMetadata:
    def test_text_to_image(
        self,
        hordelib_instance: HordeLib,
        stable_diffusion_model_name_for_testing: str,
    ):
        data = {
            "sampler_name": "k_dpmpp_2m",
            "cfg_scale": 7.5,
            "denoising_strength": 1.0,
            "seed": 123456789,
            "height": 512,
            "width": 512,
            "karras": False,
            "tiling": False,
            "hires_fix": False,
            "clip_skip": 1,
            "control_type": None,
            "image_is_control": False,
            "return_control_map": False,
            "prompt": "a secret metadata store",
            "ddim_steps": 25,
            "n_iter": 1,
            "model": stable_diffusion_model_name_for_testing,
        }
        png_data = hordelib_instance.basic_inference_rawpng(data)
        assert png_data is not None
        assert len(png_data) == 1
        image = Image.open(png_data[0])
        metadata = image.info
        assert "prompt" in metadata
        info = json.loads(metadata["prompt"])
        assert info["prompt"]["inputs"]["text"] == "a secret metadata store"
