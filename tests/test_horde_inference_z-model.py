# test_horde.py

import pytest
from PIL import Image

from hordelib.horde import HordeLib

from .testing_shared_functions import check_single_inference_image_similarity


class TestHordeInferenceZImageTurbo:

    @pytest.mark.default_z_image_turbo_model
    def test_z_image_turbo_text_to_image(
        self,
        hordelib_instance: HordeLib,
        z_image_turbo_base_model_name: str,
    ):
        data = {
            "sampler_name": "k_euler",
            "cfg_scale": 1,
            "denoising_strength": 1.0,
            "seed": 1886,
            "height": 1024,
            "width": 1024,
            "karras": False,
            "tiling": False,
            "hires_fix": False,
            "clip_skip": 1,
            "control_type": None,
            "image_is_control": False,
            "return_control_map": False,
            "prompt": (
                'a cyberpunk text that says "Z-Image Turbo Horde Engine" in neon lights, vibrant colors, '
                "futuristic cityscape background, high detail, digital art"
            ),
            "ddim_steps": 9,
            "n_iter": 1,
            "model": z_image_turbo_base_model_name,
        }
        pil_image = hordelib_instance.basic_inference_single_image(data).image
        assert pil_image is not None
        assert isinstance(pil_image, Image.Image)

        img_filename = "z_image_turbo_text_to_image.png"
        pil_image.save(f"images/{img_filename}", quality=100)

        assert check_single_inference_image_similarity(
            f"images_expected/{img_filename}",
            pil_image,
        )
