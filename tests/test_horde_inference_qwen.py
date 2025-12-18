# test_horde.py

import pytest
from PIL import Image

from hordelib.horde import HordeLib

from .testing_shared_functions import check_single_inference_image_similarity


class TestHordeInferenceQwen:

    @pytest.mark.skip(reason="32GVRAM")
    @pytest.mark.default_qwen_model
    def test_qwen_image_fp8_text_to_image(
        self,
        hordelib_instance: HordeLib,
        qwen_image_fp8_base_model_name: str,
    ):
        data = {
            "sampler_name": "k_euler",
            "cfg_scale": 2.5,
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
                'a cyberpunk text that says "Qwen Horde Engine" in neon lights, vibrant colors, '
                "futuristic cityscape background, high detail, digital art"
            ),
            "ddim_steps": 20,
            "n_iter": 1,
            "model": qwen_image_fp8_base_model_name,
        }
        pil_image = hordelib_instance.basic_inference_single_image(data).image
        assert pil_image is not None
        assert isinstance(pil_image, Image.Image)

        img_filename = "qwen_image_fp8_text_to_image.png"
        pil_image.save(f"images/{img_filename}", quality=100)

        assert check_single_inference_image_similarity(
            f"images_expected/{img_filename}",
            pil_image,
        )
