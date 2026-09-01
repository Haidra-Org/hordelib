"""Test Anima image generation through the shared Qwen-family pipeline."""

import pytest
from PIL import Image

from hordelib.horde import HordeLib, ResultingImageReturn

from .testing_shared_functions import check_single_inference_image_similarity


class TestHordeInferenceAnima:
    @pytest.mark.default_anima_model
    def test_anima_turbo_text_to_image(
        self,
        hordelib_instance: HordeLib,
        anima_turbo_base_model_name: str,
    ):
        data = {
            "sampler_name": "k_euler",
            "cfg_scale": 1.0,
            "denoising_strength": 1.0,
            "seed": 2026,
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
                "A male mage standing beneath a luminous moon, detailed anime illustration, "
                "intricate magical symbols, flowing robes, and a staff emitting a soft glow, "
                "surrounded by floating runes and mystical energy, cinematic lighting, "
                "masterpiece, best quality, score_7, rating:safe, rating:g"
                "###nsfw, explicit, worst quality, low quality, score_1, score_2, score_3, artist name, blurry, "
                "jpeg artifacts, chromatic aberration"
            ),
            "ddim_steps": 8,
            "n_iter": 1,
            "model": anima_turbo_base_model_name,
        }
        result = hordelib_instance.basic_inference_single_image(data)

        assert isinstance(result, ResultingImageReturn)
        assert isinstance(result.image, Image.Image)
        assert not result.faults

        img_filename = "anima_turbo_text_to_image.png"

        pil_image = result.image
        pil_image.save(f"images/{img_filename}", quality=100)

        assert check_single_inference_image_similarity(
            f"images_expected/{img_filename}",
            pil_image,
        )
