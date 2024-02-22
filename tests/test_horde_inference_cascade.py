# test_horde.py

from PIL import Image

from hordelib.horde import HordeLib
from tests.testing_shared_functions import (
    check_list_inference_images_similarity,
    check_single_inference_image_similarity,
)


class TestHordeInferenceCascade:
    def test_cascade_text_to_image(
        self,
        hordelib_instance: HordeLib,
        stable_cascade_base_model_name: str,
    ):
        data = {
            "sampler_name": "k_euler",
            "cfg_scale": 4,
            "denoising_strength": 1.0,
            "seed": 1313,
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
                "A magic glowing long sword lying flat on a medieval shop rack, in the style of Dungeons and Dragons. "
                "Splash art, Digital Painting, ornate handle with gems"
            ),
            "ddim_steps": 20,
            "n_iter": 1,
            "model": stable_cascade_base_model_name,
        }

        pil_image = hordelib_instance.basic_inference_single_image(data).image
        assert pil_image is not None
        assert isinstance(pil_image, Image.Image)

        img_filename = "stable_cascade_text_to_image.png"
        pil_image.save(f"images/{img_filename}", quality=100)

        assert check_single_inference_image_similarity(
            f"images_expected/{img_filename}",
            pil_image,
        )

    def test_cascade_text_to_image_n_iter(
        self,
        hordelib_instance: HordeLib,
        stable_cascade_base_model_name: str,
    ):
        data = {
            "sampler_name": "k_euler",
            "cfg_scale": 4,
            "denoising_strength": 1.0,
            "seed": 1313,
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
                "A magic glowing long sword lying flat on a medieval shop rack, in the style of Dungeons and Dragons. "
                "Splash art, Digital Painting, ornate handle with gems"
            ),
            "ddim_steps": 20,
            "n_iter": 2,
            "model": stable_cascade_base_model_name,
        }
        image_results = hordelib_instance.basic_inference(data)

        assert len(image_results) == 2

        img_pairs_to_check = []

        img_filename_base = "stable_cascade_text_to_image_n_iter_{0}.png"

        for i, image_result in enumerate(image_results):
            assert image_result.image is not None
            assert isinstance(image_result.image, Image.Image)

            img_filename = img_filename_base.format(i)

            image_result.image.save(f"images/{img_filename}", quality=100)
            img_pairs_to_check.append((f"images_expected/{img_filename}", image_result.image))

        assert check_single_inference_image_similarity(
            "images_expected/stable_cascade_text_to_image.png",
            "images/stable_cascade_text_to_image_n_iter_0.png",
        )

        assert check_list_inference_images_similarity(img_pairs_to_check)
