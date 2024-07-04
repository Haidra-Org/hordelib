# test_horde.py

import pytest
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
            "ddim_steps": 30,
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
            "ddim_steps": 30,
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

    def test_cascade_image_to_image(
        self,
        stable_cascade_base_model_name: str,
        hordelib_instance: HordeLib,
    ):
        data = {
            "sampler_name": "k_euler",
            "cfg_scale": 7.5,
            "denoising_strength": 0.4,
            "seed": 1312,
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
                "a medieval fantasy swashbuckler with a big floppy hat walking towards "
                "a camera while there's an explosion in the background"
            ),
            "ddim_steps": 30,
            "n_iter": 2,
            "model": stable_cascade_base_model_name,
            "source_image": Image.open("images/test_db0.jpg"),
            "source_processing": "img2img",
        }
        image_results = hordelib_instance.basic_inference(data)

        assert len(image_results) == 2

        img_pairs_to_check = []

        img_filename_base = "stable_cascade_image_to_image_n_iter_{0}.png"

        for i, image_result in enumerate(image_results):
            assert image_result.image is not None
            assert isinstance(image_result.image, Image.Image)

            img_filename = img_filename_base.format(i)

            image_result.image.save(f"images/{img_filename}", quality=100)
            img_pairs_to_check.append((f"images_expected/{img_filename}", image_result.image))

        assert check_list_inference_images_similarity(img_pairs_to_check)

    def test_cascade_image_remix_single(
        self,
        stable_cascade_base_model_name: str,
        hordelib_instance: HordeLib,
    ):
        data = {
            "sampler_name": "k_euler",
            "cfg_scale": 4,
            "denoising_strength": 1,
            "seed": 1312,
            "height": 1024,
            "width": 1024,
            "karras": False,
            "tiling": False,
            "hires_fix": False,
            "clip_skip": 1,
            "control_type": None,
            "image_is_control": False,
            "return_control_map": False,
            "prompt": "A herd of goats grazing under the sunset",
            "ddim_steps": 30,
            "n_iter": 1,
            "model": stable_cascade_base_model_name,
            "source_image": Image.open("images/test_mountains.png"),
            "source_processing": "remix",
        }
        pil_image = hordelib_instance.basic_inference_single_image(data).image
        assert pil_image is not None
        assert isinstance(pil_image, Image.Image)

        img_filename = "stable_cascade_image_remix_single.png"
        pil_image.save(f"images/{img_filename}", quality=100)

        assert check_single_inference_image_similarity(
            f"images_expected/{img_filename}",
            pil_image,
        )

    def test_cascade_image_remix_double(
        self,
        stable_cascade_base_model_name: str,
        hordelib_instance: HordeLib,
    ):
        data = {
            "sampler_name": "k_euler",
            "cfg_scale": 4,
            "denoising_strength": 1,
            "seed": 1312,
            "height": 1024,
            "width": 1024,
            "karras": False,
            "tiling": False,
            "hires_fix": False,
            "clip_skip": 1,
            "control_type": None,
            "image_is_control": False,
            "return_control_map": False,
            "prompt": "A herd of goats grazing",
            "ddim_steps": 30,
            "n_iter": 1,
            "model": stable_cascade_base_model_name,
            "source_image": Image.open("images/test_mountains.png"),
            "source_processing": "remix",
            "extra_source_images": [
                {
                    "image": Image.open("images/test_sunset.png"),
                },
            ],
        }
        pil_image = hordelib_instance.basic_inference_single_image(data).image
        assert pil_image is not None
        assert isinstance(pil_image, Image.Image)

        img_filename = "stable_cascade_image_remix_double.png"
        pil_image.save(f"images/{img_filename}", quality=100)

        assert check_single_inference_image_similarity(
            f"images_expected/{img_filename}",
            pil_image,
        )

    def test_cascade_image_remix_double_weak(
        self,
        stable_cascade_base_model_name: str,
        hordelib_instance: HordeLib,
    ):
        data = {
            "sampler_name": "k_euler",
            "cfg_scale": 4,
            "denoising_strength": 1,
            "seed": 1312,
            "height": 1024,
            "width": 1024,
            "karras": False,
            "tiling": False,
            "hires_fix": False,
            "clip_skip": 1,
            "control_type": None,
            "image_is_control": False,
            "return_control_map": False,
            "prompt": "A herd of goats grazing",
            "ddim_steps": 30,
            "n_iter": 1,
            "model": stable_cascade_base_model_name,
            "source_image": Image.open("images/test_mountains.png"),
            "source_processing": "remix",
            "extra_source_images": [
                {
                    "image": Image.open("images/test_sunset.png"),
                    "strength": 0.05,
                },
            ],
        }
        pil_image = hordelib_instance.basic_inference_single_image(data).image
        assert pil_image is not None
        assert isinstance(pil_image, Image.Image)

        img_filename = "stable_cascade_image_remix_double_weak.png"
        pil_image.save(f"images/{img_filename}", quality=100)

        assert check_single_inference_image_similarity(
            f"images_expected/{img_filename}",
            pil_image,
        )

    def test_cascade_image_remix_triple(
        self,
        stable_cascade_base_model_name: str,
        hordelib_instance: HordeLib,
    ):
        data = {
            "sampler_name": "k_euler",
            "cfg_scale": 4,
            "denoising_strength": 1,
            "seed": 1312,
            "height": 1024,
            "width": 1024,
            "karras": False,
            "tiling": False,
            "hires_fix": False,
            "clip_skip": 1,
            "control_type": None,
            "image_is_control": False,
            "return_control_map": False,
            "prompt": "Baking Sun",
            "ddim_steps": 30,
            "n_iter": 1,
            "model": stable_cascade_base_model_name,
            "source_image": Image.open("images/test_mountains.png"),
            "source_processing": "remix",
            "extra_source_images": [
                {
                    "image": Image.open("images/test_sunset.png"),
                },
                {
                    "image": Image.open("images/test_db0.jpg"),
                },
            ],
        }
        pil_image = hordelib_instance.basic_inference_single_image(data).image
        assert pil_image is not None
        assert isinstance(pil_image, Image.Image)

        img_filename = "stable_cascade_image_remix_triple.png"
        pil_image.save(f"images/{img_filename}", quality=100)

        assert check_single_inference_image_similarity(
            f"images_expected/{img_filename}",
            pil_image,
        )

    def test_cascade_text_to_image_hires_2pass(
        self,
        hordelib_instance: HordeLib,
        stable_cascade_base_model_name: str,
    ):
        data = {
            "sampler_name": "k_euler",
            "cfg_scale": 4,
            "denoising_strength": 1.0,
            "hires_fix_denoising_strength": 0.5,
            "seed": 1312,
            "height": 1536,
            "width": 1024,
            "karras": False,
            "tiling": False,
            "hires_fix": True,
            "clip_skip": 1,
            "control_type": None,
            "image_is_control": False,
            "return_control_map": False,
            "prompt": (
                "Lucid Creations, Deep Forest, Moss, ethereal, dreamlike, surreal, "
                "beautiful, illustration, incredible detail, 8k, abstract"
            ),
            "ddim_steps": 30,
            "n_iter": 1,
            "model": stable_cascade_base_model_name,
        }

        pil_image = hordelib_instance.basic_inference_single_image(data).image
        assert pil_image is not None
        assert isinstance(pil_image, Image.Image)

        img_filename = "stable_cascade_text_to_image_2pass.png"
        pil_image.save(f"images/{img_filename}", quality=100)

        # Check that denoising strength works
        data["hires_fix_denoising_strength"] = 0
        pil_image2 = hordelib_instance.basic_inference_single_image(data).image
        assert pil_image2 is not None
        assert isinstance(pil_image2, Image.Image)

        img_filename_denoise_0 = "stable_cascade_text_to_image_2pass_denoise_0.png"
        pil_image2.save(f"images/{img_filename_denoise_0}", quality=100)

        assert pil_image2 is not None
        assert isinstance(pil_image2, Image.Image)
        with pytest.raises(AssertionError):
            check_single_inference_image_similarity(
                pil_image2,
                pil_image,
                exception_on_fail=True,
            )
        assert check_single_inference_image_similarity(
            f"images_expected/{img_filename}",
            pil_image,
        )
        assert check_single_inference_image_similarity(
            f"images_expected/{img_filename_denoise_0}",
            pil_image2,
        )
