# test_horde.py

import pytest
from PIL import Image

from hordelib.horde import HordeLib
from hordelib.shared_model_manager import SharedModelManager

from .testing_shared_functions import check_list_inference_images_similarity, check_single_inference_image_similarity


class TestHordeInferenceFlux:

    @pytest.mark.default_flux1_model
    def test_flux_schnell_fp8_text_to_image(
        self,
        hordelib_instance: HordeLib,
        flux1_schnell_fp8_base_model_name: str,
    ):
        data = {
            "sampler_name": "k_euler",
            "cfg_scale": 1,
            "denoising_strength": 1.0,
            "seed": 13122,
            "height": 1024,
            "width": 1024,
            "karras": False,
            "tiling": False,
            "hires_fix": False,
            "clip_skip": 1,
            "control_type": None,
            "image_is_control": False,
            "return_control_map": False,
            "prompt": 'a steampunk text that says "Horde Engine" floating',
            "ddim_steps": 4,
            "n_iter": 1,
            "model": flux1_schnell_fp8_base_model_name,
        }
        pil_image = hordelib_instance.basic_inference_single_image(data).image
        assert pil_image is not None
        assert isinstance(pil_image, Image.Image)

        img_filename = "flux_schnell_fp8_text_to_image.png"
        pil_image.save(f"images/{img_filename}", quality=100)

        assert check_single_inference_image_similarity(
            f"images_expected/{img_filename}",
            pil_image,
        )

    @pytest.mark.default_flux1_model
    def test_flux_schnell_fp8_text_to_image_n_iter(
        self,
        hordelib_instance: HordeLib,
        flux1_schnell_fp8_base_model_name: str,
    ):
        data = {
            "sampler_name": "k_euler",
            "cfg_scale": 1,
            "denoising_strength": 1.0,
            "seed": 13122,
            "height": 1024,
            "width": 1024,
            "karras": False,
            "tiling": False,
            "hires_fix": False,
            "clip_skip": 1,
            "control_type": None,
            "image_is_control": False,
            "return_control_map": False,
            "prompt": 'a steampunk text that says "Horde Engine" floating',
            "ddim_steps": 4,
            "n_iter": 2,
            "model": flux1_schnell_fp8_base_model_name,
        }
        image_results = hordelib_instance.basic_inference(data)

        assert len(image_results) == 2

        img_pairs_to_check = []

        img_filename_base = "flux_schnell_fp8_text_to_image_n_iter_{0}.png"

        for i, image_result in enumerate(image_results):
            assert image_result.image is not None
            assert isinstance(image_result.image, Image.Image)

            img_filename = img_filename_base.format(i)

            image_result.image.save(f"images/{img_filename}", quality=100)
            img_pairs_to_check.append((f"images_expected/{img_filename}", image_result.image))

        assert check_single_inference_image_similarity(
            "images_expected/flux_schnell_fp8_text_to_image.png",
            "images/flux_schnell_fp8_text_to_image_n_iter_0.png",
        )

        assert check_list_inference_images_similarity(img_pairs_to_check)

    @pytest.mark.default_flux1_model
    def test_flux_schnell_fp8_image_to_image(
        self,
        hordelib_instance: HordeLib,
        flux1_schnell_fp8_base_model_name: str,
    ):
        data = {
            "sampler_name": "k_euler",
            "cfg_scale": 1,
            "denoising_strength": 0.8,
            "seed": 13122,
            "height": 1024,
            "width": 1024,
            "karras": False,
            "tiling": False,
            "hires_fix": False,
            "clip_skip": 1,
            "control_type": None,
            "image_is_control": False,
            "return_control_map": False,
            "prompt": "a steampunk cowboy walking away from an explosion",
            "ddim_steps": 4,
            "n_iter": 1,
            "source_image": Image.open("images/test_db0.jpg"),
            "source_processing": "img2img",
            "model": flux1_schnell_fp8_base_model_name,
        }
        pil_image = hordelib_instance.basic_inference_single_image(data).image
        assert pil_image is not None
        assert isinstance(pil_image, Image.Image)

        img_filename = "flux_schnell_fp8_image_to_image.png"
        pil_image.save(f"images/{img_filename}", quality=100)

        assert check_single_inference_image_similarity(
            f"images_expected/{img_filename}",
            pil_image,
        )

    # @pytest.mark.default_flux1_model
    # def test_flux_dev_fp8_image_to_image(
    #     self,
    #     hordelib_instance: HordeLib,
    #     flux1_dev_fp8_base_model_name: str,
    # ):
    #     data = {
    #         "sampler_name": "k_euler",
    #         "cfg_scale": 2,
    #         "denoising_strength": 0.85,
    #         "seed": 13122,
    #         "height": 1024,
    #         "width": 1024,
    #         "karras": False,
    #         "tiling": False,
    #         "hires_fix": False,
    #         "clip_skip": 1,
    #         "control_type": None,
    #         "image_is_control": False,
    #         "return_control_map": False,
    #         "prompt": "a goth cowboy in black walking away from an explosion",
    #         "ddim_steps": 20,
    #         "n_iter": 1,
    #         "source_image": Image.open("images/test_db0.jpg"),
    #         "source_processing": "img2img",
    #         "model": flux1_dev_fp8_base_model_name,
    #     }
    #     pil_image = hordelib_instance.basic_inference_single_image(data).image
    #     assert pil_image is not None
    #     assert isinstance(pil_image, Image.Image)

    #     img_filename = "flux_dev_fp8_image_to_image.png"
    #     pil_image.save(f"images/{img_filename}", quality=100)

    #     assert check_single_inference_image_similarity(
    #         f"images_expected/{img_filename}",
    #         pil_image,
    #     )

    @pytest.mark.default_flux1_model
    def test_flux_dev_fp8_text_to_image_lora(
        self,
        shared_model_manager: type[SharedModelManager],
        hordelib_instance: HordeLib,
        flux1_schnell_fp8_base_model_name: str,
    ):

        lora_schnell_version_id = "812384"
        lora_dev_version_id = "735063"

        if shared_model_manager.manager.lora:
            assert shared_model_manager.manager.lora.fetch_adhoc_lora(
                lora_schnell_version_id,
                timeout=45,
                is_version=True,
            )

            assert shared_model_manager.manager.lora.fetch_adhoc_lora(
                lora_dev_version_id,
                timeout=45,
                is_version=True,
            )

        data = {
            "sampler_name": "k_euler",
            "cfg_scale": 1,
            "denoising_strength": 1.0,
            "seed": 13122,
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
                "A close-up of an magical and ominous obsidian monolith, with an glowing orange cat print inside. "
                "detailmaximizer, dreamy art"
            ),
            "ddim_steps": 4,
            "n_iter": 1,
            "model": flux1_schnell_fp8_base_model_name,
            "loras": [
                {"name": lora_schnell_version_id, "model": 1.0, "clip": 1.0, "is_version": True},
                {"name": lora_dev_version_id, "model": 0.5, "clip": 1.0, "is_version": True},
            ],
        }
        pil_image = hordelib_instance.basic_inference_single_image(data).image
        assert pil_image is not None
        assert isinstance(pil_image, Image.Image)

        img_filename = "flux_schnell_fp8_text_to_image_lora.png"
        pil_image.save(f"images/{img_filename}", quality=100)

        assert check_single_inference_image_similarity(
            f"images_expected/{img_filename}",
            pil_image,
        )
