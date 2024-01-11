from collections.abc import Generator

import pytest
from PIL import Image

from hordelib.horde import HordeLib
from hordelib.shared_model_manager import SharedModelManager

from .testing_shared_functions import check_list_inference_images_similarity, check_single_inference_image_similarity


class TestHordeInference:
    @pytest.fixture(scope="class")
    def inpainting_model_for_testing(
        self,
        shared_model_manager: type[SharedModelManager],
    ) -> Generator[str, None, None]:
        """Loads the inpainting model for testing.
        This fixture returns the (str) model name."""
        model_name = "Deliberate Inpainting"
        if not shared_model_manager.manager.download_model(model_name):
            shared_model_manager.manager.download_model(model_name)
        assert shared_model_manager.manager.is_model_available(model_name)
        yield model_name

    def test_inpainting_alpha_mask(
        self,
        inpainting_model_for_testing: str,
        hordelib_instance: HordeLib,
    ):
        data = {
            "sampler_name": "euler",
            "cfg_scale": 8,
            "denoising_strength": 1,
            "seed": 836138046008,
            "height": 512,
            "width": 512,
            "karras": False,
            "tiling": False,
            "hires_fix": False,
            "clip_skip": 1,
            "control_type": None,
            "image_is_control": False,
            "return_control_map": False,
            "prompt": "a dinosaur",
            "ddim_steps": 20,
            "n_iter": 1,
            "model": inpainting_model_for_testing,
            "source_image": Image.open("images/test_inpaint_alpha.png"),
            "source_processing": "inpainting",
        }
        pil_image = hordelib_instance.basic_inference_single_image(data).image
        assert pil_image is not None

        img_filename = "inpainting_mask_alpha.png"

        assert isinstance(pil_image, Image.Image)

        pil_image.save(f"images/{img_filename}", quality=100)

        assert check_single_inference_image_similarity(
            f"images_expected/{img_filename}",
            pil_image,
        )

    def test_inpainting_separate_mask(
        self,
        inpainting_model_for_testing: str,
        hordelib_instance: HordeLib,
    ):
        data = {
            "sampler_name": "euler",
            "cfg_scale": 8,
            "denoising_strength": 1,
            "seed": 836138046008,
            "height": 512,
            "width": 512,
            "karras": False,
            "tiling": False,
            "hires_fix": False,
            "clip_skip": 1,
            "control_type": None,
            "image_is_control": False,
            "return_control_map": False,
            "prompt": "a dinosaur",
            "ddim_steps": 20,
            "n_iter": 1,
            "model": inpainting_model_for_testing,
            "source_image": Image.open("images/test_inpaint_original.png"),
            "source_mask": Image.open("images/test_inpaint_mask.png"),
            "source_processing": "inpainting",
        }
        pil_image = hordelib_instance.basic_inference_single_image(data).image
        assert pil_image is not None

        img_filename = "inpainting_mask_separate.png"

        assert isinstance(pil_image, Image.Image)

        pil_image.save(f"images/{img_filename}", quality=100)

        assert check_single_inference_image_similarity(
            f"images_expected/{img_filename}",
            pil_image,
        )

    def test_inpainting_alpha_mask_mountains(
        self,
        inpainting_model_for_testing: str,
        hordelib_instance: HordeLib,
    ):
        data = {
            "sampler_name": "k_dpmpp_2m",
            "cfg_scale": 7.5,
            "denoising_strength": 0.72,
            "seed": 1312,
            "height": 512,
            "width": 512,
            "karras": False,
            "tiling": False,
            "hires_fix": False,
            "clip_skip": 1,
            "control_type": None,
            "image_is_control": False,
            "return_control_map": False,
            "prompt": "a river through the mountains",
            "ddim_steps": 20,
            "n_iter": 1,
            "model": inpainting_model_for_testing,
            "source_image": Image.open("images/test_inpaint.png"),
            "source_processing": "inpainting",
        }
        pil_image = hordelib_instance.basic_inference_single_image(data).image
        assert pil_image is not None

        assert isinstance(pil_image, Image.Image)

        assert pil_image.size == (512, 512)

        img_filename = "inpainting_mountains.png"
        pil_image.save(f"images/{img_filename}", quality=100)

        assert check_single_inference_image_similarity(
            f"images_expected/{img_filename}",
            pil_image,
        )

    def test_outpainting_alpha_mask_mountains(
        self,
        inpainting_model_for_testing: str,
        hordelib_instance: HordeLib,
    ):
        data = {
            "sampler_name": "euler",
            "cfg_scale": 8.0,
            "denoising_strength": 1.0,
            "seed": 836913938046008,
            "height": 512,
            "width": 512,
            "karras": False,
            "tiling": False,
            "hires_fix": False,
            "clip_skip": 1,
            "control_type": None,
            "image_is_control": False,
            "return_control_map": False,
            "prompt": "a river through the mountains, blue sky with clouds.",
            "ddim_steps": 20,
            "n_iter": 1,
            "model": inpainting_model_for_testing,
            "source_image": Image.open("images/test_outpaint.png"),
            "source_processing": "outpainting",
        }
        pil_image = hordelib_instance.basic_inference_single_image(data).image

        assert pil_image is not None
        assert isinstance(pil_image, Image.Image)
        assert pil_image.size == (512, 512)

        img_filename = "outpainting_mountains.png"

        pil_image.save(f"images/{img_filename}", quality=100)

        assert check_single_inference_image_similarity(
            f"images_expected/{img_filename}",
            pil_image,
        )

    def test_inpainting_n_iter(
        self,
        inpainting_model_for_testing: str,
        hordelib_instance: HordeLib,
    ):
        data = {
            "sampler_name": "euler",
            "cfg_scale": 8,
            "denoising_strength": 1,
            "seed": 1312,
            "height": 512,
            "width": 512,
            "karras": False,
            "tiling": False,
            "hires_fix": False,
            "clip_skip": 1,
            "control_type": None,
            "image_is_control": False,
            "return_control_map": False,
            "prompt": "a dinosaur",
            "ddim_steps": 20,
            "n_iter": 2,
            "model": inpainting_model_for_testing,
            "source_image": Image.open("images/test_inpaint_original.png"),
            "source_mask": Image.open("images/test_inpaint_mask.png"),
            "source_processing": "inpainting",
        }
        image_results = hordelib_instance.basic_inference(data)

        assert len(image_results) == 2

        img_pairs_to_check = []

        img_filename_base = "inpainting_n_iter_{0}.png"

        for i, image_result in enumerate(image_results):
            assert image_result.image is not None
            assert isinstance(image_result.image, Image.Image)

            img_filename = img_filename_base.format(i)

            image_result.image.save(f"images/{img_filename}", quality=100)
            img_pairs_to_check.append((f"images_expected/{img_filename}", image_result.image))

        assert check_list_inference_images_similarity(img_pairs_to_check)
