# test_horde.py

import pytest
from PIL import Image

from hordelib.horde import HordeLib
from hordelib.shared_model_manager import SharedModelManager
from tests.testing_shared_functions import check_single_inference_image_similarity


class TestHordeInferenceQRCode:
    @pytest.fixture(autouse=True)
    def setup_and_teardown(self, shared_model_manager: type[SharedModelManager]):
        assert shared_model_manager.manager.controlnet is not None
        for preproc in HordeLib.CONTROLNET_IMAGE_PREPROCESSOR_MAP.keys():
            shared_model_manager.manager.controlnet.download_control_type(preproc, ["stable diffusion 1"])

    def test_qr_code_inference(
        self,
        shared_model_manager: type[SharedModelManager],
        hordelib_instance: HordeLib,
        stable_diffusion_model_name_for_testing: str,
    ):
        data = {
            "sampler_name": "k_euler",
            "cfg_scale": 7.5,
            "denoising_strength": 1.0,
            "seed": 1312,
            "height": 768,
            "width": 768,
            "karras": False,
            "tiling": False,
            "hires_fix": False,
            "clip_skip": 1,
            "prompt": (
                "drawing of two witches performing a seanse around a cauldron. Wispy and Ethereal, sepia colors"
                "###worst quality, bad lighting, deformed, ugly, low contrast"
            ),
            "ddim_steps": 30,
            "n_iter": 1,
            "model": stable_diffusion_model_name_for_testing,
            "workflow": "qr_code",
            "extra_texts": [
                {
                    "text": "https://aihorde.net",
                    "reference": "qr_text",
                },
                {
                    "text": "256",
                    "reference": "x_offset",
                },
                {
                    "text": "256",
                    "reference": "y_offset",
                },
            ],
        }
        assert hordelib_instance is not None
        assert shared_model_manager.manager.controlnet is not None

        pil_image = hordelib_instance.basic_inference_single_image(data).image
        assert pil_image is not None
        assert isinstance(pil_image, Image.Image)

        img_filename = "qr_code.png"
        pil_image.save(f"images/{img_filename}", quality=100)

        assert check_single_inference_image_similarity(
            f"images_expected/{img_filename}",
            pil_image,
        )

    def test_qr_code_inference_out_of_bounds(
        self,
        shared_model_manager: type[SharedModelManager],
        hordelib_instance: HordeLib,
        stable_diffusion_model_name_for_testing: str,
    ):
        data = {
            "sampler_name": "k_euler",
            "cfg_scale": 7.5,
            "denoising_strength": 1.0,
            "seed": 1312,
            "height": 768,
            "width": 768,
            "karras": False,
            "tiling": False,
            "hires_fix": False,
            "clip_skip": 1,
            "prompt": (
                "drawing of two witches performing a seanse around a cauldron. Wispy and Ethereal, sepia colors"
                "###worst quality, bad lighting, deformed, ugly, low contrast"
            ),
            "ddim_steps": 30,
            "n_iter": 1,
            "model": stable_diffusion_model_name_for_testing,
            "workflow": "qr_code",
            "extra_texts": [
                {
                    "text": "https://aihorde.net",
                    "reference": "qr_text",
                },
                {
                    "text": "-256",
                    "reference": "x_offset",
                },
                {
                    "text": "800",
                    "reference": "y_offset",
                },
            ],
        }
        assert hordelib_instance is not None
        assert shared_model_manager.manager.controlnet is not None

        pil_image = hordelib_instance.basic_inference_single_image(data).image
        assert pil_image is not None
        assert isinstance(pil_image, Image.Image)

        img_filename = "qr_code_out_of_bounds.png"
        pil_image.save(f"images/{img_filename}", quality=100)

        assert check_single_inference_image_similarity(
            f"images_expected/{img_filename}",
            pil_image,
        )

    def test_qr_code_inference_xl(
        self,
        shared_model_manager: type[SharedModelManager],
        hordelib_instance: HordeLib,
        sdxl_refined_model_name: str,
    ):
        data = {
            "sampler_name": "k_euler",
            "cfg_scale": 7.5,
            "denoising_strength": 1.0,
            "seed": 1312,
            "height": 1024,
            "width": 1024,
            "karras": False,
            "tiling": False,
            "hires_fix": False,
            "clip_skip": 1,
            "prompt": (
                "Lucid Creations, a human brain, ethereal, dreamlike, surreal, "
                "beautiful, illustration, incredible detail, 8k, abstract"
                "###worst quality, bad lighting, deformed, ugly, low contrast"
            ),
            "control_strength": 1.2,
            "ddim_steps": 25,
            "n_iter": 1,
            "model": sdxl_refined_model_name,
            "workflow": "qr_code",
            "extra_texts": [
                {
                    "text": "https://aihorde.net",
                    "reference": "qr_text",
                },
            ],
        }
        assert hordelib_instance is not None
        assert shared_model_manager.manager.controlnet is not None

        pil_image = hordelib_instance.basic_inference_single_image(data).image
        assert pil_image is not None
        assert isinstance(pil_image, Image.Image)

        img_filename = "qr_code_xl.png"
        pil_image.save(f"images/{img_filename}", quality=100)

        assert check_single_inference_image_similarity(
            f"images_expected/{img_filename}",
            pil_image,
        )

    def test_qr_code_inference_too_large_text(
        self,
        shared_model_manager: type[SharedModelManager],
        hordelib_instance: HordeLib,
        sdxl_refined_model_name: str,
    ):
        data = {
            "sampler_name": "k_euler",
            "cfg_scale": 7.5,
            "denoising_strength": 1.0,
            "seed": 1111,
            "height": 1024,
            "width": 1024,
            "karras": False,
            "tiling": False,
            "hires_fix": False,
            "clip_skip": 1,
            "prompt": "The Scream, 1893 by Edvard Munch",
            "ddim_steps": 25,
            "n_iter": 1,
            "model": sdxl_refined_model_name,
            "workflow": "qr_code",
            "control_strength": 1.4,
            "extra_texts": [
                {
                    "text": """
Sint eos pariatur architecto repellat nihil sed distinctio aut.
Aut quae modi libero. Est ea nobis blanditiis in ut quam.
Blanditiis expedita minus tenetur dolorum.
Nisi recusandae blanditiis quia mollitia. Voluptatibus omnis non eos.
Aut voluptatum ut aspernatur minima omnis.
Quia officia est nisi exercitationem sint.
Adipisci mollitia sunt dignissimos fugiat sint magnam et sit. Cumque ipsum ullam et molestiae.
Consequatur saepe occaecati amet odio molestias odio est doloremque.
Sapiente reprehenderit qui adipisci officia delectus quas totam.
Maxime commodi rerum quod voluptas dolor ducimus. Non quibusdam ut assumenda ipsum voluptatem sit a necessitatibus.
Provident est culpa delectus nemo. Dolorem velit assumenda labore ut.
Voluptatum corporis modi id dolores necessitatibus voluptatibus at voluptate.
Rerum sed incidunt commodi quo quo. Sit sint accusantium modi eligendi molestiae maxime.
                """,
                    "reference": "qr_text",
                },
                {
                    "text": "48",
                    "reference": "x_offset",
                },
                {
                    "text": "48",
                    "reference": "y_offset",
                },
            ],
        }
        assert hordelib_instance is not None
        assert shared_model_manager.manager.controlnet is not None

        pil_image = hordelib_instance.basic_inference_single_image(data).image
        assert pil_image is not None
        assert isinstance(pil_image, Image.Image)

        img_filename = "qr_code_too_long_text.png"
        pil_image.save(f"images/{img_filename}", quality=100)

        assert check_single_inference_image_similarity(
            f"images_expected/{img_filename}",
            pil_image,
        )

    def test_qr_code_control_strength(
        self,
        shared_model_manager: type[SharedModelManager],
        hordelib_instance: HordeLib,
        sdxl_refined_model_name: str,
    ):
        data = {
            "sampler_name": "k_euler",
            "cfg_scale": 7.5,
            "denoising_strength": 1.0,
            "seed": 1312,
            "height": 1024,
            "width": 1024,
            "karras": False,
            "tiling": False,
            "hires_fix": False,
            "clip_skip": 1,
            "control_strength": 0.5,
            "prompt": (
                "Lucid Creations, a human brain, ethereal, dreamlike, surreal, "
                "beautiful, illustration, incredible detail, 8k, abstract"
                "###worst quality, bad lighting, deformed, ugly, low contrast"
            ),
            "ddim_steps": 25,
            "n_iter": 1,
            "model": sdxl_refined_model_name,
            "workflow": "qr_code",
            "extra_texts": [
                {
                    "text": "https://aihorde.net",
                    "reference": "qr_text",
                },
            ],
        }
        assert hordelib_instance is not None
        assert shared_model_manager.manager.controlnet is not None

        pil_image = hordelib_instance.basic_inference_single_image(data).image
        assert pil_image is not None
        assert isinstance(pil_image, Image.Image)

        img_filename = "qr_code_strength.png"
        pil_image.save(f"images/{img_filename}", quality=100)

        assert check_single_inference_image_similarity(
            f"images_expected/{img_filename}",
            pil_image,
        )

    def test_qr_code_control_non_square(
        self,
        shared_model_manager: type[SharedModelManager],
        hordelib_instance: HordeLib,
        sdxl_refined_model_name: str,
    ):
        data = {
            "sampler_name": "k_euler",
            "cfg_scale": 7.5,
            "denoising_strength": 1.0,
            "seed": 11312,
            "height": 1280,
            "width": 768,
            "karras": False,
            "tiling": False,
            "hires_fix": False,
            "clip_skip": 1,
            "prompt": (
                "Lucid Creations, a human brain, ethereal, dreamlike, surreal, "
                "beautiful, illustration, incredible detail, 8k, abstract"
                "###worst quality, bad lighting, deformed, ugly, low contrast"
            ),
            "control_strength": 1,
            "ddim_steps": 25,
            "n_iter": 1,
            "model": sdxl_refined_model_name,
            "workflow": "qr_code",
            "extra_texts": [
                {
                    "text": "https://aihorde.net",
                    "reference": "qr_text",
                },
            ],
        }
        assert hordelib_instance is not None
        assert shared_model_manager.manager.controlnet is not None

        pil_image = hordelib_instance.basic_inference_single_image(data).image
        assert pil_image is not None
        assert isinstance(pil_image, Image.Image)

        img_filename = "qr_code_size.png"
        pil_image.save(f"images/{img_filename}", quality=100)

        assert check_single_inference_image_similarity(
            f"images_expected/{img_filename}",
            pil_image,
        )

    def test_qr_code_control_qr_texts(
        self,
        shared_model_manager: type[SharedModelManager],
        hordelib_instance: HordeLib,
        sdxl_refined_model_name: str,
    ):
        data = {
            "sampler_name": "k_euler",
            "cfg_scale": 7.5,
            "denoising_strength": 1.0,
            "seed": 1312,
            "height": 1024,
            "width": 1024,
            "karras": False,
            "tiling": False,
            "hires_fix": False,
            "clip_skip": 1,
            "prompt": (
                "Lucid Creations, a human brain, ethereal, dreamlike, surreal, "
                "beautiful, illustration, incredible detail, 8k, abstract"
                "###worst quality, bad lighting, deformed, ugly, low contrast"
            ),
            "ddim_steps": 25,
            "n_iter": 1,
            "model": sdxl_refined_model_name,
            "workflow": "qr_code",
            "extra_texts": [
                {
                    "text": "haidra.net",
                    "reference": "qr_text",
                },
                {
                    "text": "Circle",
                    "reference": "module_drawer",
                },
                {
                    "text": "https",
                    "reference": "protocol",
                },
                {
                    "text": "Lucid Creations, ethereal brain cells",
                    "reference": "function_layer_prompt",
                },
                {
                    "text": "48",
                    "reference": "x_offset",
                },
                {
                    "text": "48",
                    "reference": "y_offset",
                },
            ],
        }
        assert hordelib_instance is not None
        assert shared_model_manager.manager.controlnet is not None

        pil_image = hordelib_instance.basic_inference_single_image(data).image
        assert pil_image is not None
        assert isinstance(pil_image, Image.Image)

        img_filename = "qr_code_texts.png"
        pil_image.save(f"images/{img_filename}", quality=100)

        assert check_single_inference_image_similarity(
            f"images_expected/{img_filename}",
            pil_image,
        )
