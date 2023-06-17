# test_horde.py
import os

import pytest
from PIL import Image

from hordelib.horde import HordeLib
from hordelib.shared_model_manager import SharedModelManager

from .testing_shared_functions import check_inference_image_similarity_pytest


class TestHordeInference:
    @pytest.fixture(autouse=True)
    def setup_and_teardown(self, shared_model_manager: type[SharedModelManager]):
        for preproc in HordeLib.CONTROLNET_IMAGE_PREPROCESSOR_MAP.keys():
            shared_model_manager.manager.controlnet.download_control_type(preproc, ["stable diffusion 1"])

    def test_controlnet_sd1(
        self,
        shared_model_manager: type[SharedModelManager],
        hordelib_instance: HordeLib,
        stable_diffusion_modelname_for_testing: str,
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
            "control_type": "",
            "image_is_control": False,
            "return_control_map": False,
            "prompt": "a man walking in the snow",
            "ddim_steps": 25,
            "n_iter": 1,
            "model": stable_diffusion_modelname_for_testing,
            "source_image": Image.open("images/test_db0.jpg"),
            "source_processing": "img2img",
        }
        assert hordelib_instance is not None
        for preproc in HordeLib.CONTROLNET_IMAGE_PREPROCESSOR_MAP.keys():
            if preproc == "scribble" or preproc == "mlsd":
                # Skip
                continue
            assert (
                shared_model_manager.manager.controlnet.check_control_type_available(
                    preproc,
                    "stable diffusion 1",
                )
                is True
            )
            data["control_type"] = preproc

            pil_image = hordelib_instance.basic_inference(data)
            assert pil_image is not None

            img_filename = f"controlnet_{preproc}.png"

            pil_image.save(f"images/{img_filename}", quality=100)
            assert check_inference_image_similarity_pytest(
                f"images_expected/{img_filename}",
                pil_image,
            )

    def test_controlnet_fake_cn(
        self,
        hordelib_instance: HordeLib,
        stable_diffusion_modelname_for_testing: str,
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
            "control_type": "THIS_SHOULD_FAIL",
            "image_is_control": False,
            "return_control_map": False,
            "prompt": "a man walking in the snow",
            "ddim_steps": 25,
            "n_iter": 1,
            "model": stable_diffusion_modelname_for_testing,
            "source_image": Image.open("images/test_db0.jpg"),
            "source_processing": "img2img",
        }
        assert hordelib_instance is not None
        with pytest.raises(Exception):
            hordelib_instance.basic_inference(data)

    def test_controlnet_strength(
        self,
        hordelib_instance: HordeLib,
        stable_diffusion_modelname_for_testing: str,
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
            "control_type": "canny",
            "image_is_control": False,
            "return_control_map": False,
            "prompt": "a man walking on the moon",
            "ddim_steps": 25,
            "n_iter": 1,
            "model": stable_diffusion_modelname_for_testing,
            "source_image": Image.open("images/test_db0.jpg"),
            "source_processing": "img2img",
        }
        for strength in [1.0, 0.5, 0.2]:
            data["control_strength"] = strength

            pil_image = hordelib_instance.basic_inference(data)
            assert pil_image is not None

            img_filename = f"controlnet_strength_{strength}.png"

            pil_image.save(f"images/{img_filename}", quality=100)
            assert check_inference_image_similarity_pytest(
                f"images_expected/{img_filename}",
                pil_image,
            )

    def test_controlnet_hires_fix(
        self,
        hordelib_instance: HordeLib,
        stable_diffusion_modelname_for_testing: str,
    ):
        data = {
            "sampler_name": "k_dpmpp_2m",
            "cfg_scale": 7.5,
            "denoising_strength": 1.0,
            "seed": 1234345378856789,
            "height": 768,
            "width": 768,
            "karras": False,
            "tiling": False,
            "hires_fix": True,
            "hires_fix_denoising_strength": 0.0,
            "clip_skip": 1,
            "control_type": "canny",
            "image_is_control": False,
            "return_control_map": False,
            "prompt": "a man walking in the jungle",
            "ddim_steps": 15,
            "n_iter": 1,
            "model": stable_diffusion_modelname_for_testing,
            "source_image": Image.open("images/test_db0.jpg"),
            "source_processing": "img2img",
        }
        for denoise in [0.4, 0.5, 0.6]:
            data["hires_fix_denoising_strength"] = denoise

            pil_image = hordelib_instance.basic_inference(data)
            assert pil_image is not None

            img_filename = f"controlnet_hires_fix_denoise_{denoise}.png"

            pil_image.save(f"images/{img_filename}", quality=100)

            assert check_inference_image_similarity_pytest(
                f"images_expected/{img_filename}",
                pil_image,
            )
