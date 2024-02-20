# test_horde.py

from loguru import logger
from PIL import Image

from hordelib.horde import HordeLib

from .testing_shared_functions import check_single_inference_image_similarity, check_single_lora_image_similarity

SLOW_SAMPLERS = ["k_dpmpp_2s_a", "k_heun", "k_dpm_2", "k_dpm_2_a"]  # "k_dpmpp_sde",


class TestHordeSamplers:
    def test_ddim_sampler(
        self,
        stable_diffusion_model_name_for_testing: str,
        hordelib_instance: HordeLib,
    ):
        data = {
            "sampler_name": "ddim",
            "cfg_scale": 6.5,
            "denoising_strength": 1.0,
            "seed": 3688490319,
            "height": 512,
            "width": 512,
            "karras": False,
            "tiling": False,
            "hires_fix": False,
            "clip_skip": 1,
            "control_type": None,
            "image_is_control": False,
            "return_control_map": False,
            "prompt": (
                "a woman closeup made out of metal, (cyborg:1.1), realistic skin, (detailed wire:1.3), "
                "(intricate details), hdr, (intricate details, hyperdetailed:1.2), cinematic shot, "
                "vignette, centered"
            ),
            "ddim_steps": 30,
            "n_iter": 1,
            "model": stable_diffusion_model_name_for_testing,
        }
        pil_image = hordelib_instance.basic_inference_single_image(data).image
        assert pil_image is not None

    def test_k_dpmpp_sde_sampler(
        self,
        stable_diffusion_model_name_for_testing: str,
        hordelib_instance: HordeLib,
    ):
        data = {
            "sampler_name": "k_dpmpp_sde",
            "cfg_scale": 6.5,
            "denoising_strength": 1.0,
            "seed": 3688490319,
            "height": 512,
            "width": 512,
            "karras": False,
            "tiling": False,
            "hires_fix": False,
            "clip_skip": 1,
            "control_type": None,
            "image_is_control": False,
            "return_control_map": False,
            "prompt": (
                "a woman closeup made out of metal, (cyborg:1.1), realistic skin, (detailed wire:1.3), "
                "(intricate details), hdr, (intricate details, hyperdetailed:1.2), cinematic shot, "
                "vignette, centered"
            ),
            "ddim_steps": 30,
            "n_iter": 1,
            "model": stable_diffusion_model_name_for_testing,
        }
        pil_image = hordelib_instance.basic_inference_single_image(data).image
        assert pil_image is not None

        img_filename = "sampler_30_steps_k_dpmpp_sde.png"
        pil_image.save(f"images/{img_filename}", quality=100)

        assert check_single_lora_image_similarity(
            f"images_expected/{img_filename}",
            pil_image,
        )

    def test_samplers(
        self,
        stable_diffusion_model_name_for_testing: str,
        hordelib_instance: HordeLib,
    ):
        data = {
            "sampler_name": "",
            "cfg_scale": 6.5,
            "denoising_strength": 1.0,
            "seed": 3688490319,
            "height": 512,
            "width": 512,
            "karras": False,
            "tiling": False,
            "hires_fix": False,
            "clip_skip": 1,
            "control_type": None,
            "image_is_control": False,
            "return_control_map": False,
            "prompt": (
                "a woman closeup made out of metal, (cyborg:1.1), realistic skin, (detailed wire:1.3), "
                "(intricate details), hdr, (intricate details, hyperdetailed:1.2), cinematic shot, "
                "vignette, centered"
            ),
            "ddim_steps": 30,
            "n_iter": 1,
            "model": stable_diffusion_model_name_for_testing,
        }
        images_to_compare: list[tuple[str, Image.Image]] = []
        for sampler in HordeLib.SAMPLERS_MAP.keys():
            data["sampler_name"] = sampler.upper()  # force uppercase to ensure case insensitive

            pil_image = hordelib_instance.basic_inference_single_image(data).image
            assert pil_image is not None

            img_filename = f"sampler_30_steps_{sampler}.png"
            pil_image.save(f"images/{img_filename}", quality=100)
            images_to_compare.append((f"images_expected/{img_filename}", pil_image))

        for img_filename, pil_image in images_to_compare:
            logger.debug(f"Checking image {img_filename}")
            if "sde" not in img_filename and "lcm" not in img_filename:
                assert check_single_inference_image_similarity(
                    img_filename,
                    pil_image,
                )
            else:
                logger.warning(
                    f"Skipping image similarity check for {img_filename} due to SDE samplers being non-deterministic.",
                )

    def test_slow_samplers(
        self,
        stable_diffusion_model_name_for_testing: str,
        hordelib_instance: HordeLib,
    ):
        data = {
            "sampler_name": "",
            "cfg_scale": 6.5,
            "denoising_strength": 1.0,
            "seed": 3688390309,
            "height": 512,
            "width": 512,
            "karras": False,
            "tiling": False,
            "hires_fix": False,
            "clip_skip": 1,
            "control_type": None,
            "image_is_control": False,
            "return_control_map": False,
            "prompt": (
                "an old man closeup, (hr giger:1.1), (detailed wire:1.3), "
                "(intricate details), hdr, (intricate details, hyperdetailed:1.2), cinematic shot, "
                "vignette, centered"
            ),
            "ddim_steps": 10,
            "n_iter": 1,
            "model": stable_diffusion_model_name_for_testing,
        }

        images_to_compare: list[tuple[str, Image.Image]] = []
        for sampler in SLOW_SAMPLERS:
            data["sampler_name"] = sampler

            pil_image = hordelib_instance.basic_inference_single_image(data).image
            assert pil_image is not None

            img_filename = f"sampler_10_steps_{sampler}.png"
            pil_image.save(f"images/{img_filename}", quality=100)
            images_to_compare.append((f"images_expected/{img_filename}", pil_image))

        for img_filename, pil_image in images_to_compare:
            assert check_single_inference_image_similarity(
                img_filename,
                pil_image,
            )
