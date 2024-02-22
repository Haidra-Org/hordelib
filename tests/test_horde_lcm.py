# test_horde_lcm.py
import pytest
from PIL import Image

from hordelib.horde import HordeLib
from hordelib.shared_model_manager import SharedModelManager

from .testing_shared_functions import check_single_lora_image_similarity


class TestHordeLCM:
    @pytest.fixture(autouse=True, scope="class")
    def setup_and_teardown(self, shared_model_manager: type[SharedModelManager]):
        assert shared_model_manager.manager.lora
        shared_model_manager.manager.lora.download_default_loras()
        shared_model_manager.manager.lora.wait_for_downloads()
        yield

    def test_use_lcm_turbomix_lora_euler_a(
        self,
        shared_model_manager: type[SharedModelManager],
        hordelib_instance: HordeLib,
        stable_diffusion_model_name_for_testing: str,
    ):
        assert shared_model_manager.manager.lora

        lora_name = "246747"  # Euler A Turbo
        data = {
            "sampler_name": "euler_a",
            "cfg_scale": 1,
            "denoising_strength": 1.0,
            "seed": 1312,
            "height": 1024,
            "width": 1024,
            "karras": False,
            "tiling": False,
            "hires_fix": False,
            "clip_skip": 2,
            "control_type": None,
            "image_is_control": False,
            "return_control_map": False,
            "prompt": (
                "portrait of a (Namibian:1.2) doll with black skin, deep fine cracks,"
                "(kintsugi:1.2), soft focus,half body editorial shot,depth of field,uncanny,"
                "long white hair,stunning perfect shining eyes, shy smile,"
                "intricitaly hyperdetailed,amazing depth,expansive details, cracked surface,"
                "on display, iridescent surface,Anna Dittmann,Dominic Qwek,complex masterwork"
            ),
            "loras": [{"name": lora_name, "model": 1.0, "clip": 1.0, "inject_trigger": "any", "is_version": True}],
            "ddim_steps": 5,
            "n_iter": 1,
            "model": "AlbedoBase XL (SDXL)",
        }

        pil_image = hordelib_instance.basic_inference_single_image(data).image
        assert pil_image is not None
        assert isinstance(pil_image, Image.Image)

        img_filename = "lcm_lora_turbomix_euler_a.png"
        pil_image.save(f"images/{img_filename}", quality=100)
        assert check_single_lora_image_similarity(
            f"images_expected/{img_filename}",
            pil_image,
        )

    def test_use_lcm_turbomix_lora_lcm(
        self,
        shared_model_manager: type[SharedModelManager],
        hordelib_instance: HordeLib,
        stable_diffusion_model_name_for_testing: str,
    ):
        assert shared_model_manager.manager.lora

        lora_name = "268475"  # LCM TUrbo
        data = {
            "sampler_name": "lcm",
            "cfg_scale": 1.5,
            "denoising_strength": 1.0,
            "seed": 1312,
            "height": 1024,
            "width": 1024,
            "karras": False,
            "tiling": False,
            "hires_fix": False,
            "clip_skip": 2,
            "control_type": None,
            "image_is_control": False,
            "return_control_map": False,
            "prompt": (
                "portrait of a (Namibian:1.2) doll with black skin, deep fine cracks,"
                "(kintsugi:1.2), soft focus,half body editorial shot,depth of field,uncanny,"
                "long white hair,stunning perfect shining eyes, shy smile,"
                "intricitaly hyperdetailed,amazing depth,expansive details, cracked surface,"
                "on display, iridescent surface,Anna Dittmann,Dominic Qwek,complex masterwork"
            ),
            "loras": [{"name": lora_name, "model": 1.0, "clip": 1.0, "inject_trigger": "any", "is_version": True}],
            "ddim_steps": 4,
            "n_iter": 1,
            "model": "AlbedoBase XL (SDXL)",
        }

        pil_image = hordelib_instance.basic_inference_single_image(data).image
        assert pil_image is not None
        assert isinstance(pil_image, Image.Image)

        img_filename = "lcm_lora_turbomix_lcm.png"
        pil_image.save(f"images/{img_filename}", quality=100)
        assert check_single_lora_image_similarity(
            f"images_expected/{img_filename}",
            pil_image,
        )

    def test_use_lcm_turbomix_lora_dpmpp_sde(
        self,
        shared_model_manager: type[SharedModelManager],
        hordelib_instance: HordeLib,
        stable_diffusion_model_name_for_testing: str,
    ):
        assert shared_model_manager.manager.lora

        lora_name = "247778"  # DPMPP SDE
        data = {
            "sampler_name": "dpmpp_sde",
            "cfg_scale": 2,
            "denoising_strength": 1.0,
            "seed": 1312,
            "height": 1024,
            "width": 1024,
            "karras": True,
            "tiling": False,
            "hires_fix": False,
            "clip_skip": 2,
            "control_type": None,
            "image_is_control": False,
            "return_control_map": False,
            "prompt": (
                "portrait of a (Namibian:1.2) doll with black skin, deep fine cracks,"
                "(kintsugi:1.2), soft focus,half body editorial shot,depth of field,uncanny,"
                "long white hair,stunning perfect shining eyes, shy smile,"
                "intricitaly hyperdetailed,amazing depth,expansive details, cracked surface,"
                "on display, iridescent surface,Anna Dittmann,Dominic Qwek,complex masterwork"
            ),
            "loras": [{"name": lora_name, "model": 1.0, "clip": 1.0, "inject_trigger": "any", "is_version": True}],
            "ddim_steps": 10,
            "n_iter": 1,
            "model": "AlbedoBase XL (SDXL)",
        }

        pil_image = hordelib_instance.basic_inference_single_image(data).image
        assert pil_image is not None
        assert isinstance(pil_image, Image.Image)

        img_filename = "lcm_lora_turbomix_dpmpp_sde.png"
        pil_image.save(f"images/{img_filename}", quality=100)
        assert check_single_lora_image_similarity(
            f"images_expected/{img_filename}",
            pil_image,
        )

    def test_use_sd1_5_lora_lcm(
        self,
        shared_model_manager: type[SharedModelManager],
        hordelib_instance: HordeLib,
        stable_diffusion_model_name_for_testing: str,
    ):
        assert shared_model_manager.manager.lora

        lora_name = "225222"  # LCM for SD 1.5
        data = {
            "sampler_name": "lcm",
            "cfg_scale": 1.5,
            "denoising_strength": 1.0,
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
            "prompt": (
                "portrait of a (Namibian:1.2) doll with black skin, deep fine cracks,"
                "(kintsugi:1.2), soft focus,half body editorial shot,depth of field,uncanny,"
                "long white hair,stunning perfect shining eyes, shy smile,"
                "intricitaly hyperdetailed,amazing depth,expansive details, cracked surface,"
                "on display, iridescent surface,Anna Dittmann,Dominic Qwek,complex masterwork"
            ),
            "loras": [{"name": lora_name, "model": 1.0, "clip": 0.75, "inject_trigger": "any", "is_version": True}],
            "ddim_steps": 3,
            "n_iter": 1,
            "model": stable_diffusion_model_name_for_testing,
        }

        pil_image = hordelib_instance.basic_inference_single_image(data).image
        assert pil_image is not None
        assert isinstance(pil_image, Image.Image)

        img_filename = "lcm_lora_lcm_1_5.png"
        pil_image.save(f"images/{img_filename}", quality=100)
        assert check_single_lora_image_similarity(
            f"images_expected/{img_filename}",
            pil_image,
        )
