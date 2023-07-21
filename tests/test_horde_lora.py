# test_horde_lora.py
import os
from datetime import datetime, timedelta

import pytest
from PIL import Image

from hordelib.horde import HordeLib
from hordelib.shared_model_manager import SharedModelManager

from .testing_shared_functions import check_single_lora_image_similarity


class TestHordeLora:
    @pytest.fixture(autouse=True, scope="class")
    def setup_and_teardown(self, shared_model_manager: type[SharedModelManager]):
        assert shared_model_manager.manager.lora
        shared_model_manager.manager.lora.download_default_loras()
        shared_model_manager.manager.lora.wait_for_downloads()
        yield
        shared_model_manager.manager.lora.stop_all()

    def test_text_to_image_lora_red(
        self,
        shared_model_manager: type[SharedModelManager],
        hordelib_instance: HordeLib,
        stable_diffusion_model_name_for_testing: str,
    ):
        assert shared_model_manager.manager.lora

        # Red
        lora_name = shared_model_manager.manager.lora.get_lora_name("GlowingRunesAI")
        assert isinstance(lora_name, str)
        trigger = shared_model_manager.manager.lora.find_lora_trigger(lora_name, "red")
        data = {
            "sampler_name": "k_euler",
            "cfg_scale": 8.0,
            "denoising_strength": 1.0,
            "seed": 304886399544324,
            "height": 512,
            "width": 512,
            "karras": False,
            "tiling": False,
            "hires_fix": False,
            "clip_skip": 1,
            "control_type": None,
            "image_is_control": False,
            "return_control_map": False,
            "prompt": f"a dark magical crystal, {trigger}, 8K resolution###blurry, out of focus",
            "loras": [{"name": lora_name, "model": 1.0, "clip": 1.0}],
            "ddim_steps": 20,
            "n_iter": 1,
            "model": stable_diffusion_model_name_for_testing,
        }
        pil_image = hordelib_instance.basic_inference(data)
        assert pil_image is not None
        assert isinstance(pil_image, Image.Image)

        img_filename = "lora_red.png"
        pil_image.save(f"images/{img_filename}", quality=100)

        assert check_single_lora_image_similarity(
            f"images_expected/{img_filename}",
            pil_image,
        )

        last_use = shared_model_manager.manager.lora.get_lora_last_use("GlowingRunesAI")
        assert last_use
        assert last_use > datetime.now() - timedelta(minutes=1)

    def test_text_to_image_lora_blue(
        self,
        shared_model_manager: type[SharedModelManager],
        hordelib_instance: HordeLib,
        stable_diffusion_model_name_for_testing: str,
    ):
        assert shared_model_manager.manager.lora

        # Blue, fuzzy search on version
        lora_name = shared_model_manager.manager.lora.get_lora_name("GlowingRunesAI")
        assert lora_name
        trigger = shared_model_manager.manager.lora.find_lora_trigger(lora_name, "blue")
        data = {
            "sampler_name": "k_euler",
            "cfg_scale": 8.0,
            "denoising_strength": 1.0,
            "seed": 304886399544324,
            "height": 512,
            "width": 512,
            "karras": False,
            "tiling": False,
            "hires_fix": False,
            "clip_skip": 1,
            "control_type": None,
            "image_is_control": False,
            "return_control_map": False,
            "prompt": f"a dark magical crystal, {trigger}, 8K resolution###blurry, out of focus",
            "loras": [{"name": lora_name, "model": 1.0, "clip": 1.0}],
            "ddim_steps": 20,
            "n_iter": 1,
            "model": stable_diffusion_model_name_for_testing,
        }

        pil_image = hordelib_instance.basic_inference(data)
        assert pil_image is not None
        assert isinstance(pil_image, Image.Image)

        img_filename = "lora_blue.png"
        pil_image.save(f"images/{img_filename}", quality=100)

        assert check_single_lora_image_similarity(
            f"images_expected/{img_filename}",
            pil_image,
        )

    def test_text_to_image_lora_chained(
        self,
        shared_model_manager: type[SharedModelManager],
        hordelib_instance: HordeLib,
        stable_diffusion_model_name_for_testing: str,
    ):
        assert shared_model_manager.manager.lora

        lora_name = shared_model_manager.manager.lora.get_lora_name("GlowingRunesAI")
        assert isinstance(lora_name, str)
        trigger = shared_model_manager.manager.lora.find_lora_trigger(lora_name, "red")
        trigger2 = shared_model_manager.manager.lora.find_lora_trigger(lora_name, "blue")
        lora_name2 = shared_model_manager.manager.lora.get_lora_name("Dra9onScaleAI")
        trigger3 = shared_model_manager.manager.lora.find_lora_trigger(lora_name, "Dr490nSc4leAI")

        data = {
            "sampler_name": "k_euler",
            "cfg_scale": 8.0,
            "denoising_strength": 1.0,
            "seed": 304886399544324,
            "height": 512,
            "width": 512,
            "karras": False,
            "tiling": False,
            "hires_fix": False,
            "clip_skip": 1,
            "control_type": None,
            "image_is_control": False,
            "return_control_map": False,
            "prompt": f"a dark magical crystal, {trigger}, {trigger2}, {trigger3}, "
            "8K resolution###glow, blurry, out of focus",
            "loras": [
                {"name": lora_name, "model": 1.0, "clip": 1.0},
                {"name": lora_name2, "model": 1.0, "clip": 1.0},
            ],
            "ddim_steps": 20,
            "n_iter": 1,
            "model": stable_diffusion_model_name_for_testing,
        }

        pil_image = hordelib_instance.basic_inference(data)
        assert pil_image is not None
        assert isinstance(pil_image, Image.Image)

        img_filename = "lora_multiple.png"
        pil_image.save(f"images/{img_filename}", quality=100)

        assert check_single_lora_image_similarity(
            f"images_expected/{img_filename}",
            pil_image,
        )

    def test_text_to_image_lora_chained_bad(
        self,
        shared_model_manager: type[SharedModelManager],
        hordelib_instance: HordeLib,
        stable_diffusion_model_name_for_testing: str,
    ):
        assert shared_model_manager.manager.lora

        lora_name = shared_model_manager.manager.lora.get_lora_name("GlowingRunesAI")
        assert isinstance(lora_name, str)
        trigger = shared_model_manager.manager.lora.find_lora_trigger(lora_name, "blue")
        lora_name2 = shared_model_manager.manager.lora.get_lora_name("Dra9onScaleAI")
        trigger2 = shared_model_manager.manager.lora.find_lora_trigger(lora_name, "Dr490nSc4leAI")
        data = {
            "sampler_name": "k_euler",
            "cfg_scale": 8.0,
            "denoising_strength": 1.0,
            "seed": 304886399544324,
            "height": 512,
            "width": 512,
            "karras": False,
            "tiling": False,
            "hires_fix": False,
            "clip_skip": 1,
            "control_type": None,
            "image_is_control": False,
            "return_control_map": False,
            "prompt": f"a dark magical crystal, {trigger}, {trigger2}, 8K resolution###blurry, out of focus",
            "loras": [
                {"name": lora_name, "model": 1.0, "clip": 1.0},
                {"name": lora_name2, "model": 1.0, "clip": 1.0},
                {"name": "__TotallyDoesNotExist__", "model": 1.0, "clip": 1.0},
            ],
            "ddim_steps": 20,
            "n_iter": 1,
            "model": stable_diffusion_model_name_for_testing,
        }

        pil_image = hordelib_instance.basic_inference(data)
        assert pil_image is not None
        # Don't save this one, just testing we didn't crash and burn

    def test_lora_trigger_inject_red(
        self,
        shared_model_manager: type[SharedModelManager],
        hordelib_instance: HordeLib,
        stable_diffusion_model_name_for_testing: str,
    ):
        assert shared_model_manager.manager.lora

        # Red
        lora_name = shared_model_manager.manager.lora.get_lora_name("GlowingRunesAI")
        data = {
            "sampler_name": "k_euler",
            "cfg_scale": 8.0,
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
            "prompt": "an obsidian magical monolith, dark background, 8K resolution###glow, blurry, out of focus",
            "loras": [{"name": lora_name, "model": 1.0, "clip": 1.0, "inject_trigger": "red"}],
            "ddim_steps": 20,
            "n_iter": 1,
            "model": stable_diffusion_model_name_for_testing,
        }

        pil_image = hordelib_instance.basic_inference(data)
        assert pil_image is not None
        assert isinstance(pil_image, Image.Image)

        img_filename = "lora_inject_red.png"
        pil_image.save(f"images/{img_filename}", quality=100)

        assert check_single_lora_image_similarity(
            f"images_expected/{img_filename}",
            pil_image,
        )

    def test_lora_trigger_inject_any(
        self,
        shared_model_manager: type[SharedModelManager],
        hordelib_instance: HordeLib,
        stable_diffusion_model_name_for_testing: str,
    ):
        assert shared_model_manager.manager.lora

        # Red
        lora_name = shared_model_manager.manager.lora.get_lora_name("GlowingRunesAI")
        data = {
            "sampler_name": "k_euler",
            "cfg_scale": 8.0,
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
            "prompt": "an obsidian magical monolith, dark background, 8K resolution###blurry, out of focus",
            "loras": [{"name": lora_name, "model": 1.0, "clip": 1.0, "inject_trigger": "any"}],
            "ddim_steps": 20,
            "n_iter": 1,
            "model": stable_diffusion_model_name_for_testing,
        }

        pil_image = hordelib_instance.basic_inference(data)
        assert pil_image is not None
        assert isinstance(pil_image, Image.Image)

        img_filename = "lora_inject_any.png"
        pil_image.save(f"images/{img_filename}", quality=100)

        assert check_single_lora_image_similarity(
            f"images_expected/{img_filename}",
            pil_image,
        )

    def test_download_and_use_adhoc_lora(
        self,
        shared_model_manager: type[SharedModelManager],
        hordelib_instance: HordeLib,
        stable_diffusion_model_name_for_testing: str,
    ):
        assert shared_model_manager.manager.lora

        lora_name = "74384"
        shared_model_manager.manager.lora.ensure_lora_deleted(lora_name)
        data = {
            "sampler_name": "k_euler",
            "cfg_scale": 8.0,
            "denoising_strength": 1.0,
            "seed": 1471413,
            "height": 512,
            "width": 512,
            "karras": False,
            "tiling": False,
            "hires_fix": False,
            "clip_skip": 1,
            "control_type": None,
            "image_is_control": False,
            "return_control_map": False,
            "prompt": "pantasa, plant, wooden robot, concept artist, ruins, night, moon, global "
            "illumination, depth of field, splash art",
            "loras": [{"name": lora_name, "model": 0.75, "clip": 1.0, "inject_trigger": "any"}],
            "ddim_steps": 20,
            "n_iter": 1,
            "model": stable_diffusion_model_name_for_testing,
        }

        pil_image = hordelib_instance.basic_inference(data)
        assert pil_image is not None
        assert isinstance(pil_image, Image.Image)

        img_filename = "lora_download_adhoc.png"
        pil_image.save(f"images/{img_filename}", quality=100)
        assert check_single_lora_image_similarity(
            f"images_expected/{img_filename}",
            pil_image,
        )

    def test_for_probability_tensor_runtime_error(
        self,
        hordelib_instance: HordeLib,
        stable_diffusion_model_name_for_testing: str,
    ):
        data = {
            "sampler_name": "k_euler",
            "cfg_scale": 8.0,
            "denoising_strength": 1.0,
            "seed": 1471413,
            "height": 512,
            "width": 512,
            "karras": False,
            "tiling": False,
            "hires_fix": False,
            "clip_skip": 1,
            "control_type": None,
            "image_is_control": False,
            "return_control_map": False,
            "prompt": "pantasa, plant, wooden robot, concept artist, ruins, night, moon, global "
            "illumination, depth of field, splash art",
            "loras": [
                {"name": "48139", "model": 0.75, "clip": 1.0},
                {"name": "58390", "model": 1, "clip": 1.0},
                {"name": "13941", "model": 1, "clip": 1.0},
            ],
            "ddim_steps": 20,
            "n_iter": 1,
            "model": stable_diffusion_model_name_for_testing,
        }

        pil_image = hordelib_instance.basic_inference(data)
        assert pil_image is not None

    def test_sd21_lora_against_sd15_model(
        self,
        hordelib_instance: HordeLib,
        stable_diffusion_model_name_for_testing: str,
    ):
        data = {
            "sampler_name": "k_euler",
            "cfg_scale": 8.0,
            "denoising_strength": 1.0,
            "seed": 0,
            "height": 512,
            "width": 512,
            "karras": False,
            "tiling": False,
            "hires_fix": False,
            "clip_skip": 1,
            "control_type": None,
            "image_is_control": False,
            "return_control_map": False,
            "prompt": "pantasa, plant, wooden robot, concept artist, ruins, night, moon, global "
            "illumination, depth of field, splash art",
            "loras": [
                {"name": "35822", "model": 1, "clip": 1.0},
            ],
            "ddim_steps": 20,
            "n_iter": 1,
            "model": stable_diffusion_model_name_for_testing,
        }

        pil_image = hordelib_instance.basic_inference(data)
        assert pil_image is not None

    def test_stonepunk(
        self,
        hordelib_instance: HordeLib,
        stable_diffusion_model_name_for_testing: str,
    ):
        # Blue, fuzzy search on version
        lora_name = "51539"
        data = {
            "sampler_name": "k_euler",
            "cfg_scale": 8.0,
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
            "prompt": "An automobile, stonepunkAI",
            "loras": [{"name": lora_name, "model": 1.0, "clip": 1.0, "inject_trigger": "any"}],
            "ddim_steps": 20,
            "n_iter": 1,
            "model": stable_diffusion_model_name_for_testing,
        }

        pil_image = hordelib_instance.basic_inference(data)
        assert pil_image is not None
        assert isinstance(pil_image, Image.Image)

        img_filename = "lora_stonepunk.png"
        pil_image.save(f"images/{img_filename}", quality=100)

        assert check_single_lora_image_similarity(
            f"images_expected/{img_filename}",
            pil_image,
        )

    def test_negative_model_power(
        self,
        hordelib_instance: HordeLib,
        stable_diffusion_model_name_for_testing: str,
    ):
        lora_name = "58390"
        data = {
            "sampler_name": "k_euler",
            "cfg_scale": 8.0,
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
            "prompt": "A girl walking in a field of flowers",
            "loras": [{"name": lora_name, "model": -2.0, "clip": 1.0}],
            "ddim_steps": 20,
            "n_iter": 1,
            "model": stable_diffusion_model_name_for_testing,
        }

        pil_image = hordelib_instance.basic_inference(data)
        assert pil_image is not None
        assert isinstance(pil_image, Image.Image)

        img_filename = "lora_negative_strength.png"
        pil_image.save(f"images/{img_filename}", quality=100)

        data = {
            "sampler_name": "k_euler",
            "cfg_scale": 8.0,
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
            "prompt": "A girl walking in a field of flowers",
            "loras": [{"name": lora_name, "model": 2.0, "clip": 1.0}],
            "ddim_steps": 20,
            "n_iter": 1,
            "model": stable_diffusion_model_name_for_testing,
        }

        pil_image = hordelib_instance.basic_inference(data)
        assert pil_image is not None
        assert isinstance(pil_image, Image.Image)

        img_filename = "lora_positive_strength.png"
        pil_image.save(f"images/{img_filename}", quality=100)

        # assert check_single_lora_image_similarity(
        #     f"images_expected/{img_filename}",
        #     pil_image,
        # )
