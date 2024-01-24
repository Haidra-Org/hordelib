# test_horde_lora.py
from datetime import datetime, timedelta

import pytest
from PIL import Image

from hordelib.horde import HordeLib, ResultingImageReturn
from hordelib.shared_model_manager import SharedModelManager

from .testing_shared_functions import check_single_lora_image_similarity


class TestHordeLora:
    @pytest.fixture(autouse=True, scope="class")
    def setup_and_teardown(self, shared_model_manager: type[SharedModelManager]):
        assert shared_model_manager.manager.lora
        shared_model_manager.manager.lora.download_default_loras()
        shared_model_manager.manager.lora.wait_for_downloads()
        yield

    def test_text_to_image_lora_red(
        self,
        shared_model_manager: type[SharedModelManager],
        hordelib_instance: HordeLib,
        stable_diffusion_model_name_for_testing: str,
        lora_GlowingRunesAI: str,
    ):
        assert shared_model_manager.manager.lora

        # Red
        trigger = shared_model_manager.manager.lora.find_lora_trigger(lora_GlowingRunesAI, "red")
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
            "loras": [{"name": lora_GlowingRunesAI, "model": 1.0, "clip": 1.0}],
            "ddim_steps": 20,
            "n_iter": 1,
            "model": stable_diffusion_model_name_for_testing,
        }
        ret = hordelib_instance.basic_inference_single_image(data)
        assert isinstance(ret, ResultingImageReturn)
        pil_image = ret.image
        assert pil_image is not None
        assert isinstance(pil_image, Image.Image)
        assert len(ret.faults) == 0

        img_filename = "lora_red.png"
        pil_image.save(f"images/{img_filename}", quality=100)

        assert check_single_lora_image_similarity(
            f"images_expected/{img_filename}",
            pil_image,
        )

        last_use = shared_model_manager.manager.lora.get_lora_last_use(lora_GlowingRunesAI)
        assert last_use
        if not (last_use > datetime.now() - timedelta(minutes=1)):
            raise Exception("Last use of lora was not updated")

    def test_text_to_image_lora_blue(
        self,
        shared_model_manager: type[SharedModelManager],
        hordelib_instance: HordeLib,
        stable_diffusion_model_name_for_testing: str,
        lora_GlowingRunesAI: str,
    ):
        assert shared_model_manager.manager.lora

        # Blue, fuzzy search on version
        trigger = shared_model_manager.manager.lora.find_lora_trigger(lora_GlowingRunesAI, "blue")
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
            "loras": [{"name": lora_GlowingRunesAI, "model": 1.0, "clip": 1.0}],
            "ddim_steps": 20,
            "n_iter": 1,
            "model": stable_diffusion_model_name_for_testing,
        }

        pil_image = hordelib_instance.basic_inference_single_image(data).image
        assert pil_image is not None
        assert isinstance(pil_image, Image.Image)

        img_filename = "lora_blue.png"
        pil_image.save(f"images/{img_filename}", quality=100)

        assert check_single_lora_image_similarity(
            f"images_expected/{img_filename}",
            pil_image,
        )

    def test_text_to_image_lora_blue_tiled(
        self,
        shared_model_manager: type[SharedModelManager],
        hordelib_instance: HordeLib,
        stable_diffusion_model_name_for_testing: str,
        lora_GlowingRunesAI: str,
    ):
        assert shared_model_manager.manager.lora

        # Blue, fuzzy search on version
        trigger = shared_model_manager.manager.lora.find_lora_trigger(lora_GlowingRunesAI, "blue")
        data = {
            "sampler_name": "k_euler",
            "cfg_scale": 3.0,
            "denoising_strength": 1.0,
            "seed": 304886399544324,
            "height": 512,
            "width": 512,
            "karras": False,
            "tiling": True,
            "hires_fix": False,
            "clip_skip": 1,
            "control_type": None,
            "image_is_control": False,
            "return_control_map": False,
            "prompt": f"a dark magical crystal, {trigger}, 8K resolution###blurry, out of focus",
            "loras": [{"name": lora_GlowingRunesAI, "model": 1.0, "clip": 1.0}],
            "ddim_steps": 20,
            "n_iter": 1,
            "model": stable_diffusion_model_name_for_testing,
        }

        pil_image = hordelib_instance.basic_inference_single_image(data).image
        assert pil_image is not None
        assert isinstance(pil_image, Image.Image)

        img_filename = "lora_blue_tiled.png"
        pil_image.save(f"images/{img_filename}", quality=100)

        assert check_single_lora_image_similarity(
            f"images_expected/{img_filename}",
            pil_image,
        )

    def test_text_to_image_lora_blue_weighted(
        self,
        shared_model_manager: type[SharedModelManager],
        hordelib_instance: HordeLib,
        stable_diffusion_model_name_for_testing: str,
        lora_GlowingRunesAI: str,
    ):
        assert shared_model_manager.manager.lora

        # Blue, fuzzy search on version
        trigger = shared_model_manager.manager.lora.find_lora_trigger(lora_GlowingRunesAI, "blue")
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
            "prompt": f"a dark magical crystal, ({trigger}:1.2), 8K resolution###blurry, out of focus",
            "loras": [{"name": lora_GlowingRunesAI, "model": 1, "clip": 1.0}],
            "ddim_steps": 20,
            "n_iter": 1,
            "model": stable_diffusion_model_name_for_testing,
        }

        pil_image = hordelib_instance.basic_inference_single_image(data).image
        assert pil_image is not None
        assert isinstance(pil_image, Image.Image)

        img_filename = "lora_blue_weighted.png"
        pil_image.save(f"images/{img_filename}", quality=100)

        assert check_single_lora_image_similarity(
            f"images_expected/{img_filename}",
            pil_image,
        )

    def test_text_to_image_lora_blue_low_strength(
        self,
        shared_model_manager: type[SharedModelManager],
        hordelib_instance: HordeLib,
        stable_diffusion_model_name_for_testing: str,
        lora_GlowingRunesAI: str,
    ):
        assert shared_model_manager.manager.lora

        # Blue, fuzzy search on version
        trigger = shared_model_manager.manager.lora.find_lora_trigger(lora_GlowingRunesAI, "blue")
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
            "loras": [{"name": lora_GlowingRunesAI, "model": 0.1, "clip": 1.0}],
            "ddim_steps": 20,
            "n_iter": 1,
            "model": stable_diffusion_model_name_for_testing,
        }

        pil_image = hordelib_instance.basic_inference_single_image(data).image
        assert pil_image is not None
        assert isinstance(pil_image, Image.Image)

        img_filename = "lora_blue_low_model_strength.png"
        pil_image.save(f"images/{img_filename}", quality=100)
        assert check_single_lora_image_similarity(
            f"images_expected/{img_filename}",
            pil_image,
        )

        data["loras"] = [{"name": lora_GlowingRunesAI, "model": 1.0, "clip": 0.1}]
        pil_image = hordelib_instance.basic_inference_single_image(data).image
        assert pil_image is not None
        assert isinstance(pil_image, Image.Image)

        img_filename = "lora_blue_low_clip_strength.png"
        pil_image.save(f"images/{img_filename}", quality=100)
        assert check_single_lora_image_similarity(
            f"images_expected/{img_filename}",
            pil_image,
        )

        data["loras"] = [{"name": lora_GlowingRunesAI, "model": 0.1, "clip": 0.1}]
        pil_image = hordelib_instance.basic_inference_single_image(data).image
        assert pil_image is not None
        assert isinstance(pil_image, Image.Image)

        img_filename = "lora_blue_low_model_and_clip_strength.png"
        pil_image.save(f"images/{img_filename}", quality=100)
        assert check_single_lora_image_similarity(
            f"images_expected/{img_filename}",
            pil_image,
        )

    def test_text_to_image_lora_blue_negative_strength(
        self,
        shared_model_manager: type[SharedModelManager],
        hordelib_instance: HordeLib,
        stable_diffusion_model_name_for_testing: str,
        lora_GlowingRunesAI: str,
    ):
        assert shared_model_manager.manager.lora

        # Blue, fuzzy search on version
        trigger = shared_model_manager.manager.lora.find_lora_trigger(lora_GlowingRunesAI, "blue")
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
            "loras": [{"name": lora_GlowingRunesAI, "model": -1, "clip": 1.0}],
            "ddim_steps": 20,
            "n_iter": 1,
            "model": stable_diffusion_model_name_for_testing,
        }

        pil_image = hordelib_instance.basic_inference_single_image(data).image
        assert pil_image is not None
        assert isinstance(pil_image, Image.Image)

        img_filename = "lora_blue_negative_model_strength.png"
        pil_image.save(f"images/{img_filename}", quality=100)
        assert check_single_lora_image_similarity(
            f"images_expected/{img_filename}",
            pil_image,
        )

        data["loras"] = [{"name": lora_GlowingRunesAI, "model": 1.0, "clip": -1.0}]
        pil_image = hordelib_instance.basic_inference_single_image(data).image
        assert pil_image is not None
        assert isinstance(pil_image, Image.Image)

        img_filename = "lora_blue_negative_clip_strength.png"
        pil_image.save(f"images/{img_filename}", quality=100)
        assert check_single_lora_image_similarity(
            f"images_expected/{img_filename}",
            pil_image,
        )

        data["loras"] = [{"name": lora_GlowingRunesAI, "model": -1.0, "clip": -1.0}]
        pil_image = hordelib_instance.basic_inference_single_image(data).image
        assert pil_image is not None
        assert isinstance(pil_image, Image.Image)

        img_filename = "lora_blue_negative_model_and_clip_strength.png"
        pil_image.save(f"images/{img_filename}", quality=100)
        assert check_single_lora_image_similarity(
            f"images_expected/{img_filename}",
            pil_image,
        )

    def test_text_to_image_lora_blue_hires_fix(
        self,
        shared_model_manager: type[SharedModelManager],
        hordelib_instance: HordeLib,
        stable_diffusion_model_name_for_testing: str,
        lora_GlowingRunesAI: str,
    ):
        assert shared_model_manager.manager.lora

        # Blue, fuzzy search on version
        trigger = shared_model_manager.manager.lora.find_lora_trigger(lora_GlowingRunesAI, "blue")
        data = {
            "sampler_name": "k_euler",
            "cfg_scale": 8.0,
            "denoising_strength": 1.0,
            "seed": 304886399544324,
            "height": 1024,
            "width": 1024,
            "karras": False,
            "tiling": False,
            "hires_fix": True,
            "clip_skip": 1,
            "control_type": None,
            "image_is_control": False,
            "return_control_map": False,
            "prompt": f"a dark magical crystal, {trigger}, 8K resolution###blurry, out of focus",
            "loras": [{"name": lora_GlowingRunesAI, "model": 1.0, "clip": 1.0}],
            "ddim_steps": 20,
            "n_iter": 1,
            "model": stable_diffusion_model_name_for_testing,
        }

        pil_image = hordelib_instance.basic_inference_single_image(data).image
        assert pil_image is not None
        assert isinstance(pil_image, Image.Image)

        img_filename = "lora_blue_hires_fix.png"
        pil_image.save(f"images/{img_filename}", quality=100)

        assert check_single_lora_image_similarity(
            f"images_expected/{img_filename}",
            pil_image,
        )

    def test_text_to_image_lora_character_hires_fix(
        self,
        shared_model_manager: type[SharedModelManager],
        hordelib_instance: HordeLib,
    ):
        assert shared_model_manager.manager.lora

        shared_model_manager.manager.lora.fetch_adhoc_lora("82098")
        lora_name_1 = shared_model_manager.manager.lora.get_lora_name("82098")
        shared_model_manager.manager.lora.fetch_adhoc_lora("56586")
        lora_name_2 = shared_model_manager.manager.lora.get_lora_name("56586")
        assert lora_name_1
        assert lora_name_2

        assert shared_model_manager.manager.compvis

        data = {
            "sampler_name": "k_dpmpp_sde",
            "cfg_scale": 7,
            "denoising_strength": 1.0,
            "seed": 3238200406,
            "height": 1536,
            "width": 1024,
            "karras": True,
            "tiling": False,
            "hires_fix": True,
            "clip_skip": 2,
            "control_type": None,
            "image_is_control": False,
            "return_control_map": False,
            "prompt": (
                "masterpiece, highest quality, RAW, analog style, A stunning portrait of a beautiful woman, pink hair,"
                " pale skin, vibrant blue eyes, wearing black and red armor, red cape, (highly detailed skin, skin"
                " details), sharp focus, 8k UHD, DSLR, high quality, film grain, Fujifilm XT3, frowning, intricate"
                " details, highly detailed, cluttered and detailed background ### freckles, cat ears, large breasts,"
                " straight hair, deformed eyes, close up, ((disfigured)), ((bad art)), ((deformed)), ((extra limbs)),"
                " (((duplicate))), ((morbid)), ((mutilated)), out of frame, extra fingers, mutated hands, poorly drawn"
                " eyes, ((poorly drawn hands)), ((poorly drawn face)), (((mutation))), blurry, ((bad anatomy)), (((bad"
                " proportions))), cloned face, body out of frame, out of frame, bad anatomy, gross proportions,"
                " (malformed limbs), ((missing arms)), ((missing legs)), (((extra arms))), (((extra legs))), (fused"
                " fingers), (too many fingers), (((long neck))), tiling, poorly drawn, mutated, cross-eye, canvas"
                " frame, frame, cartoon, 3d, weird colors, blurry, watermark, trademark, logo"
            ),
            "loras": [
                {"name": lora_name_1, "model": 0.7, "clip": 1.0},
                {"name": lora_name_2, "model": 0.67, "clip": 1.0},
            ],
            "ddim_steps": 30,
            "n_iter": 1,
            "model": "Rev Animated",
        }

        pil_image = hordelib_instance.basic_inference_single_image(data).image
        assert pil_image is not None
        assert isinstance(pil_image, Image.Image)

        img_filename = "lora_character_hires_fix.png"
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
        lora_GlowingRunesAI: str,
    ):
        assert shared_model_manager.manager.lora

        trigger = shared_model_manager.manager.lora.find_lora_trigger(lora_GlowingRunesAI, "red")
        trigger2 = shared_model_manager.manager.lora.find_lora_trigger(lora_GlowingRunesAI, "blue")
        lora_name2 = shared_model_manager.manager.lora.fetch_adhoc_lora("Dra9onScaleAI")
        trigger3 = shared_model_manager.manager.lora.find_lora_trigger(lora_GlowingRunesAI, "Dr490nSc4leAI")

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
            "prompt": (
                f"a dark magical crystal, {trigger}, {trigger2}, {trigger3}, "
                "8K resolution###glow, blurry, out of focus"
            ),
            "loras": [
                {"name": lora_GlowingRunesAI, "model": 1.0, "clip": 1.0},
                {"name": lora_name2, "model": 1.0, "clip": 1.0},
            ],
            "ddim_steps": 20,
            "n_iter": 1,
            "model": stable_diffusion_model_name_for_testing,
        }

        pil_image = hordelib_instance.basic_inference_single_image(data).image
        assert pil_image is not None
        assert isinstance(pil_image, Image.Image)
        img_filename = "lora_multiple.png"
        pil_image.save(f"images/{img_filename}", quality=100)

        assert check_single_lora_image_similarity(
            f"images_expected/{img_filename}",
            pil_image,
        )

        data["loras"] = [
            {"name": lora_name2, "model": 1.0, "clip": 1.0},
            {"name": lora_GlowingRunesAI, "model": 1.0, "clip": 1.0},
        ]
        pil_image = hordelib_instance.basic_inference_single_image(data).image
        assert pil_image is not None
        assert isinstance(pil_image, Image.Image)
        img_filename_2 = "lora_multiple_reordered.png"
        pil_image.save(f"images/{img_filename_2}", quality=100)
        assert check_single_lora_image_similarity(
            f"images_expected/{img_filename}",
            pil_image,
        )

    def test_text_to_image_lora_chained_bad(
        self,
        shared_model_manager: type[SharedModelManager],
        hordelib_instance: HordeLib,
        stable_diffusion_model_name_for_testing: str,
        lora_GlowingRunesAI: str,
    ):
        assert shared_model_manager.manager.lora

        trigger = shared_model_manager.manager.lora.find_lora_trigger(lora_GlowingRunesAI, "blue")
        lora_name2 = shared_model_manager.manager.lora.fetch_adhoc_lora("Dra9onScaleAI")
        trigger2 = shared_model_manager.manager.lora.find_lora_trigger(lora_GlowingRunesAI, "Dr490nSc4leAI")
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
                {"name": lora_GlowingRunesAI, "model": 1.0, "clip": 1.0},
                {"name": lora_name2, "model": 1.0, "clip": 1.0},
                {"name": "__TotallyDoesNotExist__", "model": 1.0, "clip": 1.0},
            ],
            "ddim_steps": 20,
            "n_iter": 1,
            "model": stable_diffusion_model_name_for_testing,
        }

        ret = hordelib_instance.basic_inference_single_image(data)
        pil_image = ret.image
        assert pil_image is not None
        assert len(ret.faults) == 1
        # Don't save this one, just testing we didn't crash and burn

    def test_lora_trigger_inject_red(
        self,
        shared_model_manager: type[SharedModelManager],
        hordelib_instance: HordeLib,
        stable_diffusion_model_name_for_testing: str,
        lora_GlowingRunesAI: str,
    ):
        assert shared_model_manager.manager.lora

        # Red
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
            "loras": [{"name": lora_GlowingRunesAI, "model": 1.0, "clip": 1.0, "inject_trigger": "red"}],
            "ddim_steps": 20,
            "n_iter": 1,
            "model": stable_diffusion_model_name_for_testing,
        }

        pil_image = hordelib_instance.basic_inference_single_image(data).image
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
        lora_GlowingRunesAI: str,
    ):
        assert shared_model_manager.manager.lora

        # Red
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
            "loras": [{"name": lora_GlowingRunesAI, "model": 1.0, "clip": 1.0, "inject_trigger": "any"}],
            "ddim_steps": 20,
            "n_iter": 1,
            "model": stable_diffusion_model_name_for_testing,
        }

        pil_image = hordelib_instance.basic_inference_single_image(data).image
        assert pil_image is not None
        assert isinstance(pil_image, Image.Image)

        img_filename = "lora_inject_any.png"
        pil_image.save(f"images/{img_filename}", quality=100)

        pil_image_2 = hordelib_instance.basic_inference_single_image(data).image
        assert pil_image_2 is not None
        assert isinstance(pil_image_2, Image.Image)

        img_filename = "lora_inject_any_2.png"
        pil_image_2.save(f"images/{img_filename}", quality=100)

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
            "prompt": (
                "pantasa, plant, wooden robot, concept artist, ruins, night, moon, global "
                "illumination, depth of field, splash art"
            ),
            "loras": [{"name": lora_name, "model": 0.75, "clip": 1.0, "inject_trigger": "any"}],
            "ddim_steps": 20,
            "n_iter": 1,
            "model": stable_diffusion_model_name_for_testing,
        }

        pil_image = hordelib_instance.basic_inference_single_image(data).image
        assert pil_image is not None
        assert isinstance(pil_image, Image.Image)

        img_filename = "lora_download_adhoc.png"
        pil_image.save(f"images/{img_filename}", quality=100)
        assert check_single_lora_image_similarity(
            f"images_expected/{img_filename}",
            pil_image,
        )

    def test_download_and_use_specific_version_lora(
        self,
        shared_model_manager: type[SharedModelManager],
        hordelib_instance: HordeLib,
        stable_diffusion_model_name_for_testing: str,
    ):
        assert shared_model_manager.manager.lora

        lora_name = "238435"
        shared_model_manager.manager.lora.ensure_lora_deleted("180780")
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
            "prompt": "cat made of crystalz in a mythical forest, masterpiece, intricate details, wide shot",
            "loras": [{"name": lora_name, "model": 1.0, "clip": 1.0, "inject_trigger": "any", "is_version": True}],
            "ddim_steps": 20,
            "n_iter": 1,
            "model": stable_diffusion_model_name_for_testing,
        }

        pil_image = hordelib_instance.basic_inference_single_image(data).image
        assert pil_image is not None
        assert isinstance(pil_image, Image.Image)

        img_filename = "lora_version_adhoc.png"
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
            "prompt": (
                "pantasa, plant, wooden robot, concept artist, ruins, night, moon, global "
                "illumination, depth of field, splash art"
            ),
            "loras": [
                {"name": "48139", "model": 0.75, "clip": 1.0},
                {"name": "58390", "model": 1, "clip": 1.0},
                {"name": "13941", "model": 1, "clip": 1.0},
            ],
            "ddim_steps": 20,
            "n_iter": 1,
            "model": stable_diffusion_model_name_for_testing,
        }

        pil_image = hordelib_instance.basic_inference_single_image(data).image
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
            "prompt": (
                "pantasa, plant, wooden robot, concept artist, ruins, night, moon, global "
                "illumination, depth of field, splash art"
            ),
            "loras": [
                {"name": "35822", "model": 1, "clip": 1.0},
            ],
            "ddim_steps": 20,
            "n_iter": 1,
            "model": stable_diffusion_model_name_for_testing,
        }

        ret = hordelib_instance.basic_inference_single_image(data)
        pil_image = ret.image
        assert pil_image is not None

        img_filename = "lora_baseline_mismatch.png"
        pil_image.save(f"images/{img_filename}", quality=100)

        assert check_single_lora_image_similarity(
            f"images_expected/{img_filename}",
            pil_image,
        )

    def test_stonepunk(
        self,
        hordelib_instance: HordeLib,
        shared_model_manager: type[SharedModelManager],
        stable_diffusion_model_name_for_testing: str,
    ):
        assert shared_model_manager.manager.lora
        lora_name: str | None = "51539"
        lora_name = shared_model_manager.manager.lora.fetch_adhoc_lora(lora_name)
        assert shared_model_manager.manager.lora.is_model_available(lora_name)
        assert lora_name

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

        pil_image = hordelib_instance.basic_inference_single_image(data).image
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
        shared_model_manager: type[SharedModelManager],
        stable_diffusion_model_name_for_testing: str,
    ):
        assert shared_model_manager.manager.lora
        lora_name: str | None = "58390"
        lora_name = shared_model_manager.manager.lora.fetch_adhoc_lora(lora_name)
        assert shared_model_manager.manager.lora.is_model_available(lora_name)
        assert lora_name

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

        pil_image = hordelib_instance.basic_inference_single_image(data).image
        assert pil_image is not None
        assert isinstance(pil_image, Image.Image)

        img_filename = "lora_negative_strength.png"
        pil_image.save(f"images/{img_filename}", quality=100)

        assert check_single_lora_image_similarity(
            f"images_expected/{img_filename}",
            pil_image,
        )

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

        pil_image = hordelib_instance.basic_inference_single_image(data).image
        assert pil_image is not None
        assert isinstance(pil_image, Image.Image)

        img_filename = "lora_positive_strength.png"
        pil_image.save(f"images/{img_filename}", quality=100)

        assert check_single_lora_image_similarity(
            f"images_expected/{img_filename}",
            pil_image,
        )

    def test_civitai_token(
        self,
        shared_model_manager: type[SharedModelManager],
    ):
        assert shared_model_manager.manager.lora
        assert shared_model_manager.manager.lora._civitai_api_token is not None

    def test_login_gated_lora(
        self,
        shared_model_manager: type[SharedModelManager],
        hordelib_instance: HordeLib,
    ):
        assert shared_model_manager.manager.lora

        download_gated_lora_version_id = "214296"

        assert shared_model_manager.manager.lora.fetch_adhoc_lora(
            download_gated_lora_version_id,
            timeout=45,
            is_version=True,
        )

        trigger = shared_model_manager.manager.lora.find_lora_trigger(
            download_gated_lora_version_id,
            "text logo",
            is_version=True,
        )
        data = {
            "sampler_name": "k_euler_a",
            "cfg_scale": 6.0,
            "denoising_strength": 1.0,
            "seed": 1234567890,
            "height": 1024,
            "width": 1024,
            "karras": True,
            "tiling": False,
            "hires_fix": False,
            "clip_skip": 1,
            "control_type": None,
            "image_is_control": False,
            "return_control_map": False,
            "prompt": f"a slick, modern, sign ('Success' {trigger}:2.0)",
            "loras": [{"name": download_gated_lora_version_id, "model": 1.0, "clip": 1.0, "is_version": True}],
            "ddim_steps": 25,
            "n_iter": 1,
            "model": "SDXL 1.0",
        }
        ret = hordelib_instance.basic_inference_single_image(data)
        assert isinstance(ret, ResultingImageReturn)
        pil_image = ret.image
        assert pil_image is not None
        assert isinstance(pil_image, Image.Image)
        assert len(ret.faults) == 0

        img_filename = "lora_download_gated.png"
        pil_image.save(f"images/{img_filename}", quality=100)

        assert check_single_lora_image_similarity(
            f"images_expected/{img_filename}",
            pil_image,
        )
