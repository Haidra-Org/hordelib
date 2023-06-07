# test_horde_lora.py
import os
from datetime import datetime, timedelta

import pytest
from PIL import Image

from hordelib.horde import HordeLib
from hordelib.shared_model_manager import SharedModelManager
from hordelib.utils.distance import are_images_identical


class TestHordeLora:
    @pytest.fixture(autouse=True, scope="class")
    def setup_and_teardown(self):
        TestHordeLora.horde = HordeLib()

        TestHordeLora.default_model_manager_args = {
            "compvis": True,
            "lora": True,
        }
        SharedModelManager.loadModelManagers(**self.default_model_manager_args)
        assert SharedModelManager.manager is not None
        assert SharedModelManager.manager.lora is not None
        SharedModelManager.manager.load("Deliberate")
        SharedModelManager.manager.lora.download_default_loras()
        SharedModelManager.manager.lora.wait_for_downloads()
        TestHordeLora.distance_threshold = int(os.getenv("IMAGE_DISTANCE_THRESHOLD", "100000"))
        yield
        SharedModelManager.manager.lora.stop_all()
        del TestHordeLora.horde
        SharedModelManager._instance = None
        SharedModelManager.manager = None

    def test_text_to_image_lora_red(self):

        # Red
        lora_name = SharedModelManager.manager.lora.get_lora_name("GlowingRunesAI")
        trigger = SharedModelManager.manager.lora.find_lora_trigger(lora_name, "red")
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
            "model": "Deliberate",
        }
        assert self.horde is not None
        pil_image = self.horde.basic_inference(data)
        assert pil_image is not None
        img_filename = "lora_red.png"
        pil_image.save(f"images/{img_filename}", quality=100)
        assert are_images_identical(f"images_expected/{img_filename}", pil_image, self.distance_threshold)
        assert SharedModelManager.manager.lora.get_lora_last_use("GlowingRunesAI") > datetime.now() - timedelta(
            minutes=1,
        )

    def test_text_to_image_lora_blue(self):

        # Blue, fuzzy search on version
        lora_name = SharedModelManager.manager.lora.get_lora_name("GlowingRunesAI")
        trigger = SharedModelManager.manager.lora.find_lora_trigger(lora_name, "blue")
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
            "model": "Deliberate",
        }
        assert self.horde is not None
        pil_image = self.horde.basic_inference(data)
        assert pil_image is not None
        img_filename = "lora_blue.png"
        pil_image.save(f"images/{img_filename}", quality=100)
        assert are_images_identical(f"images_expected/{img_filename}", pil_image, self.distance_threshold)

    def test_text_to_image_lora_chained(self):
        lora_name = SharedModelManager.manager.lora.get_lora_name("GlowingRunesAI")
        trigger = SharedModelManager.manager.lora.find_lora_trigger(lora_name, "red")
        trigger2 = SharedModelManager.manager.lora.find_lora_trigger(lora_name, "blue")
        lora_name2 = SharedModelManager.manager.lora.get_lora_name("Dra9onScaleAI")
        trigger3 = SharedModelManager.manager.lora.find_lora_trigger(lora_name, "Dr490nSc4leAI")

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
            "model": "Deliberate",
        }
        assert self.horde is not None
        pil_image = self.horde.basic_inference(data)
        assert pil_image is not None
        img_filename = "lora_multiple.png"
        pil_image.save(f"images/{img_filename}", quality=100)
        assert are_images_identical(f"images_expected/{img_filename}", pil_image, self.distance_threshold)

    def test_text_to_image_lora_chained_bad(self):
        lora_name = SharedModelManager.manager.lora.get_lora_name("GlowingRunesAI")
        trigger = SharedModelManager.manager.lora.find_lora_trigger(lora_name, "blue")
        lora_name2 = SharedModelManager.manager.lora.get_lora_name("Dra9onScaleAI")
        trigger2 = SharedModelManager.manager.lora.find_lora_trigger(lora_name, "Dr490nSc4leAI")
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
            "model": "Deliberate",
        }
        assert self.horde is not None
        pil_image = self.horde.basic_inference(data)
        assert pil_image is not None
        # Don't save this one, just testing we didn't crash and burn

    def test_lora_trigger_inject_red(self):
        # Red
        lora_name = SharedModelManager.manager.lora.get_lora_name("GlowingRunesAI")
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
            "model": "Deliberate",
        }
        assert self.horde is not None
        pil_image = self.horde.basic_inference(data)
        assert pil_image is not None
        img_filename = "lora_inject_red.png"
        pil_image.save(f"images/{img_filename}", quality=100)
        assert are_images_identical(f"images_expected/{img_filename}", pil_image, self.distance_threshold)

    def test_lora_trigger_inject_any(self):
        # Red
        lora_name = SharedModelManager.manager.lora.get_lora_name("GlowingRunesAI")
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
            "model": "Deliberate",
        }
        assert self.horde is not None
        pil_image = self.horde.basic_inference(data)
        assert pil_image is not None
        img_filename = "lora_inject_any.png"
        pil_image.save(f"images/{img_filename}", quality=100)
        assert are_images_identical(f"images_expected/{img_filename}", pil_image, self.distance_threshold)

    def test_download_and_use_adhoc_lora(self):
        lora_name = "74384"
        SharedModelManager.manager.lora.ensure_lora_deleted(lora_name)
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
            "model": "Deliberate",
        }
        assert self.horde is not None
        pil_image = self.horde.basic_inference(data)
        assert pil_image is not None
        img_filename = "lora_download_adhoc.png"
        pil_image.save(f"images/{img_filename}", quality=100)
        # assert are_images_identical(f"images_expected/{img_filename}", pil_image, self.distance_threshold)

    def test_for_probability_tensor_runtime_error(self):
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
            "model": "Deliberate",
        }
        assert self.horde is not None
        pil_image = self.horde.basic_inference(data)
        assert pil_image is not None

    def test_sd21_lora_against_sd15_model(self):
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
            "model": "Deliberate",
        }
        assert self.horde is not None
        pil_image = self.horde.basic_inference(data)
        assert pil_image is not None

    def test_stonepunk(self):

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
            "model": "Deliberate",
        }
        assert self.horde is not None
        pil_image = self.horde.basic_inference(data)
        assert pil_image is not None
        img_filename = "lora_stonepunk.png"
        pil_image.save(f"images/{img_filename}", quality=100)
        assert are_images_identical(f"images_expected/{img_filename}", pil_image, self.distance_threshold)
