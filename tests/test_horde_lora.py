# test_horde_lora.py
import pytest
from PIL import Image

from hordelib.horde import HordeLib
from hordelib.shared_model_manager import SharedModelManager


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
        yield
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
            "karras": True,
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
        print(data)
        assert self.horde is not None
        pil_image = self.horde.basic_inference(data)
        assert pil_image is not None
        pil_image.save("images/lora_red.webp", quality=90)

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
            "karras": True,
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
        pil_image.save("images/lora_blue.webp", quality=90)

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
            "karras": True,
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
        pil_image.save("images/lora_multiple.webp", quality=90)

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
            "karras": True,
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
            "karras": True,
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
        pil_image.save("images/lora_inject_red.webp", quality=90)

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
            "karras": True,
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
        pil_image.save("images/lora_inject_any.webp", quality=90)
