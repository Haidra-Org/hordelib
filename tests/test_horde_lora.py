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
        TestHordeLora.lora1 = None
        for lora in SharedModelManager.manager.lora.model_reference:
            if len(SharedModelManager.manager.lora.model_reference[lora]["triggers"]) >= 2:
                TestHordeLora.lora1 = lora
        if TestHordeLora.lora1 is None:
            TestHordeLora.lora1 = list(SharedModelManager.manager.lora.model_reference.keys())[0]
        TestHordeLora.trigger11 = SharedModelManager.manager.lora.model_reference[TestHordeLora.lora1]["triggers"][0]
        TestHordeLora.trigger12 = SharedModelManager.manager.lora.model_reference[TestHordeLora.lora1]["triggers"][1]
        TestHordeLora.filename1 = SharedModelManager.manager.lora.get_lora_filename(TestHordeLora.lora1)
        for lora in SharedModelManager.manager.lora.model_reference:
            if lora != TestHordeLora.lora1:
                TestHordeLora.lora2 = lora
                break
        TestHordeLora.trigger21 = SharedModelManager.manager.lora.model_reference[TestHordeLora.lora2]["triggers"][0]
        TestHordeLora.filename2 = SharedModelManager.manager.lora.get_lora_filename(TestHordeLora.lora2)
        yield
        del TestHordeLora.horde
        SharedModelManager._instance = None
        SharedModelManager.manager = None

    def test_text_to_image_lora_red(self):

        # Red
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
            "prompt": "a dark magical crystal, GlowingRunesAIV2_red",
            "loras": [{"name": "GlowingRunesAI", "model": 1.0, "clip": 1.0}],
            "ddim_steps": 20,
            "n_iter": 1,
            "model": "Deliberate",
        }
        assert self.horde is not None
        pil_image = self.horde.basic_inference(data)
        assert pil_image is not None
        pil_image.save("images/lora_red.webp", quality=90)

    def test_text_to_image_lora_blue(self):

        # Blue, fuzzy search on version
        data = {
            "sampler_name": "k_euler",
            "cfg_scale": 8.0,
            "denoising_strength": 1.0,
            "seed": 851616030078638,
            "height": 512,
            "width": 512,
            "karras": True,
            "tiling": False,
            "hires_fix": False,
            "clip_skip": 1,
            "control_type": None,
            "image_is_control": False,
            "return_control_map": False,
            "prompt": "a dark magical crystal, GlowingRunesAIV2_paleblue",
            "loras": [{"name": "GlowingRunesAIV6", "model": 1.0, "clip": 1.0}],
            "ddim_steps": 20,
            "n_iter": 1,
            "model": "Deliberate",
        }
        assert self.horde is not None
        pil_image = self.horde.basic_inference(data)
        assert pil_image is not None
        pil_image.save("images/lora_blue.webp", quality=90)

    def test_text_to_image_lora_chained(self):

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
            "prompt": "a dark magical crystal, GlowingRunesAIV2_red, Dr490nSc4leAI",
            "loras": [
                {"name": "GlowingRunesAIV6", "model": 1.0, "clip": 1.0},
                {"name": "Dra9onScaleAI", "model": 1.0, "clip": 1.0},
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
            "prompt": f"an open field of flowers, {TestHordeLora.trigger11}, {TestHordeLora.trigger21}",
            "loras": [
                {"name": TestHordeLora.filename1, "model": 1.0, "clip": 1.0},
                {"name": TestHordeLora.filename2, "model": 1.0, "clip": 1.0},
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

    # This is risky, one time it came up with: "an open field of flowers, balls_deep"
    # def test_text_to_image_lora_random(self):

    #     # Trigger1
    #     data = {
    #         "sampler_name": "k_euler",
    #         "cfg_scale": 8.0,
    #         "denoising_strength": 1.0,
    #         "seed": 304886399544324,
    #         "height": 512,
    #         "width": 512,
    #         "karras": True,
    #         "tiling": False,
    #         "hires_fix": False,
    #         "clip_skip": 1,
    #         "control_type": None,
    #         "image_is_control": False,
    #         "return_control_map": False,
    #         "prompt": f"an open field of flowers, {TestHordeLora.trigger11}",
    #         "loras": [{"name": TestHordeLora.filename1, "model": 1.0, "clip": 1.0}],
    #         "ddim_steps": 20,
    #         "n_iter": 1,
    #         "model": "Deliberate",
    #     }
    #     assert self.horde is not None
    #     pil_image = self.horde.basic_inference(data)
    #     assert pil_image is not None
    #     pil_image.save("images/lora_random.webp", quality=90)
