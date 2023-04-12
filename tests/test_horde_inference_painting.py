# test_horde.py
import pytest
from PIL import Image

from hordelib.horde import HordeLib
from hordelib.shared_model_manager import SharedModelManager


class TestHordeInference:
    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        self.horde = HordeLib()

        self.default_model_manager_args = {
            # aitemplate
            # "blip": True,
            # "clip": True,
            # "codeformer": True,
            # "compvis": True,
            # "controlnet": True,
            "diffusers": True,
            # "esrgan": True,
            # "gfpgan": True,
            # "safety_checker": True,
        }
        SharedModelManager.loadModelManagers(**self.default_model_manager_args)
        assert SharedModelManager.manager is not None
        yield
        del self.horde
        SharedModelManager._instance = None
        SharedModelManager.manager = None

    def test_inpainting_alpha_mask(self):
        SharedModelManager.manager.load("stable_diffusion_inpainting")
        assert (
            SharedModelManager.manager.diffusers.is_model_loaded(
                "stable_diffusion_inpainting",
            )
            is True
        )
        data = {
            "sampler_name": "k_dpmpp_2m",
            "cfg_scale": 7.5,
            "denoising_strength": 0.6,
            "seed": 3,
            "height": 512,
            "width": 512,
            "karras": True,
            "clip_skip": 1,
            "prompt": "a mecha robot sitting on a bench",
            "ddim_steps": 20,
            "n_iter": 1,
            "model": "stable_diffusion_inpainting",
            "source_image": Image.open("images/test_inpaint_alpha.png"),
            "source_processing": "img2img",
        }
        assert self.horde is not None
        pil_image = self.horde.basic_inference(data)
        assert pil_image is not None
        pil_image.save("images/horde_inpainting_alpha_mask.webp", quality=90)

    def test_inpainting_separate_mask(self):
        SharedModelManager.manager.load("stable_diffusion_inpainting")
        assert (
            SharedModelManager.manager.diffusers.is_model_loaded(
                "stable_diffusion_inpainting",
            )
            is True
        )
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
            "model": "stable_diffusion_inpainting",
            "source_image": Image.open("images/test_inpaint_original.png"),
            "source_mask": Image.open("images/test_inpaint_mask.png"),
            "source_processing": "inpainting",
        }
        assert self.horde is not None
        pil_image = self.horde.basic_inference(data)
        assert pil_image is not None
        pil_image.save("images/horde_inpainting_separate_mask.webp", quality=90)
