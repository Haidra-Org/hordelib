import pytest
from PIL import Image

from hordelib.horde import HordeLib
from hordelib.shared_model_manager import SharedModelManager


class TestHordeInference:
    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        self.horde = HordeLib()

        self.default_model_manager_args = {
            "compvis": True,
        }
        SharedModelManager.loadModelManagers(**self.default_model_manager_args)
        assert SharedModelManager.manager is not None
        SharedModelManager.manager.load("Deliberate Inpainting")
        assert (
            SharedModelManager.manager.compvis.is_model_loaded(
                "Deliberate Inpainting",
            )
            is True
        )
        yield
        del self.horde
        SharedModelManager._instance = None
        SharedModelManager.manager = None

    def test_inpainting_alpha_mask(self):
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
            "model": "Deliberate Inpainting",
            "source_image": Image.open("images/test_inpaint_alpha.png"),
            "source_processing": "inpainting",
        }
        assert self.horde is not None
        pil_image = self.horde.basic_inference(data)
        assert pil_image is not None
        pil_image.save("images/inpainting_mask_alpha.webp", quality=90)

    def test_inpainting_separate_mask(self):
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
            "model": "Deliberate Inpainting",
            "source_image": Image.open("images/test_inpaint_original.png"),
            "source_mask": Image.open("images/test_inpaint_mask.png"),
            "source_processing": "inpainting",
        }
        assert self.horde is not None
        pil_image = self.horde.basic_inference(data)
        assert pil_image is not None
        pil_image.save("images/inpainting_mask_separate.webp", quality=90)

    def test_inpainting_alpha_mask_mountains(self):
        data = {
            "sampler_name": "k_dpmpp_2m",
            "cfg_scale": 7.5,
            "denoising_strength": 0.72,
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
            "prompt": "a river through the mountains",
            "ddim_steps": 20,
            "n_iter": 1,
            "model": "Deliberate Inpainting",
            "source_image": Image.open("images/test_inpaint.png"),
            "source_processing": "inpainting",
        }
        assert self.horde is not None
        pil_image = self.horde.basic_inference(data)
        assert pil_image is not None
        assert pil_image.size == (512, 512)
        pil_image.save("images/inpainting_mountains.webp", quality=90)

    def test_outpainting_alpha_mask_mountains(self):
        data = {
            "sampler_name": "euler",
            "cfg_scale": 8.0,
            "denoising_strength": 1.0,
            "seed": 836913938046008,
            "height": 512,
            "width": 512,
            "karras": False,
            "tiling": False,
            "hires_fix": False,
            "clip_skip": 1,
            "control_type": None,
            "image_is_control": False,
            "return_control_map": False,
            "prompt": "a river through the mountains, blue sky with clouds.",
            "ddim_steps": 20,
            "n_iter": 1,
            "model": "Deliberate Inpainting",
            "source_image": Image.open("images/test_outpaint.png"),
            "source_processing": "outpainting",
        }
        assert self.horde is not None
        pil_image = self.horde.basic_inference(data)
        assert pil_image is not None
        assert pil_image.size == (512, 512)
        pil_image.save("images/outpainting_mountains.webp", quality=90)
