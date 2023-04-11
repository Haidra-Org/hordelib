# test_horde.py
import pytest
from PIL import Image

from hordelib.horde import HordeLib
from hordelib.shared_model_manager import SharedModelManager


class TestHordeInference:
    @pytest.fixture(autouse=True, scope="class")
    def setup_and_teardown(self):
        TestHordeInference.horde = HordeLib()

        TestHordeInference.default_model_manager_args = {
            # aitemplate
            # "blip": True,
            # "clip": True,
            # "codeformer": True,
            "compvis": True,
            # "controlnet": True,
            # "diffusers": True,
            # "esrgan": True,
            # "gfpgan": True,
            # "safety_checker": True,
        }
        SharedModelManager.loadModelManagers(**self.default_model_manager_args)
        assert SharedModelManager.manager is not None
        SharedModelManager.manager.load("Deliberate")
        yield
        del TestHordeInference.horde
        SharedModelManager._instance = None
        SharedModelManager.manager = None

    def test_image_to_image(self):
        data = {
            "sampler_name": "k_dpmpp_2m",
            "cfg_scale": 7.5,
            "denoising_strength": 0.4,
            "seed": 666,
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
            "ddim_steps": 25,
            "n_iter": 1,
            "model": "Deliberate",
            "source_image": Image.open("images/test_db0.jpg"),
            "source_processing": "img2img",
        }
        assert self.horde is not None
        pil_image = self.horde.basic_inference(data)
        assert pil_image is not None
        assert pil_image.size == (512, 512)
        assert data["source_image"].size == (512, 512)
        pil_image.save("images/horde_image_to_image.webp", quality=90)

    def test_image_to_image_hires_fix_small(self):
        data = {
            "sampler_name": "k_dpmpp_2m",
            "cfg_scale": 7.5,
            "denoising_strength": 0.4,
            "seed": 666,
            "height": 512,
            "width": 512,
            "karras": False,
            "tiling": False,
            "hires_fix": True,
            "clip_skip": 1,
            "control_type": None,
            "image_is_control": False,
            "return_control_map": False,
            "prompt": "a dinosaur",
            "ddim_steps": 25,
            "n_iter": 1,
            "model": "Deliberate",
            "source_image": Image.open("images/test_db0.jpg"),
            "source_processing": "img2img",
        }
        assert self.horde is not None
        pil_image = self.horde.basic_inference(data)
        assert pil_image is not None
        assert pil_image.size == (512, 512)
        pil_image.save("images/horde_image_to_image_hires_fix_small.webp", quality=90)

    def test_image_to_image_hires_fix_large(self):
        data = {
            "sampler_name": "k_dpmpp_2m",
            "cfg_scale": 7.5,
            "denoising_strength": 0.4,
            "seed": 1312,
            "height": 768,
            "width": 768,
            "karras": False,
            "tiling": False,
            "hires_fix": True,
            "clip_skip": 1,
            "control_type": None,
            "image_is_control": False,
            "return_control_map": False,
            "prompt": "a dinosaur",
            "ddim_steps": 25,
            "n_iter": 1,
            "model": "Deliberate",
            "source_image": Image.open("images/test_db0.jpg"),
            "source_processing": "img2img",
        }
        assert self.horde is not None
        pil_image = self.horde.basic_inference(data)
        assert pil_image is not None
        assert pil_image.size == (768, 768)
        pil_image.save("images/horde_image_to_image_hires_fix_large.webp", quality=90)

    def test_image_to_image_mask_in(self):
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
            "model": "Deliberate",
            "source_image": Image.open("images/test_inpaint.png"),
            "source_processing": "inpainting",
        }
        assert self.horde is not None
        pil_image = self.horde.basic_inference(data)
        assert pil_image is not None
        assert pil_image.size == (512, 512)
        pil_image.save("images/horde_image_to_image_mask_in.webp", quality=90)

    def test_image_to_image_mask_out(self):
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
            "model": "Deliberate",
            "source_image": Image.open("images/test_outpaint.png"),
            "source_processing": "outpainting",
        }
        assert self.horde is not None
        pil_image = self.horde.basic_inference(data)
        assert pil_image is not None
        assert pil_image.size == (512, 512)
        pil_image.save("images/horde_image_to_image_mask_out.webp", quality=90)

    def test_img2img_alpha_to_inpainting_convert(self):
        data = {
            "sampler_name": "k_dpmpp_2m",
            "cfg_scale": 7.5,
            "denoising_strength": 1,
            "seed": 3,
            "height": 512,
            "width": 512,
            "karras": True,
            "clip_skip": 1,
            "prompt": "a mecha robot sitting on a bench",
            "ddim_steps": 20,
            "n_iter": 1,
            "model": "Deliberate",
            "source_image": Image.open("images/test_inpaint_alpha.png"),
            "source_processing": "img2img",
        }
        assert self.horde is not None
        pil_image = self.horde.basic_inference(data)
        assert pil_image is not None
        assert pil_image.size == (512, 512)
        pil_image.save("images/horde_img2img_to_inpainting.webp", quality=90)

    def test_image_to_faulty_source_image(self):
        data = {
            "sampler_name": "k_dpmpp_2m",
            "cfg_scale": 7.5,
            "denoising_strength": 0.4,
            "seed": 250636385744582,
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
            "ddim_steps": 25,
            "n_iter": 1,
            "model": "Deliberate",
            "source_image": "THIS SHOULD FAILOVER TO TEXT2IMG",
            "source_processing": "img2img",
        }
        assert self.horde is not None
        pil_image = self.horde.basic_inference(data)
        assert pil_image is not None
        assert "source_image" not in data
        assert "source_processing" not in data
