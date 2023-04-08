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
            "blip": True,
            "clip": True,
            "codeformer": True,
            "compvis": True,
            "controlnet": True,
            "diffusers": True,
            "esrgan": True,
            "gfpgan": True,
            "safety_checker": True,
        }
        SharedModelManager.loadModelManagers(**self.default_model_manager_args)
        assert SharedModelManager.manager is not None
        SharedModelManager.manager.load("Deliberate")
        SharedModelManager.manager.load("RealESRGAN_x4plus")
        yield
        del self.horde
        SharedModelManager._instance = None
        SharedModelManager.manager = None

    def test_parameter_remap_text_to_image_simple(self):
        data = {
            "sampler_name": "k_lms",
            "cfg_scale": 5,
            "denoising_strength": 0.75,
            "seed": "23113",
            "height": 512,
            "width": 512,
            "karras": True,
            "tiling": False,
            "hires_fix": False,
            "clip_skip": 1,
            "control_type": "canny",
            "image_is_control": False,
            "return_control_map": False,
            "prompt": "a dog ### cat, mouse, lion",
            "ddim_steps": 30,
            "n_iter": 1,
            "model": "Deliberate",
        }

        expected = {
            "sampler.sampler_name": "lms",
            "sampler.cfg": 5,
            "sampler.denoise": 0.75,
            "sampler.seed": 23113,
            "empty_latent_image.height": 512,
            "empty_latent_image.width": 512,
            "sampler.steps": 30,
            "empty_latent_image.batch_size": 1,
            "model_loader.model_name": "Deliberate",
            "sampler.scheduler": "karras",
            "prompt.text": "a dog",
            "negative_prompt.text": "cat, mouse, lion",
            "clip_skip.stop_at_clip_layer": -1,
            "model_loader.model_manager": SharedModelManager,
        }
        assert self.horde is not None
        result = self.horde._parameter_remap_text_to_image(data)
        assert result == expected, f"Dictionaries don't match: {result} != {expected}"

    def test_parameter_remap_variation(self):
        data = {
            "sampler_name": "k_lms",
            "cfg_scale": 5,
            "denoising_strength": 0.75,
            "seed": "23113",
            "height": 512,
            "width": 512,
            "karras": False,
            "tiling": False,
            "hires_fix": False,
            "clip_skip": 1,
            "control_type": "canny",
            "image_is_control": False,
            "return_control_map": False,
            "prompt": "a dog",
            "ddim_steps": 30,
            "n_iter": 1,
            "model": "Deliberate",
        }

        expected = {
            "sampler.sampler_name": "lms",
            "sampler.cfg": 5,
            "sampler.denoise": 0.75,
            "sampler.seed": 23113,
            "empty_latent_image.height": 512,
            "empty_latent_image.width": 512,
            "sampler.steps": 30,
            "empty_latent_image.batch_size": 1,
            "model_loader.model_name": "Deliberate",
            "sampler.scheduler": "normal",
            "prompt.text": "a dog",
            "negative_prompt.text": "",
            "clip_skip.stop_at_clip_layer": -1,
            "model_loader.model_manager": SharedModelManager,
        }
        assert self.horde is not None
        result = self.horde._parameter_remap_text_to_image(data)
        assert result == expected, f"Dictionaries don't match: {result} != {expected}"

    def test_text_to_image(self):
        data = {
            "sampler_name": "k_dpmpp_2m",
            "cfg_scale": 7.5,
            "denoising_strength": 1.0,
            "seed": 123456789,
            "height": 512,
            "width": 512,
            "karras": True,
            "tiling": False,
            "hires_fix": False,
            "clip_skip": 1,
            "control_type": "canny",
            "image_is_control": False,
            "return_control_map": False,
            "prompt": "an ancient llamia monster",
            "ddim_steps": 25,
            "n_iter": 1,
            "model": "Deliberate",
        }
        assert self.horde is not None
        pil_image = self.horde.text_to_image(data)
        assert pil_image is not None
        pil_image.save("images/horde_text_to_image.png")

    def test_text_to_image_small(self):
        data = {
            "sampler_name": "k_dpmpp_2m",
            "cfg_scale": 7.5,
            "denoising_strength": 1.0,
            "seed": 123456789,
            "height": 256,
            "width": 256,
            "karras": True,
            "tiling": False,
            "hires_fix": False,
            "clip_skip": 1,
            "control_type": "canny",
            "image_is_control": False,
            "return_control_map": False,
            "prompt": "dinosaur ### painting, drawing, artwork",
            "ddim_steps": 12,
            "n_iter": 1,
            "model": "Deliberate",
        }
        assert self.horde is not None
        pil_image = self.horde.text_to_image(data)
        assert pil_image is not None
        pil_image.save("images/horde_text_to_image_small.png")

    def test_text_to_image_clip_skip_2(self):
        data = {
            "sampler_name": "k_dpmpp_2m",
            "cfg_scale": 7.5,
            "denoising_strength": 1.0,
            "seed": 123456789,
            "height": 512,
            "width": 512,
            "karras": True,
            "tiling": False,
            "hires_fix": False,
            "clip_skip": 2,
            "control_type": "canny",
            "image_is_control": False,
            "return_control_map": False,
            "prompt": "an ancient llamia monster",
            "ddim_steps": 25,
            "n_iter": 1,
            "model": "Deliberate",
        }
        assert self.horde is not None
        pil_image = self.horde.text_to_image(data)
        assert pil_image is not None
        pil_image.save("images/horde_text_to_image_clip_skip_2.png")

    def test_text_to_image_hires_fix(self):
        data = {
            "sampler_name": "k_dpmpp_2m",
            "cfg_scale": 7.5,
            "denoising_strength": 1.0,
            "seed": 123456789,
            "height": 768,
            "width": 768,
            "karras": True,
            "tiling": False,
            "hires_fix": True,
            "clip_skip": 1,
            "control_type": "canny",
            "image_is_control": False,
            "return_control_map": False,
            "prompt": "an ancient llamia monster",
            "ddim_steps": 25,
            "n_iter": 1,
            "model": "Deliberate",
        }
        assert self.horde is not None
        pil_image = self.horde.text_to_image(data)
        assert pil_image is not None
        pil_image.save("images/horde_text_to_image_hires_fix.png")

    def test_image_to_image(self):
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
            "control_type": "canny",
            "image_is_control": False,
            "return_control_map": False,
            "prompt": "a dinosaur",
            "ddim_steps": 25,
            "n_iter": 1,
            "model": "Deliberate",
            "source_image": Image.open("images/horde_text_to_image.png"),
            "source_processing": "img2img",
        }
        assert self.horde is not None
        pil_image = self.horde.text_to_image(data)
        assert pil_image is not None
        pil_image.save("images/horde_image_to_image.png")

    def test_image_upscale(self):
        data = {
            "model": "RealESRGAN_x4plus",
            "source_image": Image.open("images/horde_text_to_image_small.png"),
        }
        assert self.horde is not None
        pil_image = self.horde.image_upscale(data)
        assert pil_image is not None
        pil_image.save("images/horde_image_upscale.png")

    def test_image_to_image_hires_fix_small(self):
        data = {
            "sampler_name": "k_dpmpp_2m",
            "cfg_scale": 7.5,
            "denoising_strength": 0.4,
            "seed": 250636385744582,
            "height": 512,
            "width": 512,
            "karras": False,
            "tiling": False,
            "hires_fix": True,
            "clip_skip": 1,
            "control_type": "canny",
            "image_is_control": False,
            "return_control_map": False,
            "prompt": "a dinosaur",
            "ddim_steps": 25,
            "n_iter": 1,
            "model": "Deliberate",
            "source_image": Image.open("images/horde_text_to_image.png"),
            "source_processing": "img2img",
        }
        assert self.horde is not None
        pil_image = self.horde.text_to_image(data)
        assert pil_image is not None
        pil_image.save("images/horde_image_to_image_hires_fix_small.png")

    def test_image_to_image_hires_fix_large(self):
        data = {
            "sampler_name": "k_dpmpp_2m",
            "cfg_scale": 7.5,
            "denoising_strength": 0.4,
            "seed": 250636385744582,
            "height": 768,
            "width": 768,
            "karras": False,
            "tiling": False,
            "hires_fix": True,
            "clip_skip": 1,
            "control_type": "canny",
            "image_is_control": False,
            "return_control_map": False,
            "prompt": "a dinosaur",
            "ddim_steps": 25,
            "n_iter": 1,
            "model": "Deliberate",
            "source_image": Image.open("images/horde_text_to_image.png"),
            "source_processing": "img2img",
        }
        assert self.horde is not None
        pil_image = self.horde.text_to_image(data)
        assert pil_image is not None
        pil_image.save("images/horde_image_to_image_hires_fix_large.png")

    def test_image_to_image_inpainting(self):
        data = {
            "sampler_name": "k_dpmpp_2m",
            "cfg_scale": 7.5,
            "denoising_strength": 0.72,
            "seed": 836913938046008,
            "height": 512,
            "width": 512,
            "karras": False,
            "tiling": False,
            "hires_fix": False,
            "clip_skip": 1,
            "control_type": "canny",
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
        pil_image = self.horde.text_to_image(data)
        assert pil_image is not None
        pil_image.save("images/horde_image_to_image_inpainting.png")

    def test_image_to_image_outpainting(self):
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
            "control_type": "canny",
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
        pil_image = self.horde.text_to_image(data)
        assert pil_image is not None
        pil_image.save("images/horde_image_to_image_outpainting.png")
