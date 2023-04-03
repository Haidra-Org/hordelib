# test_horde.py
import pytest

from hordelib import set_horde_model_manager
from hordelib.horde import HordeLib
from hordelib.model_manager.hyper import ModelManager


class TestHordeInference:
    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        self.horde = HordeLib()
        model_manager = ModelManager(
            compvis=True,
        )
        model_manager.load("Deliberate")
        set_horde_model_manager(model_manager)
        yield
        self.horde = None
        del model_manager

    def test_parameter_remap_simple(self):
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
            "model_loader.ckpt_name": "Deliberate",
            "sampler.scheduler": "karras",
            "prompt.text": "a dog",
            "negative_prompt.text": "cat, mouse, lion",
        }

        result = self.horde._parameter_remap(data)
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
            "model_loader.ckpt_name": "Deliberate",
            "sampler.scheduler": "normal",
            "prompt.text": "a dog",
            "negative_prompt.text": "",
        }

        result = self.horde._parameter_remap(data)
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
        pil_image = self.horde.text_to_image(data)
        pil_image.save("horde_text_to_image.png")
