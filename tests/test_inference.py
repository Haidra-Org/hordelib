# test_inference.py
import pytest
from PIL import Image

from hordelib import set_horde_model_manager
from hordelib.comfy import Comfy
from hordelib.model_manager.hyper import ModelManager


class TestInference:
    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        self.comfy = Comfy()
        model_manager = ModelManager(
            compvis=True,
        )
        model_manager.load("Deliberate")
        set_horde_model_manager(model_manager)
        yield
        self.comfy = None
        del model_manager

    def test_unknown_pipeline(self):
        result = self.comfy.run_pipeline("non-existent-pipeline", {})
        assert result is None

    def test_stable_diffusion_pipeline(self):
        params = {
            "sampler.sampler_name": "dpmpp_2m",
            "sampler.cfg": 7.5,
            "sampler.denoise": 1.0,
            "sampler.seed": 12345,
            "empty_latent_image.width": 768,
            "empty_latent_image.height": 768,
            "empty_latent_image.batch_size": 1,
            "sampler.scheduler": "karras",
            "sampler.steps": 25,
            "prompt.text": "a closeup photo of a confused dog",
            "negative_prompt.text": "cat, black and white, deformed",
            "model_loader.ckpt_name": "Deliberate",
            "clip_skip.stop_at_clip_layer": -1,
        }
        assert self.comfy is not None
        images = self.comfy.run_image_pipeline("stable_diffusion", params)

        image = Image.open(images[0]["imagedata"])
        image.save("pipeline_stable_diffusion.png")

    def test_stable_diffusion_pipeline_clip_skip(self):
        params = {
            "sampler.sampler_name": "dpmpp_2m",
            "sampler.cfg": 7.5,
            "sampler.denoise": 1.0,
            "sampler.seed": 12345,
            "empty_latent_image.width": 768,
            "empty_latent_image.height": 768,
            "empty_latent_image.batch_size": 1,
            "sampler.scheduler": "karras",
            "sampler.steps": 25,
            "prompt.text": "a closeup photo of a confused dog",
            "negative_prompt.text": "cat, black and white, deformed",
            "model_loader.ckpt_name": "Deliberate",
            "clip_skip.stop_at_clip_layer": -2,
        }
        assert self.comfy is not None
        images = self.comfy.run_image_pipeline("stable_diffusion", params)

        image = Image.open(images[0]["imagedata"])
        image.save("pipeline_stable_diffusion_clip_skip_2.png")

    def test_stable_diffusion_hires_fix_pipeline(self):
        params = {
            "sampler.seed": 12345,
            "sampler.cfg": 7.5,
            "sampler.scheduler": "normal",
            "sampler.sampler_name": "dpmpp_sde",
            "sampler.denoise": 1.0,
            "sampler.steps": 12,
            "prompt.text": (
                "(masterpiece) HDR victorian portrait painting of (girl), "
                "blonde hair, mountain nature, blue sky"
            ),
            "negative_prompt.text": "bad hands, text, watermark",
            "model_loader.ckpt_name": "Deliberate",
            "empty_latent_image.width": 768,
            "empty_latent_image.height": 768,
            "latent_upscale.width": 1216,
            "latent_upscale.height": 1216,
            "latent_upscale.crop": "disabled",
            "latent_upscale.upscale_method": "nearest-exact",
            "upscale_sampler.seed": 45678,
            "upscale_sampler.steps": 15,
            "upscale_sampler.cfg": 8.0,
            "upscale_sampler.sampler_name": "dpmpp_2m",
            "upscale_sampler.scheduler": "simple",
            "upscale_sampler.denoise": 0.5,
        }
        images = self.comfy.run_image_pipeline("stable_diffusion_hires_fix", params)

        image = Image.open(images[0]["imagedata"])
        image.save("pipeline_stable_diffusion_hires_fix.png")
