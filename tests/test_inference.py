# test_inference.py
import pytest
from hordelib.comfy import Comfy
from PIL import Image


class TestInference:
    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        self.comfy = Comfy()
        yield
        self.comfy = None

    def test_stable_diffusion_pipeline(self):
        params = {
            "sampler.seed": 12345,
            "sampler.cfg": 7.5,
            "sampler.scheduler": "karras",
            "sampler.sampler_name": "dpmpp_2m",
            "sampler.steps": 25,
            "prompt.text": "a closeup photo of a confused dog",
            "negative_prompt.text": "cat, black and white, deformed",
            "model_loader.ckpt_name": "model.ckpt",
            "empty_latent_image.width": 768,
            "empty_latent_image.height": 768,
        }
        images = self.comfy.run_image_pipeline("stable_diffusion", params)

        # XXX for proof of concept development we just save the image to a file
        image = Image.open(images[0]["imagedata"])
        image.save("test-image.png")
