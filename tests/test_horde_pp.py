# test_horde.py
import pytest
from PIL import Image

from hordelib.horde import HordeLib
from hordelib.shared_model_manager import SharedModelManager


class TestHordeUpscaling:
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
            # "diffusers": True,
            "esrgan": True,
            # "gfpgan": True,
            # "safety_checker": True,
        }
        SharedModelManager.loadModelManagers(**self.default_model_manager_args)
        assert SharedModelManager.manager is not None
        self.image = Image.open("images/test_db0.jpg")
        self.width, self.height = self.image.size
        yield
        del self.horde
        SharedModelManager._instance = None
        SharedModelManager.manager = None

    def test_image_upscale_RealESRGAN_x4plus(self):
        SharedModelManager.manager.load("RealESRGAN_x4plus")
        assert (
            SharedModelManager.manager.esrgan.is_model_loaded("RealESRGAN_x4plus")
            is True
        )
        data = {
            "model": "RealESRGAN_x4plus",
            "source_image": self.image,
        }
        assert self.horde is not None
        pil_image = self.horde.image_upscale(data)
        assert pil_image is not None
        width, height = pil_image.size
        # assert width == self.width * 4
        # assert height == self.height * 4
        pil_image.save("images/horde_image_upscale_RealESRGAN_x4plus.webp", quality=90)

    def test_image_upscale_RealESRGAN_x2plus(self):
        SharedModelManager.manager.load("RealESRGAN_x2plus")
        assert (
            SharedModelManager.manager.esrgan.is_model_loaded("RealESRGAN_x2plus")
            is True
        )
        data = {
            "model": "RealESRGAN_x2plus",
            "source_image": self.image,
        }
        pil_image = self.horde.image_upscale(data)
        assert pil_image is not None
        width, height = pil_image.size
        # assert width == self.width * 2
        # assert height == self.height * 2
        pil_image.save("images/horde_image_upscale_RealESRGAN_x2plus.webp", quality=90)

    def test_image_upscale_NMKD_Siax(self):
        SharedModelManager.manager.load("NMKD_Siax")
        assert SharedModelManager.manager.esrgan.is_model_loaded("NMKD_Siax") is True
        data = {
            "model": "NMKD_Siax",
            "source_image": self.image,
        }
        pil_image = self.horde.image_upscale(data)
        assert pil_image is not None
        width, height = pil_image.size
        # assert width == self.width * 4
        # assert height == self.height * 4
        pil_image.save("images/horde_image_upscale_NMKD_Siax.webp", quality=90)

    def test_image_upscale_RealESRGAN_x4plus_anime_6B(self):
        SharedModelManager.manager.load("RealESRGAN_x4plus_anime_6B")
        assert (
            SharedModelManager.manager.esrgan.is_model_loaded(
                "RealESRGAN_x4plus_anime_6B"
            )
            is True
        )
        data = {
            "model": "RealESRGAN_x4plus_anime_6B",
            "source_image": self.image,
        }
        pil_image = self.horde.image_upscale(data)
        assert pil_image is not None
        width, height = pil_image.size
        # assert width == self.width * 4
        # assert height == self.height * 4
        pil_image.save(
            "images/horde_image_upscale_RealESRGAN_x4plus_anime_6B.webp", quality=90
        )

    def test_image_upscale_4x_AnimeSharp(self):
        SharedModelManager.manager.load("4x_AnimeSharp")
        assert (
            SharedModelManager.manager.esrgan.is_model_loaded("4x_AnimeSharp") is True
        )
        data = {
            "model": "4x_AnimeSharp",
            "source_image": self.image,
        }
        pil_image = self.horde.image_upscale(data)
        assert pil_image is not None
        width, height = pil_image.size
        # assert width == self.width * 4
        # assert height == self.height * 4
        pil_image.save("images/horde_image_upscale_4x_AnimeSharp.webp", quality=90)
