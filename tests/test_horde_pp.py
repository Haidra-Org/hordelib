# test_horde.py
import pytest
from PIL import Image

from hordelib.horde import HordeLib
from hordelib.shared_model_manager import SharedModelManager


class TestHordeUpscaling:
    @pytest.fixture(autouse=True, scope="class")
    def setup_and_teardown(self):
        TestHordeUpscaling.horde = HordeLib()

        TestHordeUpscaling.image = Image.open("images/test_db0.jpg")
        (
            TestHordeUpscaling.width,
            TestHordeUpscaling.height,
        ) = TestHordeUpscaling.image.size
        yield
        del TestHordeUpscaling.horde

    @pytest.fixture(autouse=True)
    def setup_model(self, request):
        mm_type = request.node.get_closest_marker("mm_model").args[0]
        print(mm_type)
        self.default_model_manager_args = {
            mm_type: True,
        }
        SharedModelManager.loadModelManagers(**self.default_model_manager_args)
        assert SharedModelManager.manager is not None
        yield
        SharedModelManager._instance = None
        SharedModelManager.manager = None

    @pytest.mark.mm_model("esrgan")
    def test_image_upscale_RealESRGAN_x4plus(self):
        SharedModelManager.manager.load("RealESRGAN_x4plus")
        assert SharedModelManager.manager.esrgan.is_model_loaded("RealESRGAN_x4plus") is True
        data = {
            "model": "RealESRGAN_x4plus",
            "source_image": self.image,
        }
        assert self.horde is not None
        pil_image = self.horde.image_upscale(data)
        assert pil_image is not None
        width, height = pil_image.size
        assert width == self.width * 4
        assert height == self.height * 4
        pil_image.save("images/horde_image_upscale_RealESRGAN_x4plus.webp", quality=90)

    @pytest.mark.mm_model("esrgan")
    def test_image_upscale_RealESRGAN_x2plus(self):
        SharedModelManager.manager.load("RealESRGAN_x2plus")
        assert SharedModelManager.manager.esrgan.is_model_loaded("RealESRGAN_x2plus") is True
        data = {
            "model": "RealESRGAN_x2plus",
            "source_image": self.image,
        }
        pil_image = self.horde.image_upscale(data)
        assert pil_image is not None
        width, height = pil_image.size
        assert width == self.width * 2
        assert height == self.height * 2
        pil_image.save("images/horde_image_upscale_RealESRGAN_x2plus.webp", quality=90)

    @pytest.mark.mm_model("esrgan")
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
        assert width == self.width * 4
        assert height == self.height * 4
        pil_image.save("images/horde_image_upscale_NMKD_Siax.webp", quality=90)

    @pytest.mark.mm_model("esrgan")
    def test_image_upscale_RealESRGAN_x4plus_anime_6B(self):
        SharedModelManager.manager.load("RealESRGAN_x4plus_anime_6B")
        assert (
            SharedModelManager.manager.esrgan.is_model_loaded(
                "RealESRGAN_x4plus_anime_6B",
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
        assert width == self.width * 4
        assert height == self.height * 4
        pil_image.save(
            "images/horde_image_upscale_RealESRGAN_x4plus_anime_6B.webp",
            quality=90,
        )

    @pytest.mark.mm_model("esrgan")
    def test_image_upscale_4x_AnimeSharp(self):
        SharedModelManager.manager.load("4x_AnimeSharp")
        assert SharedModelManager.manager.esrgan.is_model_loaded("4x_AnimeSharp") is True
        data = {
            "model": "4x_AnimeSharp",
            "source_image": self.image,
        }
        pil_image = self.horde.image_upscale(data)
        assert pil_image is not None
        width, height = pil_image.size
        assert width == self.width * 4
        assert height == self.height * 4
        pil_image.save("images/horde_image_upscale_4x_AnimeSharp.webp", quality=90)

    @pytest.mark.mm_model("codeformer")
    def test_image_facefix_codeformers(self):
        SharedModelManager.manager.load("CodeFormers")
        assert SharedModelManager.manager.codeformer.is_model_loaded("CodeFormers") is True
        data = {
            "model": "CodeFormers",
            "source_image": Image.open("images/test_facefix.png"),
        }
        pil_image = self.horde.image_facefix(data)
        assert pil_image is not None
        width, height = pil_image.size
        pil_image.save("images/horde_image_facefix_codeformers.webp", quality=90)

    @pytest.mark.mm_model("gfpgan")
    def test_image_facefix_gfpgan(self):
        SharedModelManager.manager.load("GFPGAN")
        assert SharedModelManager.manager.gfpgan.is_model_loaded("GFPGAN") is True
        data = {
            "model": "GFPGAN",
            "source_image": Image.open("images/test_facefix.png"),
        }
        pil_image = self.horde.image_facefix(data)
        assert pil_image is not None
        width, height = pil_image.size
        pil_image.save("images/horde_image_facefix_gfpgan.webp", quality=90)
