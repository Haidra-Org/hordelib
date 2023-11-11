# test_horde.py
import typing

import PIL.Image
import pytest
from PIL import Image

from hordelib.horde import HordeLib, ResultingImageReturn
from hordelib.shared_model_manager import SharedModelManager
from hordelib.utils.distance import HistogramDistanceResultCode

from .testing_shared_functions import (
    CosineSimilarityResultCode,
    ImageSimilarityConstraints,
    check_image_similarity_pytest,
)


class TestHordeUpscaling:
    # @pytest.fixture(scope="class")
    # def image_distance_threshold(self) -> int:
    #     return int(os.getenv("IMAGE_DISTANCE_THRESHOLD", "100000"))

    shared_model_manager: type[SharedModelManager]
    hordelib_instance: HordeLib
    db0_test_image: PIL.Image.Image
    real_image: PIL.Image.Image

    @pytest.fixture(scope="class", autouse=True)
    def upscale_setup_and_teardown(
        self,
        shared_model_manager: type[SharedModelManager],
        hordelib_instance: HordeLib,
        db0_test_image: PIL.Image.Image,
        real_image: PIL.Image.Image,
    ):
        TestHordeUpscaling.shared_model_manager = shared_model_manager
        TestHordeUpscaling.hordelib_instance = hordelib_instance
        TestHordeUpscaling.db0_test_image = db0_test_image
        TestHordeUpscaling.real_image = real_image

    @staticmethod
    def is_upscaled_to_correct_scale(
        *,
        source_image: PIL.Image.Image,
        upscaled_image: PIL.Image.Image,
        factor: float,
    ) -> bool:
        width, height = source_image.size
        upscaled_width, upscaled_height = upscaled_image.size
        return upscaled_width == width * factor and upscaled_height == height * factor

    @classmethod
    def post_processor_check(
        cls,
        *,
        model_name: str,
        image_filename: str,
        target_image: PIL.Image.Image,
        expected_scale_factor: float,
        custom_data: dict | None = None,
        post_process_function: typing.Callable[[dict], ResultingImageReturn | None],
        similarity_constraints: ImageSimilarityConstraints | None = None,
    ):
        if similarity_constraints is None:
            similarity_constraints = ImageSimilarityConstraints(
                cosine_fail_floor=CosineSimilarityResultCode.PERCEPTUALLY_IDENTICAL,
                cosine_warn_floor=CosineSimilarityResultCode.EXTREMELY_SIMILAR,
                histogram_fail_threshold=HistogramDistanceResultCode.VERY_DISSIMILAR_DISTRIBUTION,
                histogram_warn_threshold=HistogramDistanceResultCode.SIMILAR_DISTRIBUTION,
            )
        assert cls.shared_model_manager.manager.download_model(model_name)
        assert cls.shared_model_manager.manager.is_model_available(model_name) is True

        data: dict = (
            custom_data
            if custom_data
            else {
                "model": model_name,
                "source_image": target_image,
            }
        )
        image_ret = post_process_function(data)
        assert isinstance(image_ret, ResultingImageReturn)
        pil_image = image_ret.image
        assert pil_image is not None
        pil_image.save(f"images/{image_filename}", quality=100)

        assert cls.is_upscaled_to_correct_scale(
            source_image=target_image,
            upscaled_image=pil_image,
            factor=expected_scale_factor,
        )

        assert check_image_similarity_pytest(
            f"images_expected/{image_filename}",
            pil_image,
            similarity_constraints=similarity_constraints,
        )

    def test_image_upscale_RealESRGAN_x4plus(self, db0_test_image: PIL.Image.Image):
        self.post_processor_check(
            model_name="RealESRGAN_x4plus",
            image_filename="image_upscale_RealESRGAN_x4plus.png",
            target_image=db0_test_image,
            expected_scale_factor=4.0,
            post_process_function=self.hordelib_instance.image_upscale,
        )

    def test_image_upscale_RealESRGAN_x2plus(
        self,
        db0_test_image: PIL.Image.Image,
    ):
        self.post_processor_check(
            model_name="RealESRGAN_x2plus",
            image_filename="image_upscale_RealESRGAN_x2plus.png",
            target_image=db0_test_image,
            expected_scale_factor=2.0,
            post_process_function=self.hordelib_instance.image_upscale,
        )

    def test_image_upscale_NMKD_Siax(self, db0_test_image: PIL.Image.Image):
        self.post_processor_check(
            model_name="NMKD_Siax",
            image_filename="image_upscale_NMKD_Siax.png",
            target_image=db0_test_image,
            expected_scale_factor=4.0,
            post_process_function=self.hordelib_instance.image_upscale,
        )

    def test_image_upscale_NMKD_Siax_resize(self, real_image: PIL.Image.Image):
        real_image_width, real_image_height = real_image.size
        scale_factor = 2.5
        scaled_image_width = int(real_image_width * scale_factor)
        scaled_image_height = int(real_image_height * scale_factor)
        assert scaled_image_width % 64 == 0
        assert scaled_image_height % 64 == 0

        self.post_processor_check(
            model_name="NMKD_Siax",
            image_filename="image_upscale_NMKD_Siax_resize.png",
            target_image=real_image,
            expected_scale_factor=scale_factor,
            custom_data={
                "model": "NMKD_Siax",
                "source_image": real_image,
                "width": scaled_image_width,
                "height": scaled_image_height,
            },
            post_process_function=self.hordelib_instance.image_upscale,
        )

    def test_image_upscale_RealESRGAN_x4plus_anime_6B(self, db0_test_image: PIL.Image.Image):
        similarity_constraints = ImageSimilarityConstraints(
            cosine_fail_floor=CosineSimilarityResultCode.PARTIALLY_SIMILAR,
            cosine_warn_floor=CosineSimilarityResultCode.CONSIDERABLY_SIMILAR,
            histogram_fail_threshold=HistogramDistanceResultCode.VERY_DISSIMILAR_DISTRIBUTION,
            histogram_warn_threshold=HistogramDistanceResultCode.SIMILAR_DISTRIBUTION,
        )
        # This model has been shown to vary its results between machines, so we loosen up the similarity constraints.

        self.post_processor_check(
            model_name="RealESRGAN_x4plus_anime_6B",
            image_filename="image_upscale_RealESRGAN_x4plus_anime_6B.png",
            target_image=db0_test_image,
            expected_scale_factor=4.0,
            post_process_function=self.hordelib_instance.image_upscale,
            similarity_constraints=similarity_constraints,
        )

    def test_image_upscale_4x_AnimeSharp(self, db0_test_image: PIL.Image.Image):
        self.post_processor_check(
            model_name="4x_AnimeSharp",
            image_filename="image_upscale_4x_AnimeSharp.png",
            target_image=db0_test_image,
            expected_scale_factor=4.0,
            post_process_function=self.hordelib_instance.image_upscale,
        )

    def test_image_facefix_codeformers(self):
        self.post_processor_check(
            model_name="CodeFormers",
            image_filename="image_facefix_codeformers.png",
            target_image=Image.open("images/test_facefix.png"),
            expected_scale_factor=1.0,
            post_process_function=self.hordelib_instance.image_facefix,
        )

    def test_image_facefix_gfpgan(self):
        self.post_processor_check(
            model_name="GFPGAN",
            image_filename="image_facefix_gfpgan.png",
            target_image=Image.open("images/test_facefix.png"),
            expected_scale_factor=1.0,
            post_process_function=self.hordelib_instance.image_facefix,
        )
