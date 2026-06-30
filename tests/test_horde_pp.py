# test_horde.py
import os
import typing
from collections.abc import Generator

import PIL.Image
import pytest
from horde_model_reference import (
    MODEL_REFERENCE_CATEGORY,
    PENDING_SOURCE_ID,
    ModelReferenceManager,
    PendingModelProvider,
)
from PIL import Image

from hordelib.beta_models import BETA_CATEGORIES_ENV_VAR
from hordelib.horde import HordeLib, ResultingImageReturn
from hordelib.shared_model_manager import SharedModelManager
from hordelib.utils.distance import HistogramDistanceResultCode

from .testing_shared_functions import (
    CosineSimilarityResultCode,
    ImageSimilarityConstraints,
    check_image_similarity_pytest,
)

# The modern upscalers and face restorers are distributed as *beta* models: they live in the
# horde-model-reference PRIMARY service's pending queue (https://models.aihorde.net) rather than the
# canonical reference, exactly as a worker consumes them. These tests therefore opt the esrgan/gfpgan
# categories into beta (the same BETA_CATEGORIES_ENV_VAR a worker sets) and let the pending provider
# resolve them, instead of duplicating the records here. The names below are the pending-queue entries
# to exercise, paired with their expected upscale factor (1.0 for face restorers).
_BETA_UPSCALER_SCALE_FACTORS: dict[str, float] = {
    "4xNomos8kSC": 4.0,
    "4xLSDIRplus": 4.0,
    "4xNomosWebPhoto_RealPLKSR": 4.0,
    "4xNomos2_realplksr_dysample": 4.0,
    "4xNomos2_hq_dat2": 4.0,
    "2xModernSpanimationV1": 2.0,
}
_BETA_FACEFIXERS: tuple[str, ...] = ("GFPGANv1.3", "RestoreFormer")
_BETA_PRIMARY_API_URL = "https://models.aihorde.net/api"


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

    @pytest.fixture(scope="class", autouse=True)
    def enable_pending_beta_models(
        self,
        shared_model_manager: type[SharedModelManager],
    ) -> Generator[None, None, None]:
        """Surface the beta upscalers/face restorers from the PRIMARY pending queue, as a worker does.

        Opts the esrgan/gfpgan categories into beta via the same env var a worker sets and registers
        the live :class:`PendingModelProvider` under the ``"pending"`` source, then reloads those two
        managers so their references include the pending-queue records. A failed/empty fetch degrades to
        the canonical reference (per ``beta_source_for``), and the per-model tests skip themselves, so a
        network outage never masks the canonical upscaler tests in this class. The canonical-only state
        is restored on teardown so the session-scoped managers do not leak beta records into other tests.
        """
        manager = shared_model_manager.manager
        assert manager is not None
        assert manager.esrgan is not None
        assert manager.gfpgan is not None

        previous_categories = os.environ.get(BETA_CATEGORIES_ENV_VAR)
        os.environ[BETA_CATEGORIES_ENV_VAR] = (
            f"{MODEL_REFERENCE_CATEGORY.esrgan.value},{MODEL_REFERENCE_CATEGORY.gfpgan.value}"
        )

        ref_manager = ModelReferenceManager.get_instance()
        ref_manager.register_provider(
            PendingModelProvider(
                primary_api_url=_BETA_PRIMARY_API_URL,
                apikey="0000000000",
                categories={MODEL_REFERENCE_CATEGORY.esrgan, MODEL_REFERENCE_CATEGORY.gfpgan},
            ),
            replace=True,
        )

        manager.esrgan.load_model_database()
        manager.gfpgan.load_model_database()
        try:
            yield
        finally:
            ref_manager.unregister_provider(PENDING_SOURCE_ID)
            if previous_categories is None:
                os.environ.pop(BETA_CATEGORIES_ENV_VAR, None)
            else:
                os.environ[BETA_CATEGORIES_ENV_VAR] = previous_categories
            manager.esrgan.load_model_database()
            manager.gfpgan.load_model_database()

    @classmethod
    def _skip_if_beta_model_absent(cls, model_name: str) -> None:
        """Skip the current test if the beta model did not resolve from the pending queue."""
        manager = cls.shared_model_manager.manager
        assert manager is not None and manager.esrgan is not None and manager.gfpgan is not None
        if model_name not in manager.esrgan.model_reference and model_name not in manager.gfpgan.model_reference:
            pytest.skip(f"Beta model {model_name!r} not available from the pending queue (service/network?)")

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
                histogram_fail_threshold=HistogramDistanceResultCode.VERY_SIMILAR_DISTRIBUTION,
                histogram_warn_threshold=HistogramDistanceResultCode.EXTREMELY_SIMILAR_DISTRIBUTION,
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

    @pytest.mark.parametrize(
        ("model_name", "expected_scale_factor"),
        list(_BETA_UPSCALER_SCALE_FACTORS.items()),
    )
    def test_image_upscale_beta(
        self,
        db0_test_image: PIL.Image.Image,
        model_name: str,
        expected_scale_factor: float,
    ):
        """The modern (beta/pending-queue) upscalers download, run, and upscale by their declared factor."""
        self._skip_if_beta_model_absent(model_name)
        self.post_processor_check(
            model_name=model_name,
            image_filename=f"image_upscale_{model_name}.png",
            target_image=db0_test_image,
            expected_scale_factor=expected_scale_factor,
            post_process_function=self.hordelib_instance.image_upscale,
        )

    def test_image_facefix_gfpgan_v1_3(self):
        """GFPGANv1.3 (beta/pending-queue) runs through the existing GFPGAN face-restore path."""
        self._skip_if_beta_model_absent("GFPGANv1.3")
        self.post_processor_check(
            model_name="GFPGANv1.3",
            image_filename="image_facefix_gfpgan_v1_3.png",
            target_image=Image.open("images/test_facefix.png"),
            expected_scale_factor=1.0,
            post_process_function=self.hordelib_instance.image_facefix,
        )

    def test_image_facefix_restoreformer(self):
        """RestoreFormer (beta/pending-queue) loads through spandrel and runs the face-restore graph."""
        self._skip_if_beta_model_absent("RestoreFormer")
        self.post_processor_check(
            model_name="RestoreFormer",
            image_filename="image_facefix_restoreformer.png",
            target_image=Image.open("images/test_facefix.png"),
            expected_scale_factor=1.0,
            post_process_function=self.hordelib_instance.image_facefix,
        )
