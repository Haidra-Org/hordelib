# test_horde_generate.py
"""The typed `HordeLib.generate` entry point, end to end on a GPU.

The dict-based tests pin the compatibility shim; these pin the typed front door: SDK
generic parameters in, explicit pipeline selection, images out.
"""

import uuid

import pytest
from horde_sdk.generation_parameters.image import (
    BasicImageGenerationParameters,
    HiresFixGenerationParameters,
    ImageGenerationParameters,
)
from horde_sdk.generation_parameters.image.object_models import ImageGenerationComponentContainer
from PIL import Image

from hordelib.horde import HordeLib
from hordelib.pipeline.identifiers import AUTO_PIPELINE, ImagePipeline

from .testing_shared_functions import check_single_inference_image_similarity


def _base_params(model: str, **overrides) -> BasicImageGenerationParameters:
    fields = {
        "model": model,
        "prompt": "an ancient llamia monster",
        "seed": "123456789",
        "width": 512,
        "height": 512,
        "steps": 25,
        "cfg_scale": 7.5,
        "sampler_name": "k_dpmpp_2m",
        "scheduler": "normal",
        "clip_skip": 1,
        "denoising_strength": 1.0,
    }
    fields.update(overrides)
    return BasicImageGenerationParameters(**fields)


class TestHordeGenerate:
    @pytest.mark.default_sd15_model
    def test_generate_auto_matches_dict_path(
        self,
        hordelib_instance: HordeLib,
        stable_diffusion_model_name_for_testing: str,
    ):
        """A typed txt2img generation renders the same image as the dict path's pinned output."""
        params = ImageGenerationParameters(
            result_ids=[uuid.uuid4()],
            batch_size=1,
            source_processing="txt2img",
            base_params=_base_params(stable_diffusion_model_name_for_testing),
        )

        assert hordelib_instance.select_pipeline(params) == ImagePipeline.STABLE_DIFFUSION

        results = hordelib_instance.generate(params, pipeline=AUTO_PIPELINE)

        assert len(results) == 1
        pil_image = results[0].image
        assert isinstance(pil_image, Image.Image)

        img_filename = "typed_generate_text_to_image.png"
        pil_image.save(f"images/{img_filename}", quality=100)

        # The typed path must render the same image the dict path pinned for these parameters.
        assert check_single_inference_image_similarity(
            "images_expected/text_to_image.png",
            pil_image,
        )

    @pytest.mark.default_sd15_model
    def test_generate_explicit_hires_fix_pipeline(
        self,
        hordelib_instance: HordeLib,
        stable_diffusion_model_name_for_testing: str,
    ):
        """An explicit hires-fix pipeline choice with SDK-computed two-pass values renders."""
        first_pass = _base_params(stable_diffusion_model_name_for_testing, width=512, height=512)
        second_pass = _base_params(
            stable_diffusion_model_name_for_testing,
            width=1024,
            height=1024,
            steps=13,
            denoising_strength=0.65,
        )
        params = ImageGenerationParameters(
            result_ids=[uuid.uuid4()],
            batch_size=1,
            source_processing="txt2img",
            base_params=_base_params(stable_diffusion_model_name_for_testing, width=1024, height=1024),
            additional_params=ImageGenerationComponentContainer(
                components=[HiresFixGenerationParameters(first_pass=first_pass, second_pass=second_pass)],
            ),
        )

        assert hordelib_instance.select_pipeline(params) == ImagePipeline.STABLE_DIFFUSION_HIRES_FIX

        results = hordelib_instance.generate(
            params,
            pipeline=ImagePipeline.STABLE_DIFFUSION_HIRES_FIX,
        )

        assert len(results) == 1
        pil_image = results[0].image
        assert isinstance(pil_image, Image.Image)
        assert pil_image.size == (1024, 1024)

        pil_image.save("images/typed_generate_hires_fix.png", quality=100)
