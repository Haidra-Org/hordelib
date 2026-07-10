"""GPU perceptual-identity matrix for the disaggregated stage path.

A full run through the stage entry points (``encode_text_stage`` -> optional ``vae_encode_stage`` ->
``sample_stage`` -> ``decode_stage``) must reproduce the image the monolithic ``generate`` renders for
the same parameters. The disaggregation cuts the same materialized family graph at the CONDITIONING and
LATENT boundaries, so at a fixed seed the two paths converge to the same pixels. This module exercises
that identity across the v1 feature surface (txt2img on SD1.5 and SDXL, SD1.5 img2img, hires-fix, LoRA,
and TI), so a regression in any stage-cut or injection node surfaces as a divergence rather than a
silently different render.

These are real-GPU tests: each is marked ``slow`` (deselected by the CI default ``-m "not slow"``) plus
the model marker its checkpoint needs. Run them manually and serially, for example::

    uv run --no-sync pytest tests/test_stage_disaggregation.py -m slow
"""

from __future__ import annotations

import tempfile
import uuid
from pathlib import Path

import pytest
from horde_sdk.generation_parameters.image import (
    BasicImageGenerationParameters,
    HiresFixGenerationParameters,
    Image2ImageGenerationParameters,
    ImageGenerationParameters,
    LoRaEntry,
    TIEntry,
)
from horde_sdk.generation_parameters.image.object_models import ImageGenerationComponentContainer
from PIL import Image

from hordelib.horde import HordeLib, ResultingImageReturn
from hordelib.pipeline.identifiers import AUTO_PIPELINE
from hordelib.shared_model_manager import SharedModelManager
from hordelib.utils.distance import CosineSimilarityResultCode, evaluate_image_distance


def _params(
    model: str,
    *,
    seed: str,
    width: int = 512,
    height: int = 512,
    prompt: str = "an ancient llamia monster",
    denoising_strength: float = 1.0,
    source_processing: str = "txt2img",
    clip_skip: int = 1,
    components: list | None = None,
) -> ImageGenerationParameters:
    base = BasicImageGenerationParameters(
        model=model,
        prompt=prompt,
        seed=seed,
        width=width,
        height=height,
        steps=25,
        cfg_scale=7.5,
        sampler_name="k_dpmpp_2m",
        scheduler="normal",
        clip_skip=clip_skip,
        denoising_strength=denoising_strength,
    )
    return ImageGenerationParameters(
        result_ids=[uuid.uuid4()],
        batch_size=1,
        source_processing=source_processing,
        base_params=base,
        additional_params=ImageGenerationComponentContainer(components=components or []),
    )


def _run_stage_pipeline(
    hordelib_instance: HordeLib,
    params: ImageGenerationParameters,
    *,
    img2img: bool = False,
) -> list[ResultingImageReturn]:
    """Drive the disaggregated stage path for ``params`` and return its decoded results.

    For img2img the source latent is produced by ``vae_encode_stage`` and injected into
    ``sample_stage`` (matching the image-lane split), rather than letting the sampler VAE-encode.
    """
    positive_bytes, negative_bytes = hordelib_instance.encode_text_stage(params)

    source_latent_bytes = hordelib_instance.vae_encode_stage(params) if img2img else None

    latent_bytes = hordelib_instance.sample_stage(
        params,
        positive_conditioning_bytes=positive_bytes,
        negative_conditioning_bytes=negative_bytes,
        source_latent_bytes=source_latent_bytes,
    )
    staged_results, _faults = hordelib_instance.decode_stage(params, latent_bytes=latent_bytes)
    return staged_results


def _assert_stage_identity(
    case_name: str,
    monolithic: list[ResultingImageReturn],
    staged: list[ResultingImageReturn],
) -> None:
    """Assert the stage path matched the monolithic render in image count and perceptual identity.

    On divergence, both images are written to a temp directory (so the mismatch can be inspected by
    eye) and the failure carries the cosine metric; thresholds are never relaxed to make a case pass.
    """
    assert len(staged) == len(monolithic), (
        f"{case_name}: stage path produced {len(staged)} image(s), monolithic produced {len(monolithic)}"
    )

    for index, (mono, stage) in enumerate(zip(monolithic, staged, strict=True)):
        assert isinstance(mono.image, Image.Image)
        assert isinstance(stage.image, Image.Image)

        cosine, _histogram = evaluate_image_distance(mono.image, stage.image)
        print(f"STAGE_IDENTITY {case_name} image {index}: {cosine}")
        if cosine.cosine_similarity < CosineSimilarityResultCode.PERCEPTUALLY_IDENTICAL:
            dump_dir = Path(tempfile.mkdtemp(prefix=f"stage_disagg_fail_{case_name}_"))
            mono.image.save(dump_dir / f"{case_name}_{index}_monolithic.png")
            stage.image.save(dump_dir / f"{case_name}_{index}_staged.png")
            raise AssertionError(
                f"{case_name} (image {index}): disaggregated stages diverged from the monolithic render: "
                f"{cosine}; images written to {dump_dir}",
            )


class TestStageDisaggregationPerceptualIdentity:
    @pytest.mark.slow
    @pytest.mark.default_sd15_model
    def test_txt2img_stages_match_monolithic(
        self,
        hordelib_instance: HordeLib,
        stable_diffusion_model_name_for_testing: str,
    ) -> None:
        params = _params(stable_diffusion_model_name_for_testing, seed="123456789")

        monolithic = hordelib_instance.generate(params, pipeline=AUTO_PIPELINE)
        assert len(monolithic) == 1

        staged = _run_stage_pipeline(hordelib_instance, params)

        _assert_stage_identity("sd15_txt2img", monolithic, staged)

    @pytest.mark.slow
    @pytest.mark.default_sd15_model
    def test_clip_skip_txt2img_stages_match_monolithic(
        self,
        hordelib_instance: HordeLib,
        stable_diffusion_model_name_for_testing: str,
    ) -> None:
        # clip_skip > 1 inserts a CLIPSetLastLayer between the loader and the encoders; the decode
        # stage disables the CLIP subset, so the cut must keep that node off the reused image output.
        params = _params(stable_diffusion_model_name_for_testing, seed="123456789", clip_skip=2)

        monolithic = hordelib_instance.generate(params, pipeline=AUTO_PIPELINE)
        assert len(monolithic) == 1

        staged = _run_stage_pipeline(hordelib_instance, params)

        _assert_stage_identity("sd15_clip_skip", monolithic, staged)

    @pytest.mark.slow
    @pytest.mark.default_sdxl_model
    def test_sdxl_txt2img_stages_match_monolithic(
        self,
        hordelib_instance: HordeLib,
        shared_model_manager: type[SharedModelManager],
        sdxl_1_0_base_model_name: str,
    ) -> None:
        assert shared_model_manager.manager.compvis is not None
        if sdxl_1_0_base_model_name not in shared_model_manager.manager.compvis.available_models:
            pytest.skip(f"{sdxl_1_0_base_model_name} checkpoint is not available on disk")

        params = _params(
            sdxl_1_0_base_model_name,
            seed="987654321",
            width=1024,
            height=1024,
        )

        monolithic = hordelib_instance.generate(params, pipeline=AUTO_PIPELINE)
        assert len(monolithic) == 1

        staged = _run_stage_pipeline(hordelib_instance, params)

        _assert_stage_identity("sdxl_txt2img", monolithic, staged)

    @pytest.mark.slow
    @pytest.mark.default_sd15_model
    def test_img2img_stages_match_monolithic(
        self,
        hordelib_instance: HordeLib,
        stable_diffusion_model_name_for_testing: str,
    ) -> None:
        source_image_bytes = Path("images/test_db0.jpg").read_bytes()
        params = _params(
            stable_diffusion_model_name_for_testing,
            seed="666",
            prompt="a dinosaur",
            denoising_strength=0.4,
            source_processing="img2img",
            components=[
                Image2ImageGenerationParameters(source_image=source_image_bytes, source_mask=None),
            ],
        )

        monolithic = hordelib_instance.generate(params, pipeline=AUTO_PIPELINE)
        assert len(monolithic) == 1

        staged = _run_stage_pipeline(hordelib_instance, params, img2img=True)

        _assert_stage_identity("sd15_img2img", monolithic, staged)

    @pytest.mark.slow
    @pytest.mark.default_sd15_model
    def test_hires_fix_txt2img_stages_match_monolithic(
        self,
        hordelib_instance: HordeLib,
        stable_diffusion_model_name_for_testing: str,
    ) -> None:
        prompt = "an ancient llamia monster"
        first_pass = BasicImageGenerationParameters(
            model=stable_diffusion_model_name_for_testing,
            prompt=prompt,
            width=512,
            height=512,
            steps=25,
            denoising_strength=1.0,
        )
        second_pass = BasicImageGenerationParameters(
            model=stable_diffusion_model_name_for_testing,
            prompt=prompt,
            width=768,
            height=768,
            steps=13,
            denoising_strength=0.65,
        )
        params = _params(
            stable_diffusion_model_name_for_testing,
            seed="123456789",
            prompt=prompt,
            components=[HiresFixGenerationParameters(first_pass=first_pass, second_pass=second_pass)],
        )

        monolithic = hordelib_instance.generate(params, pipeline=AUTO_PIPELINE)
        assert len(monolithic) == 1

        staged = _run_stage_pipeline(hordelib_instance, params)

        _assert_stage_identity("sd15_hires_fix", monolithic, staged)

    @pytest.mark.slow
    @pytest.mark.default_sd15_model
    def test_lora_txt2img_stages_match_monolithic(
        self,
        hordelib_instance: HordeLib,
        shared_model_manager: type[SharedModelManager],
        stable_diffusion_model_name_for_testing: str,
        lora_GlowingRunesAI: str,
    ) -> None:
        assert shared_model_manager.manager.lora is not None
        trigger = shared_model_manager.manager.lora.find_lora_trigger(lora_GlowingRunesAI, "red")

        params = _params(
            stable_diffusion_model_name_for_testing,
            seed="304886399544324",
            prompt=f"a dark magical crystal, {trigger}, 8K resolution",
            components=[
                LoRaEntry(
                    name=lora_GlowingRunesAI,
                    remote_version_id=None,
                    source="CIVITAI",
                    model_strength=1.0,
                    clip_strength=1.0,
                ),
            ],
        )

        monolithic = hordelib_instance.generate(params, pipeline=AUTO_PIPELINE)
        assert len(monolithic) == 1

        staged = _run_stage_pipeline(hordelib_instance, params)

        _assert_stage_identity("sd15_lora", monolithic, staged)

    @pytest.mark.slow
    @pytest.mark.default_sd15_model
    def test_ti_txt2img_stages_match_monolithic(
        self,
        hordelib_instance: HordeLib,
        shared_model_manager: type[SharedModelManager],
        stable_diffusion_model_name_for_testing: str,
    ) -> None:
        assert shared_model_manager.manager.ti is not None

        params = _params(
            stable_diffusion_model_name_for_testing,
            seed="1312",
            prompt="(embedding:7808:1.0), a dark magical crystal, 8K resolution",
            components=[
                TIEntry(
                    name="7808",
                    remote_version_id=None,
                    source="HORDELING",
                    model_strength=1.0,
                ),
            ],
        )

        monolithic = hordelib_instance.generate(params, pipeline=AUTO_PIPELINE)
        assert len(monolithic) == 1

        staged = _run_stage_pipeline(hordelib_instance, params)

        _assert_stage_identity("sd15_ti", monolithic, staged)
