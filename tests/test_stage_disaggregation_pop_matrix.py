"""GPU perceptual-identity matrix over converter-produced parameters from production-shaped job pops.

Each case builds a real :class:`ImageGenerateJobPopResponse` in the API shape a worker lane receives,
converts it with :func:`convert_image_job_pop_response_to_parameters` exactly as the lane processes do,
and then renders it two ways from the SAME converter-produced parameters: the monolithic ``generate``
and the disaggregated stage path (``encode_text_stage`` -> optional ``vae_encode_stage`` ->
``sample_stage`` -> ``decode_stage``). At a fixed seed the two paths converge to the same pixels, so a
discriminator hiding in the converter seam or in a stage cut surfaces as a per-image divergence rather
than a silently different render. The identity bar is perceptual identity and is never relaxed.

The matrix spans the production feature surface: the production ``img2img``-labelled-without-source
shape that resolves to txt2img, sampler and CLIP knobs, genuine img2img with a source latent, hires
fix, LoRA, TI, and batch (with a transposition cross-check), on SD1.5 and, where the checkpoint is
available, SDXL.

Two non-matrix tests ride alongside: a CPU converter-determinism check (repeated conversion of one
response must serialize identically, with the divergence from the sibling hand-built parameters logged
for forensics) and a long-lived-lane interleave (decode job A, post-process through the same instance,
decode job B) that both decodes must still match.

The stage entry points cold-load their subset from disk only when the loader's in-RAM cache is empty;
this module clears ``SharedModelManager.manager._models_in_ram`` before each stage to force the
per-process production path in-process. On divergence both images are written under ``_DUMP_ROOT`` with
case-descriptive names and the cosine plus histogram metrics ride on the assertion message.

These are real-GPU tests, marked ``slow`` plus the checkpoint's model marker, deselected by the CI
default ``-m "not slow"`` (the converter-determinism case is CPU-only and unmarked). Run manually and
serially, for example::

    uv run --no-sync pytest tests/test_stage_disaggregation_pop_matrix.py -m slow
"""

from __future__ import annotations

import base64
import io
import uuid
from pathlib import Path

import pytest
from horde_model_reference import ModelReferenceManager, PrefetchStrategy
from horde_sdk.ai_horde_api.apimodels import (
    ImageGenerateJobPopPayload,
    ImageGenerateJobPopResponse,
    ImageGenerateJobPopSkippedStatus,
)
from horde_sdk.ai_horde_api.apimodels.base import LorasPayloadEntry, TIPayloadEntry
from horde_sdk.generation_parameters.image import (
    HiresFixGenerationParameters,
    Image2ImageGenerationParameters,
    ImageGenerationParameters,
)
from horde_sdk.generation_parameters.image.consts import KNOWN_IMAGE_SAMPLERS, KNOWN_IMAGE_SOURCE_PROCESSING
from horde_sdk.worker.dispatch.ai_horde.image.convert import convert_image_job_pop_response_to_parameters
from PIL import Image

from hordelib.horde import HordeLib, ResultingImageReturn
from hordelib.pipeline.identifiers import AUTO_PIPELINE
from hordelib.pipeline.payload_pp import UpscalePayload
from hordelib.shared_model_manager import SharedModelManager
from hordelib.utils.distance import CosineSimilarityResultCode, evaluate_image_distance

from .test_stage_disaggregation import _params as _handbuilt_params

_DUMP_ROOT = Path(
    r"G:\_temp_\claude\G--mxd-ai-python-local-dev-git-repos-tazlin-horde-worker-reGen"
    r"\fa65ea7e-1f5d-4a0b-b681-7c4f696bf546\scratchpad\pop_matrix",
)

_UPSCALE_MODEL = "RealESRGAN_x4plus"


def _offline_reference_manager() -> ModelReferenceManager:
    """Return a read-only reference manager for the converter, mirroring the worker subprocess helper.

    Reuses an already-initialized singleton (present under the hordelib session fixtures) rather than
    resetting it, so the shared model managers are left undisturbed; only when none exists is an
    offline, never-download instance created.
    """
    if ModelReferenceManager.has_instance():
        return ModelReferenceManager.get_instance()
    return ModelReferenceManager(offline=True, prefetch_strategy=PrefetchStrategy.NONE)


def _gradient_png_base64(width: int, height: int) -> str:
    """Return a base64-encoded PNG of a smooth RGB gradient, the inline source-image shape a pop carries."""
    image = Image.new("RGB", (width, height))
    pixels = image.load()
    assert pixels is not None
    for y in range(height):
        for x in range(width):
            pixels[x, y] = (int(255 * x / max(1, width - 1)), int(255 * y / max(1, height - 1)), 128)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _job_pop_response(
    model: str,
    *,
    seed: str,
    n_iter: int = 1,
    width: int = 512,
    height: int = 512,
    prompt: str = "an ancient llamia monster",
    sampler_name: str = KNOWN_IMAGE_SAMPLERS.k_dpmpp_2m,
    karras: bool = False,
    clip_skip: int = 1,
    denoising_strength: float | None = None,
    source_processing: str = KNOWN_IMAGE_SOURCE_PROCESSING.txt2img,
    source_image: str | None = None,
    hires_fix: bool = False,
    transparent: bool | None = None,
    loras: list[LorasPayloadEntry] | None = None,
    tis: list[TIPayloadEntry] | None = None,
) -> ImageGenerateJobPopResponse:
    """Build a realistic job pop response, the API shape the converter consumes in each lane process."""
    ids = [uuid.uuid4() for _ in range(n_iter)]
    return ImageGenerateJobPopResponse(
        ids=ids,
        payload=ImageGenerateJobPopPayload(
            prompt=prompt,
            seed=seed,
            width=width,
            height=height,
            ddim_steps=25,
            cfg_scale=7.5,
            sampler_name=sampler_name,
            karras=karras,
            clip_skip=clip_skip,
            denoising_strength=denoising_strength,
            hires_fix=hires_fix,
            transparent=transparent,
            loras=loras,
            tis=tis,
            n_iter=n_iter,
        ),
        skipped=ImageGenerateJobPopSkippedStatus(),
        model=model,
        source_processing=source_processing,
        source_image=source_image,
        r2_uploads=[f"https://not.a.real.url.internal/upload/{id_}" for id_ in ids],
    )


def _converter_params(
    response: ImageGenerateJobPopResponse,
    reference_manager: ModelReferenceManager,
) -> ImageGenerationParameters:
    """Convert a job pop response to generation parameters exactly as the lane processes do."""
    return convert_image_job_pop_response_to_parameters(
        api_response=response,
        model_reference_manager=reference_manager,
    ).generation_parameters


def _has_component(params: ImageGenerationParameters, component_type: type) -> bool:
    """Whether the converter attached a component of the given type to the parameters."""
    return any(isinstance(component, component_type) for component in params.additional_params.components)


def _clear_loader_cache() -> None:
    """Empty the loader's in-RAM component cache so the next stage cold-loads its subset from disk."""
    SharedModelManager.manager._models_in_ram = {}


def _cold_encode(hordelib_instance: HordeLib, params: ImageGenerationParameters) -> tuple[bytes, bytes]:
    """Cold-load the CLIP subset and return the (positive, negative) conditioning blobs."""
    _clear_loader_cache()
    return hordelib_instance.encode_text_stage(params)


def _cold_vae_encode(hordelib_instance: HordeLib, params: ImageGenerationParameters) -> bytes:
    """Cold-load the VAE subset and return the source latent blob for the img2img sample injection."""
    _clear_loader_cache()
    return hordelib_instance.vae_encode_stage(params)


def _cold_sample(
    hordelib_instance: HordeLib,
    params: ImageGenerationParameters,
    positive_bytes: bytes,
    negative_bytes: bytes,
    source_latent_bytes: bytes | None,
) -> bytes:
    """Cold-load the UNet subset and return the sampled latent blob."""
    _clear_loader_cache()
    return hordelib_instance.sample_stage(
        params,
        positive_conditioning_bytes=positive_bytes,
        negative_conditioning_bytes=negative_bytes,
        source_latent_bytes=source_latent_bytes,
    )


def _cold_decode(
    hordelib_instance: HordeLib,
    params: ImageGenerationParameters,
    latent_bytes: bytes,
) -> list[ResultingImageReturn]:
    """Cold-load the VAE subset and decode the injected latent to images."""
    _clear_loader_cache()
    results, _faults = hordelib_instance.decode_stage(params, latent_bytes=latent_bytes)
    return results


def _run_cold_pipeline(
    hordelib_instance: HordeLib,
    params: ImageGenerationParameters,
    *,
    img2img: bool = False,
) -> list[ResultingImageReturn]:
    """Drive the disaggregated stage path with every stage cold-loading its subset from disk.

    For img2img the source latent is produced by ``vae_encode_stage`` and injected into
    ``sample_stage`` (matching the image-lane split), rather than letting the sampler VAE-encode.
    """
    positive_bytes, negative_bytes = _cold_encode(hordelib_instance, params)
    source_latent_bytes = _cold_vae_encode(hordelib_instance, params) if img2img else None
    latent_bytes = _cold_sample(hordelib_instance, params, positive_bytes, negative_bytes, source_latent_bytes)
    return _cold_decode(hordelib_instance, params, latent_bytes)


def _flatten_to_rgb(image: Image.Image) -> Image.Image:
    """Composite an image onto opaque white and return RGB, so an RGBA render compares over three channels.

    The similarity metric converts through a three-channel colour space; a transparent (RGBA) render is
    flattened against white to a legible RGB reference before comparison.
    """
    if image.mode == "RGBA":
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[-1])
        return background
    return image.convert("RGB")


def _assert_identity(
    case_name: str,
    monolithic: list[ResultingImageReturn],
    staged: list[ResultingImageReturn],
) -> None:
    """Assert the stage path matched the monolithic render in image count and per-index perceptual identity.

    On divergence both images are written under ``_DUMP_ROOT`` with case-descriptive names and the
    failure carries the cosine and histogram metrics; the identity bar is never relaxed.
    """
    assert len(staged) == len(
        monolithic,
    ), f"{case_name}: stage path produced {len(staged)} image(s), monolithic produced {len(monolithic)}"

    for index, (mono, stage) in enumerate(zip(monolithic, staged, strict=True)):
        assert isinstance(mono.image, Image.Image)
        assert isinstance(stage.image, Image.Image)

        cosine, histogram = evaluate_image_distance(mono.image, stage.image)
        print(f"POP_MATRIX_IDENTITY {case_name} image {index}: {cosine} | {histogram}")
        if cosine.cosine_similarity < CosineSimilarityResultCode.PERCEPTUALLY_IDENTICAL:
            _DUMP_ROOT.mkdir(parents=True, exist_ok=True)
            mono_path = _DUMP_ROOT / f"{case_name}_{index}_monolithic.png"
            stage_path = _DUMP_ROOT / f"{case_name}_{index}_disagg.png"
            mono.image.save(mono_path)
            stage.image.save(stage_path)
            raise AssertionError(
                f"{case_name} (image {index}): converter-parameter disaggregated stages diverged from the "
                f"monolithic render: {cosine}; {histogram}; monolithic written to {mono_path}, "
                f"disaggregated written to {stage_path}",
            )


def _json_diff(converter: object, handbuilt: object, path: str = "") -> list[str]:
    """Return human-readable lines describing where two JSON-ready structures differ, for forensics."""
    lines: list[str] = []
    if isinstance(converter, dict) and isinstance(handbuilt, dict):
        for key in sorted(set(converter) | set(handbuilt)):
            child_path = f"{path}.{key}" if path else str(key)
            if key not in converter:
                lines.append(f"{child_path}: converter=<absent> handbuilt={handbuilt[key]!r}")
            elif key not in handbuilt:
                lines.append(f"{child_path}: converter={converter[key]!r} handbuilt=<absent>")
            else:
                lines.extend(_json_diff(converter[key], handbuilt[key], child_path))
    elif isinstance(converter, list) and isinstance(handbuilt, list):
        if len(converter) != len(handbuilt):
            lines.append(f"{path}: converter len={len(converter)} handbuilt len={len(handbuilt)}")
        for index in range(min(len(converter), len(handbuilt))):
            lines.extend(_json_diff(converter[index], handbuilt[index], f"{path}[{index}]"))
    elif converter != handbuilt:
        lines.append(f"{path}: converter={converter!r} handbuilt={handbuilt!r}")
    return lines


_SD15_TXT2IMG_KNOB_CASES = [
    pytest.param(
        "sd15_production_shape_txt2img",
        {"source_processing": KNOWN_IMAGE_SOURCE_PROCESSING.img2img},
        id="production_shape",
    ),
    pytest.param(
        "sd15_karras_clip_skip",
        {"karras": True, "clip_skip": 2},
        id="karras_clip_skip",
    ),
]


class TestPopMatrixStageDisaggregation:
    @pytest.mark.slow
    @pytest.mark.default_sd15_model
    @pytest.mark.parametrize(("case_name", "pop_kwargs"), _SD15_TXT2IMG_KNOB_CASES)
    def test_sd15_txt2img_knob_variants(
        self,
        hordelib_instance: HordeLib,
        stable_diffusion_model_name_for_testing: str,
        case_name: str,
        pop_kwargs: dict,
    ) -> None:
        reference_manager = _offline_reference_manager()
        response = _job_pop_response(stable_diffusion_model_name_for_testing, seed="123456789", **pop_kwargs)
        params = _converter_params(response, reference_manager)

        # The production shape (source_processing=img2img with no source) must resolve to txt2img with no
        # img2img source component, so the stage run takes the txt2img path with no source latent.
        assert params.source_processing == KNOWN_IMAGE_SOURCE_PROCESSING.txt2img
        assert not _has_component(params, Image2ImageGenerationParameters)
        assert params.batch_size == 1

        monolithic = hordelib_instance.generate(params, pipeline=AUTO_PIPELINE)
        assert len(monolithic) == 1

        staged = _run_cold_pipeline(hordelib_instance, params)

        _assert_identity(case_name, monolithic, staged)

    @pytest.mark.slow
    @pytest.mark.default_sd15_model
    def test_sd15_img2img_stages_match_monolithic(
        self,
        hordelib_instance: HordeLib,
        stable_diffusion_model_name_for_testing: str,
    ) -> None:
        response = _job_pop_response(
            stable_diffusion_model_name_for_testing,
            seed="666",
            prompt="a dinosaur",
            denoising_strength=0.6,
            source_processing=KNOWN_IMAGE_SOURCE_PROCESSING.img2img,
            source_image=_gradient_png_base64(512, 512),
        )
        params = _converter_params(response, _offline_reference_manager())
        assert params.source_processing == KNOWN_IMAGE_SOURCE_PROCESSING.img2img
        assert _has_component(params, Image2ImageGenerationParameters)

        monolithic = hordelib_instance.generate(params, pipeline=AUTO_PIPELINE)
        assert len(monolithic) == 1

        staged = _run_cold_pipeline(hordelib_instance, params, img2img=True)

        _assert_identity("sd15_img2img", monolithic, staged)

    @pytest.mark.slow
    @pytest.mark.default_sd15_model
    def test_sd15_hires_fix_stages_match_monolithic(
        self,
        hordelib_instance: HordeLib,
        stable_diffusion_model_name_for_testing: str,
    ) -> None:
        # Hires fix only applies when the target resolution exceeds the SD1.5 native resolution, so the
        # pop requests 768x768; the converter must then attach the second-pass parameters.
        response = _job_pop_response(
            stable_diffusion_model_name_for_testing,
            seed="123456789",
            width=768,
            height=768,
            hires_fix=True,
        )
        params = _converter_params(response, _offline_reference_manager())
        assert _has_component(params, HiresFixGenerationParameters)

        monolithic = hordelib_instance.generate(params, pipeline=AUTO_PIPELINE)
        assert len(monolithic) == 1

        staged = _run_cold_pipeline(hordelib_instance, params)

        _assert_identity("sd15_hires_fix", monolithic, staged)

    @pytest.mark.slow
    @pytest.mark.default_sd15_model
    def test_sd15_lora_stages_match_monolithic(
        self,
        hordelib_instance: HordeLib,
        shared_model_manager: type[SharedModelManager],
        stable_diffusion_model_name_for_testing: str,
        lora_GlowingRunesAI: str,
    ) -> None:
        assert shared_model_manager.manager.lora is not None
        trigger = shared_model_manager.manager.lora.find_lora_trigger(lora_GlowingRunesAI, "red")

        response = _job_pop_response(
            stable_diffusion_model_name_for_testing,
            seed="304886399544324",
            prompt=f"a dark magical crystal, {trigger}, 8K resolution",
            loras=[LorasPayloadEntry(name=lora_GlowingRunesAI, model=1.0, clip=1.0)],
        )
        params = _converter_params(response, _offline_reference_manager())

        monolithic = hordelib_instance.generate(params, pipeline=AUTO_PIPELINE)
        assert len(monolithic) == 1

        staged = _run_cold_pipeline(hordelib_instance, params)

        _assert_identity("sd15_lora", monolithic, staged)

    @pytest.mark.slow
    @pytest.mark.default_sd15_model
    def test_sd15_ti_stages_match_monolithic(
        self,
        hordelib_instance: HordeLib,
        shared_model_manager: type[SharedModelManager],
        stable_diffusion_model_name_for_testing: str,
    ) -> None:
        assert shared_model_manager.manager.ti is not None

        response = _job_pop_response(
            stable_diffusion_model_name_for_testing,
            seed="1312",
            prompt="(embedding:7808:1.0), a dark magical crystal, 8K resolution",
            tis=[TIPayloadEntry(name="7808", strength=1.0)],
        )
        params = _converter_params(response, _offline_reference_manager())

        monolithic = hordelib_instance.generate(params, pipeline=AUTO_PIPELINE)
        assert len(monolithic) == 1

        staged = _run_cold_pipeline(hordelib_instance, params)

        _assert_identity("sd15_ti", monolithic, staged)

    @pytest.mark.slow
    @pytest.mark.default_sd15_model
    def test_sd15_batch_stages_match_monolithic(
        self,
        hordelib_instance: HordeLib,
        stable_diffusion_model_name_for_testing: str,
    ) -> None:
        response = _job_pop_response(stable_diffusion_model_name_for_testing, seed="123456789", n_iter=2)
        params = _converter_params(response, _offline_reference_manager())
        assert params.batch_size == 2

        monolithic = hordelib_instance.generate(params, pipeline=AUTO_PIPELINE)
        assert len(monolithic) == 2

        staged = _run_cold_pipeline(hordelib_instance, params)
        assert len(staged) == 2, f"batch stage path produced {len(staged)} image(s), expected 2"

        # A transposition (staged image order swapped relative to the monolithic batch) is a distinct
        # failure from per-image corruption; measure the off-diagonal similarity so a swap is legible.
        cross_cosine, _cross_histogram = evaluate_image_distance(monolithic[1].image, staged[0].image)
        print(f"POP_MATRIX_BATCH transposition cross-check staged[0] vs monolithic[1]: {cross_cosine}")

        for index, (mono, stage) in enumerate(zip(monolithic, staged, strict=True)):
            assert isinstance(mono.image, Image.Image)
            assert isinstance(stage.image, Image.Image)

            cosine, histogram = evaluate_image_distance(mono.image, stage.image)
            print(f"POP_MATRIX_IDENTITY sd15_batch image {index}: {cosine} | {histogram}")
            if cosine.cosine_similarity < CosineSimilarityResultCode.PERCEPTUALLY_IDENTICAL:
                _DUMP_ROOT.mkdir(parents=True, exist_ok=True)
                mono_path = _DUMP_ROOT / f"sd15_batch_{index}_monolithic.png"
                stage_path = _DUMP_ROOT / f"sd15_batch_{index}_disagg.png"
                mono.image.save(mono_path)
                stage.image.save(stage_path)
                raise AssertionError(
                    f"sd15_batch (image {index}): converter-parameter disaggregated stages diverged from "
                    f"the monolithic render: {cosine}; {histogram}; transposition cross-check staged[0] vs "
                    f"monolithic[1]: {cross_cosine}; monolithic written to {mono_path}, disaggregated "
                    f"written to {stage_path}",
                )

    @pytest.mark.slow
    @pytest.mark.default_sdxl_model
    def test_sdxl_production_shape_txt2img_stages_match_monolithic(
        self,
        hordelib_instance: HordeLib,
        shared_model_manager: type[SharedModelManager],
        sdxl_1_0_base_model_name: str,
    ) -> None:
        assert shared_model_manager.manager.compvis is not None
        if sdxl_1_0_base_model_name not in shared_model_manager.manager.compvis.available_models:
            pytest.skip(f"{sdxl_1_0_base_model_name} checkpoint is not available on disk")

        response = _job_pop_response(
            sdxl_1_0_base_model_name,
            seed="987654321",
            width=1024,
            height=1024,
            source_processing=KNOWN_IMAGE_SOURCE_PROCESSING.img2img,
        )
        params = _converter_params(response, _offline_reference_manager())
        assert params.source_processing == KNOWN_IMAGE_SOURCE_PROCESSING.txt2img
        assert not _has_component(params, Image2ImageGenerationParameters)

        monolithic = hordelib_instance.generate(params, pipeline=AUTO_PIPELINE)
        assert len(monolithic) == 1

        staged = _run_cold_pipeline(hordelib_instance, params)

        _assert_identity("sdxl_production_shape_txt2img", monolithic, staged)

    @pytest.mark.slow
    @pytest.mark.default_sdxl_model
    def test_sdxl_img2img_stages_match_monolithic(
        self,
        hordelib_instance: HordeLib,
        shared_model_manager: type[SharedModelManager],
        sdxl_1_0_base_model_name: str,
    ) -> None:
        assert shared_model_manager.manager.compvis is not None
        if sdxl_1_0_base_model_name not in shared_model_manager.manager.compvis.available_models:
            pytest.skip(f"{sdxl_1_0_base_model_name} checkpoint is not available on disk")

        response = _job_pop_response(
            sdxl_1_0_base_model_name,
            seed="666",
            prompt="a dinosaur",
            width=1024,
            height=1024,
            denoising_strength=0.6,
            source_processing=KNOWN_IMAGE_SOURCE_PROCESSING.img2img,
            source_image=_gradient_png_base64(1024, 1024),
        )
        params = _converter_params(response, _offline_reference_manager())
        assert _has_component(params, Image2ImageGenerationParameters)

        monolithic = hordelib_instance.generate(params, pipeline=AUTO_PIPELINE)
        assert len(monolithic) == 1

        staged = _run_cold_pipeline(hordelib_instance, params, img2img=True)

        _assert_identity("sdxl_img2img", monolithic, staged)

    @pytest.mark.slow
    @pytest.mark.default_sdxl_model
    def test_sdxl_clip_skip_txt2img_stages_match_monolithic(
        self,
        hordelib_instance: HordeLib,
        shared_model_manager: type[SharedModelManager],
        sdxl_1_0_base_model_name: str,
    ) -> None:
        assert shared_model_manager.manager.compvis is not None
        if sdxl_1_0_base_model_name not in shared_model_manager.manager.compvis.available_models:
            pytest.skip(f"{sdxl_1_0_base_model_name} checkpoint is not available on disk")

        response = _job_pop_response(
            sdxl_1_0_base_model_name,
            seed="987654321",
            width=1024,
            height=1024,
            clip_skip=2,
        )
        params = _converter_params(response, _offline_reference_manager())

        monolithic = hordelib_instance.generate(params, pipeline=AUTO_PIPELINE)
        assert len(monolithic) == 1

        staged = _run_cold_pipeline(hordelib_instance, params)

        _assert_identity("sdxl_clip_skip", monolithic, staged)

    def test_converter_determinism_and_handbuilt_diff(
        self,
        stable_diffusion_model_name_for_testing: str,
    ) -> None:
        reference_manager = _offline_reference_manager()
        response = _job_pop_response(stable_diffusion_model_name_for_testing, seed="123456789", n_iter=1)

        serialized = [_converter_params(response, reference_manager).model_dump_json() for _ in range(3)]
        assert serialized[0] == serialized[1] == serialized[2], (
            "converter produced non-identical generation parameters across repeated calls on the same response"
        )

        converter_params = _converter_params(response, reference_manager)
        handbuilt = _handbuilt_params(stable_diffusion_model_name_for_testing, seed="123456789")

        diff = _json_diff(
            converter_params.model_dump(mode="json"),
            handbuilt.model_dump(mode="json"),
        )
        print("POP_MATRIX_CONVERTER_VS_HANDBUILT_DIFF (converter vs sibling hand-built params):")
        for line in diff:
            print(f"  {line}")

    @pytest.mark.slow
    @pytest.mark.default_sd15_model
    def test_lane_sequence_interleaved_decode_and_post_process(
        self,
        hordelib_instance: HordeLib,
        shared_model_manager: type[SharedModelManager],
        stable_diffusion_model_name_for_testing: str,
    ) -> None:
        reference_manager = _offline_reference_manager()
        assert shared_model_manager.manager is not None
        assert shared_model_manager.manager.download_model(_UPSCALE_MODEL)

        params_a = _converter_params(
            _job_pop_response(stable_diffusion_model_name_for_testing, seed="123456789"),
            reference_manager,
        )
        params_b = _converter_params(
            _job_pop_response(stable_diffusion_model_name_for_testing, seed="987654321"),
            reference_manager,
        )

        monolithic_a = hordelib_instance.generate(params_a, pipeline=AUTO_PIPELINE)
        assert len(monolithic_a) == 1
        monolithic_b = hordelib_instance.generate(params_b, pipeline=AUTO_PIPELINE)
        assert len(monolithic_b) == 1

        positive_a, negative_a = _cold_encode(hordelib_instance, params_a)
        latent_a = _cold_sample(hordelib_instance, params_a, positive_a, negative_a, None)
        positive_b, negative_b = _cold_encode(hordelib_instance, params_b)
        latent_b = _cold_sample(hordelib_instance, params_b, positive_b, negative_b, None)

        # The hostile interleave: decode job A, run a post-processing model through the same instance
        # (which loads and frees a different graph), then decode job B. Both decodes must still match.
        decode_a = _cold_decode(hordelib_instance, params_a, latent_a)

        post_process_source = Image.open("images/test_db0.jpg")
        post_process_result = hordelib_instance.post_process(
            UpscalePayload(model=_UPSCALE_MODEL, source_image=post_process_source),
        )
        assert post_process_result.image is not None

        decode_b = _cold_decode(hordelib_instance, params_b, latent_b)

        _assert_identity("lane_sequence_decode_a", monolithic_a, decode_a)
        _assert_identity("lane_sequence_decode_b", monolithic_b, decode_b)


class TestPopMatrixExploratory:
    """Exploratory evidence for re-including transparent jobs in disaggregation routing.

    Transparent jobs are currently excluded from disagg routing because the staged transparent path is
    not identity-validated. This case runs it and reports; it is deliberately kept out of the pass/fail
    matrix above until the evidence supports re-inclusion.
    """

    @pytest.mark.slow
    @pytest.mark.default_sd15_model
    def test_exploratory_transparent_txt2img_staged_vs_monolithic(
        self,
        hordelib_instance: HordeLib,
        stable_diffusion_model_name_for_testing: str,
    ) -> None:
        response = _job_pop_response(
            stable_diffusion_model_name_for_testing,
            seed="123456789",
            transparent=True,
        )
        params = _converter_params(response, _offline_reference_manager())

        monolithic = hordelib_instance.generate(params, pipeline=AUTO_PIPELINE)
        assert len(monolithic) == 1
        assert isinstance(monolithic[0].image, Image.Image)

        staged = _run_cold_pipeline(hordelib_instance, params)
        assert len(staged) == 1, f"transparent stage path produced {len(staged)} image(s), expected 1"
        assert isinstance(staged[0].image, Image.Image)

        # The monolithic transparent render (RGBA via layerdiffuse) is the reference; both renders are
        # flattened over white to RGB so the three-channel similarity metric can compare them.
        mono_rgb = _flatten_to_rgb(monolithic[0].image)
        stage_rgb = _flatten_to_rgb(staged[0].image)

        cosine, histogram = evaluate_image_distance(mono_rgb, stage_rgb)
        print(f"POP_MATRIX_TRANSPARENT_EXPLORATORY sd15_transparent: {cosine} | {histogram}")
        if cosine.cosine_similarity < CosineSimilarityResultCode.PERCEPTUALLY_IDENTICAL:
            _DUMP_ROOT.mkdir(parents=True, exist_ok=True)
            mono_path = _DUMP_ROOT / "sd15_transparent_monolithic.png"
            stage_path = _DUMP_ROOT / "sd15_transparent_disagg.png"
            monolithic[0].image.save(mono_path)
            staged[0].image.save(stage_path)
            raise AssertionError(
                f"sd15_transparent exploratory: staged transparent path diverged from the monolithic "
                f"render: {cosine}; {histogram}; monolithic written to {mono_path}, disaggregated written "
                f"to {stage_path}",
            )
