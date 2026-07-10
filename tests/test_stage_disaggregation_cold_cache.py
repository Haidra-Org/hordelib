"""GPU cold-cache reproduction for the disaggregated stage path.

The stage entry points cut a single materialized family graph down to one component and toggle the
combined ``HordeCheckpointLoader`` subset flags (output_model/output_clip/output_vae). Correctness of
that subset cut is only exercised when the loader actually cold-loads the subset from disk. When a prior
full ``generate`` has populated the loader's in-RAM cache with the complete (model, clip, vae) tuple,
every subsequent stage request is a superset cache hit that reuses the full components, so the genuine
cold subset-load path never runs. In production each stage runs in its own process and does cold-load
its subset, so this module forces that path in-process by clearing the loader cache before a stage.

The seam is ``SharedModelManager.manager._models_in_ram`` (see :mod:`hordelib.model_manager.hyper`),
keyed by the bare horde model name for the combined-loader stage path. The loader consults only this
dict when deciding to reuse (see :mod:`hordelib.nodes.node_model_loader`); emptying it forces the
disk-load branch with the stage's subset flags in effect.

Two case sets, both at a fixed seed compared against the monolithic ``generate`` render:

- Full-cold: the cache is cleared before EVERY stage, so each cold-loads only its subset. This is the
  in-process analog of the per-process production path.
- Per-stage isolation: the cache is cleared before exactly ONE stage and restored to the full tuple for
  the others, so a divergence attributes to that single stage's cold subset load.

The bar is perceptual identity (never relaxed). On divergence both images are written to a stable
directory and the cosine plus histogram metrics ride on the assertion message.

These are real-GPU tests, marked ``slow`` plus the checkpoint's model marker (matching
``tests/test_stage_disaggregation.py``), deselected by the CI default ``-m "not slow"``. Run manually and
serially, for example::

    uv run --no-sync pytest tests/test_stage_disaggregation_cold_cache.py -m slow
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest
from horde_sdk.generation_parameters.image import (
    BasicImageGenerationParameters,
    ImageGenerationParameters,
)
from horde_sdk.generation_parameters.image.object_models import ImageGenerationComponentContainer
from PIL import Image

from hordelib.horde import HordeLib, ResultingImageReturn
from hordelib.pipeline.identifiers import AUTO_PIPELINE
from hordelib.shared_model_manager import SharedModelManager
from hordelib.utils.distance import CosineSimilarityResultCode, evaluate_image_distance

STAGE_ENCODE = "encode"
STAGE_SAMPLE = "sample"
STAGE_DECODE = "decode"

_DUMP_ROOT = Path(
    r"G:\_temp_\claude\G--mxd-ai-python-local-dev-git-repos-tazlin-horde-worker-reGen"
    r"\fa65ea7e-1f5d-4a0b-b681-7c4f696bf546\scratchpad\cold_cache_repro",
)


def _params(
    model: str,
    *,
    seed: str,
    width: int = 512,
    height: int = 512,
    prompt: str = "an ancient llamia monster",
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
        clip_skip=1,
        denoising_strength=1.0,
    )
    return ImageGenerationParameters(
        result_ids=[uuid.uuid4()],
        batch_size=1,
        source_processing="txt2img",
        base_params=base,
        additional_params=ImageGenerationComponentContainer(components=[]),
    )


def _clear_loader_cache() -> None:
    """Empty the loader's in-RAM component cache so the next stage cold-loads its subset from disk."""
    SharedModelManager.manager._models_in_ram = {}


def _snapshot_loader_cache() -> dict:
    """Copy the current loader cache so a warm stage can be handed back the full-tuple entry it needs."""
    return dict(SharedModelManager.manager._models_in_ram)


def _restore_loader_cache(snapshot: dict) -> None:
    """Reinstate a snapshot as the loader cache so a warm stage takes a superset hit on the full tuple.

    The snapshot holds live references, so garbage collection cannot free the cached component objects
    between the snapshot and the restore; the warm stage reuses exactly the tuple the monolithic render
    loaded.
    """
    SharedModelManager.manager._models_in_ram = dict(snapshot)


def _run_cold_pipeline(
    hordelib_instance: HordeLib,
    params: ImageGenerationParameters,
    *,
    cold_stages: set[str],
    warm_snapshot: dict,
) -> list[ResultingImageReturn]:
    """Drive the txt2img disaggregated stage path, cold-loading the designated stages.

    Before each stage the loader cache is emptied when that stage is in ``cold_stages`` (forcing a cold
    subset load) or restored to ``warm_snapshot`` otherwise (a superset hit on the full tuple).
    """
    if STAGE_ENCODE in cold_stages:
        _clear_loader_cache()
    else:
        _restore_loader_cache(warm_snapshot)
    positive_bytes, negative_bytes = hordelib_instance.encode_text_stage(params)

    if STAGE_SAMPLE in cold_stages:
        _clear_loader_cache()
    else:
        _restore_loader_cache(warm_snapshot)
    latent_bytes = hordelib_instance.sample_stage(
        params,
        positive_conditioning_bytes=positive_bytes,
        negative_conditioning_bytes=negative_bytes,
        source_latent_bytes=None,
    )

    if STAGE_DECODE in cold_stages:
        _clear_loader_cache()
    else:
        _restore_loader_cache(warm_snapshot)
    staged_results, _faults = hordelib_instance.decode_stage(params, latent_bytes=latent_bytes)
    return staged_results


def _assert_cold_identity(
    case_name: str,
    monolithic: list[ResultingImageReturn],
    staged: list[ResultingImageReturn],
) -> None:
    """Assert the cold-cache stage path matched the monolithic render in count and perceptual identity.

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
        print(f"COLD_CACHE_IDENTITY {case_name} image {index}: {cosine} | {histogram}")
        if cosine.cosine_similarity < CosineSimilarityResultCode.PERCEPTUALLY_IDENTICAL:
            _DUMP_ROOT.mkdir(parents=True, exist_ok=True)
            mono_path = _DUMP_ROOT / f"{case_name}_{index}_monolithic.png"
            stage_path = _DUMP_ROOT / f"{case_name}_{index}_disagg.png"
            mono.image.save(mono_path)
            stage.image.save(stage_path)
            raise AssertionError(
                f"{case_name} (image {index}): cold-cache disaggregated stages diverged from the "
                f"monolithic render: {cosine}; {histogram}; monolithic written to {mono_path}, "
                f"disaggregated written to {stage_path}",
            )


class TestStageDisaggregationColdCache:
    @pytest.mark.slow
    @pytest.mark.default_sd15_model
    def test_sd15_txt2img_full_cold(
        self,
        hordelib_instance: HordeLib,
        stable_diffusion_model_name_for_testing: str,
    ) -> None:
        params = _params(stable_diffusion_model_name_for_testing, seed="123456789")

        monolithic = hordelib_instance.generate(params, pipeline=AUTO_PIPELINE)
        assert len(monolithic) == 1
        warm_snapshot = _snapshot_loader_cache()

        staged = _run_cold_pipeline(
            hordelib_instance,
            params,
            cold_stages={STAGE_ENCODE, STAGE_SAMPLE, STAGE_DECODE},
            warm_snapshot=warm_snapshot,
        )

        _assert_cold_identity("sd15_txt2img_full_cold", monolithic, staged)

    @pytest.mark.slow
    @pytest.mark.default_sdxl_model
    def test_sdxl_txt2img_full_cold(
        self,
        hordelib_instance: HordeLib,
        shared_model_manager: type[SharedModelManager],
        sdxl_1_0_base_model_name: str,
    ) -> None:
        assert shared_model_manager.manager.compvis is not None
        if sdxl_1_0_base_model_name not in shared_model_manager.manager.compvis.available_models:
            pytest.skip(f"{sdxl_1_0_base_model_name} checkpoint is not available on disk")

        params = _params(sdxl_1_0_base_model_name, seed="987654321", width=1024, height=1024)

        monolithic = hordelib_instance.generate(params, pipeline=AUTO_PIPELINE)
        assert len(monolithic) == 1
        warm_snapshot = _snapshot_loader_cache()

        staged = _run_cold_pipeline(
            hordelib_instance,
            params,
            cold_stages={STAGE_ENCODE, STAGE_SAMPLE, STAGE_DECODE},
            warm_snapshot=warm_snapshot,
        )

        _assert_cold_identity("sdxl_txt2img_full_cold", monolithic, staged)

    @pytest.mark.slow
    @pytest.mark.default_sd15_model
    @pytest.mark.parametrize("cold_stage", [STAGE_ENCODE, STAGE_SAMPLE, STAGE_DECODE])
    def test_sd15_txt2img_single_stage_cold(
        self,
        hordelib_instance: HordeLib,
        stable_diffusion_model_name_for_testing: str,
        cold_stage: str,
    ) -> None:
        params = _params(stable_diffusion_model_name_for_testing, seed="123456789")

        monolithic = hordelib_instance.generate(params, pipeline=AUTO_PIPELINE)
        assert len(monolithic) == 1
        warm_snapshot = _snapshot_loader_cache()

        staged = _run_cold_pipeline(
            hordelib_instance,
            params,
            cold_stages={cold_stage},
            warm_snapshot=warm_snapshot,
        )

        _assert_cold_identity(f"sd15_txt2img_cold_{cold_stage}", monolithic, staged)

    @pytest.mark.slow
    @pytest.mark.default_sdxl_model
    @pytest.mark.parametrize("cold_stage", [STAGE_ENCODE, STAGE_SAMPLE, STAGE_DECODE])
    def test_sdxl_txt2img_single_stage_cold(
        self,
        hordelib_instance: HordeLib,
        shared_model_manager: type[SharedModelManager],
        sdxl_1_0_base_model_name: str,
        cold_stage: str,
    ) -> None:
        assert shared_model_manager.manager.compvis is not None
        if sdxl_1_0_base_model_name not in shared_model_manager.manager.compvis.available_models:
            pytest.skip(f"{sdxl_1_0_base_model_name} checkpoint is not available on disk")

        params = _params(sdxl_1_0_base_model_name, seed="987654321", width=1024, height=1024)

        monolithic = hordelib_instance.generate(params, pipeline=AUTO_PIPELINE)
        assert len(monolithic) == 1
        warm_snapshot = _snapshot_loader_cache()

        staged = _run_cold_pipeline(
            hordelib_instance,
            params,
            cold_stages={cold_stage},
            warm_snapshot=warm_snapshot,
        )

        _assert_cold_identity(f"sdxl_txt2img_cold_{cold_stage}", monolithic, staged)
