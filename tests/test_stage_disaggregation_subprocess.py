"""GPU subprocess-per-stage reproduction for the disaggregated stage path.

In production each disaggregated stage (encode, sample, decode) runs in its OWN process, so each cold
initializes ComfyUI global state, cold-loads its checkpoint subset from disk, and adopts mmap-backed
weights across process boundaries. The in-process stage-identity and cold-cache modules keep all stages
inside a single interpreter that has already run ``generate``, so neither exercises fresh per-process
ComfyUI state nor cross-process weight adoption. This module closes that gap: the monolithic reference
is rendered in the test process, then encode -> sample -> decode each run via ``subprocess.run`` of
:mod:`tests.stage_subprocess_runner` under the repo venv python, with blobs handed between processes as
files. The decoded image must reproduce the monolithic render at perceptual identity.

The bar is perceptual identity (never relaxed). On divergence both images are written under the dump
root and the cosine plus histogram metrics ride on the assertion message. Each subprocess's stdout and
stderr are captured to files under the dump root for forensics.

These are real-GPU tests, marked ``slow`` plus the checkpoint's model marker (matching
``tests/test_stage_disaggregation.py``), deselected by the CI default ``-m "not slow"``. Run manually
and serially, for example::

    uv run --no-sync pytest tests/test_stage_disaggregation_subprocess.py -m slow
"""

from __future__ import annotations

import json
import os
import subprocess
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

_REPO_ROOT = Path(__file__).resolve().parent.parent
_RUNNER = _REPO_ROOT / "tests" / "stage_subprocess_runner.py"
_VENV_PYTHON = _REPO_ROOT / ".venv" / "Scripts" / "python.exe"

_DUMP_ROOT = Path(
    r"G:\_temp_\claude\G--mxd-ai-python-local-dev-git-repos-tazlin-horde-worker-reGen"
    r"\fa65ea7e-1f5d-4a0b-b681-7c4f696bf546\scratchpad\subproc_repro",
)

_STAGE_TIMEOUT_SECONDS = 900


def _params(
    model: str,
    *,
    seed: str,
    width: int,
    height: int,
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


def _run_stage_subprocess(
    case_name: str,
    stage: str,
    spec: dict,
    work_dir: Path,
    *,
    disable_zero_copy: bool,
) -> None:
    """Run one stage in a fresh venv-python process and fail loudly on nonzero exit or timeout.

    The job spec is written to a JSON file the runner reads. Each process's stdout and stderr are
    captured to files under the dump root so a failing stage can be inspected after the fact.
    """
    spec_path = work_dir / f"{stage}_spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")

    env = dict(os.environ)
    env["TESTS_ONGOING"] = "1"
    if disable_zero_copy:
        env["HORDELIB_DISABLE_ZERO_COPY"] = "1"

    _DUMP_ROOT.mkdir(parents=True, exist_ok=True)
    stdout_path = _DUMP_ROOT / f"{case_name}_{stage}_stdout.log"
    stderr_path = _DUMP_ROOT / f"{case_name}_{stage}_stderr.log"

    try:
        completed = subprocess.run(
            [str(_VENV_PYTHON), str(_RUNNER), str(spec_path)],
            cwd=str(_REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=_STAGE_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired as timeout_error:
        stdout_path.write_text(timeout_error.stdout or "", encoding="utf-8")
        stderr_path.write_text(timeout_error.stderr or "", encoding="utf-8")
        raise AssertionError(
            f"{case_name}: {stage} stage subprocess timed out after {_STAGE_TIMEOUT_SECONDS}s; "
            f"stdout at {stdout_path}, stderr at {stderr_path}",
        ) from timeout_error

    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")

    if completed.returncode != 0:
        raise AssertionError(
            f"{case_name}: {stage} stage subprocess exited {completed.returncode}; "
            f"stdout at {stdout_path}, stderr at {stderr_path}\n"
            f"--- tail of stderr ---\n{completed.stderr[-4000:]}",
        )


def _run_subprocess_pipeline(
    case_name: str,
    params: ImageGenerationParameters,
    work_dir: Path,
    *,
    disable_zero_copy: bool,
) -> Image.Image:
    """Drive encode -> sample -> decode, each in its own process, and return the decoded image."""
    params_json_path = work_dir / "params.json"
    params_json_path.write_text(params.model_dump_json(), encoding="utf-8")

    positive_path = work_dir / "positive.cond"
    negative_path = work_dir / "negative.cond"
    latent_path = work_dir / "latent.bin"
    image_path = work_dir / "decoded.png"

    _run_stage_subprocess(
        case_name,
        "encode",
        {
            "stage": "encode",
            "params_json_path": str(params_json_path),
            "outputs": {"positive": str(positive_path), "negative": str(negative_path)},
        },
        work_dir,
        disable_zero_copy=disable_zero_copy,
    )
    _run_stage_subprocess(
        case_name,
        "sample",
        {
            "stage": "sample",
            "params_json_path": str(params_json_path),
            "inputs": {"positive": str(positive_path), "negative": str(negative_path)},
            "outputs": {"latent": str(latent_path)},
        },
        work_dir,
        disable_zero_copy=disable_zero_copy,
    )
    _run_stage_subprocess(
        case_name,
        "decode",
        {
            "stage": "decode",
            "params_json_path": str(params_json_path),
            "inputs": {"latent": str(latent_path)},
            "outputs": {"image": str(image_path)},
        },
        work_dir,
        disable_zero_copy=disable_zero_copy,
    )

    return Image.open(image_path).copy()


def _assert_subprocess_identity(
    case_name: str,
    monolithic: list[ResultingImageReturn],
    decoded: Image.Image,
) -> None:
    """Assert the subprocess stage path matched the monolithic render at perceptual identity.

    On divergence both images are written under the dump root with case-descriptive names and the
    failure carries the cosine and histogram metrics; the identity bar is never relaxed.
    """
    assert len(monolithic) == 1
    mono = monolithic[0]
    assert isinstance(mono.image, Image.Image)

    cosine, histogram = evaluate_image_distance(mono.image, decoded)
    print(f"SUBPROC_IDENTITY {case_name}: {cosine} | {histogram}")
    if cosine.cosine_similarity < CosineSimilarityResultCode.PERCEPTUALLY_IDENTICAL:
        _DUMP_ROOT.mkdir(parents=True, exist_ok=True)
        mono_path = _DUMP_ROOT / f"{case_name}_monolithic.png"
        stage_path = _DUMP_ROOT / f"{case_name}_subprocess.png"
        mono.image.save(mono_path)
        decoded.save(stage_path)
        raise AssertionError(
            f"{case_name}: subprocess-per-stage path diverged from the monolithic render: "
            f"{cosine}; {histogram}; monolithic written to {mono_path}, subprocess written to {stage_path}",
        )


class TestStageDisaggregationSubprocess:
    @pytest.mark.slow
    @pytest.mark.default_sd15_model
    def test_sd15_txt2img_subprocess_stages_match_monolithic(
        self,
        hordelib_instance: HordeLib,
        stable_diffusion_model_name_for_testing: str,
        tmp_path: Path,
    ) -> None:
        assert _VENV_PYTHON.exists(), f"repo venv python not found at {_VENV_PYTHON}"

        params = _params(stable_diffusion_model_name_for_testing, seed="123456789", width=512, height=512)

        monolithic = hordelib_instance.generate(params, pipeline=AUTO_PIPELINE)
        assert len(monolithic) == 1

        decoded = _run_subprocess_pipeline("sd15_txt2img", params, tmp_path, disable_zero_copy=False)

        _assert_subprocess_identity("sd15_txt2img", monolithic, decoded)

    @pytest.mark.slow
    @pytest.mark.default_sdxl_model
    def test_sdxl_txt2img_subprocess_stages_match_monolithic(
        self,
        hordelib_instance: HordeLib,
        shared_model_manager: type[SharedModelManager],
        sdxl_1_0_base_model_name: str,
        tmp_path: Path,
    ) -> None:
        assert _VENV_PYTHON.exists(), f"repo venv python not found at {_VENV_PYTHON}"
        assert shared_model_manager.manager.compvis is not None
        if sdxl_1_0_base_model_name not in shared_model_manager.manager.compvis.available_models:
            pytest.skip(f"{sdxl_1_0_base_model_name} checkpoint is not available on disk")

        params = _params(sdxl_1_0_base_model_name, seed="987654321", width=1024, height=1024)

        monolithic = hordelib_instance.generate(params, pipeline=AUTO_PIPELINE)
        assert len(monolithic) == 1

        decoded = _run_subprocess_pipeline("sdxl_txt2img", params, tmp_path, disable_zero_copy=False)

        _assert_subprocess_identity("sdxl_txt2img", monolithic, decoded)
