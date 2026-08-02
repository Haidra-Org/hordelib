"""Unit tests for the sampler-truncation record crossing the disaggregated sample stage (CPU only).

The sample stage runs in its own process and its LATENT is decoded elsewhere, so the record has to
ride the stage's return value or it is lost at the lane split. No ComfyUI, no weights: the backend
is stubbed and the recording bracket is driven directly.
"""

import io
from typing import Any

from hordelib.execution.adaptive_sampler_bound import (
    SAMPLER_TRUNCATION_METADATA_KEY,
    SamplerTruncation,
    _record_truncation,
    begin_run_recording,
)
from hordelib.execution.in_process import InProcessComfyBackend
from hordelib.execution.interface import OutputArtifact, OutputKind, OutputSpec
from hordelib.horde import HordeLib, SampleStageResult

_TRUNCATION = SamplerTruncation(sampler="dpm_adaptive", nominal_steps=20, iterations=25)


class _StubBackend:
    """Returns one LATENT artifact carrying whatever metadata the test wants to surface."""

    def __init__(self, metadata: dict[str, Any]) -> None:
        self.metadata = metadata

    def run_pipeline(self, _graph: dict, **_kwargs: Any) -> list[OutputArtifact]:
        return [
            OutputArtifact(
                data=io.BytesIO(b"LATENT"),
                mime_type="application/octet-stream",
                kind=OutputKind.LATENT,
                source_node="latent_output",
                metadata=dict(self.metadata),
            ),
        ]


class _StubGraph:
    def to_api_dict(self) -> dict:
        return {}


def _sample_stage_with(metadata: dict[str, Any], monkeypatch) -> SampleStageResult:
    """Run ``sample_stage`` against a stub backend whose artifact carries *metadata*."""
    import hordelib.horde as horde_module

    horde = object.__new__(HordeLib)
    horde.backend = _StubBackend(metadata)
    monkeypatch.setattr(
        HordeLib,
        "_materialize_stage_graph",
        lambda _self, _params: (_StubGraph(), (), []),
    )
    monkeypatch.setattr(
        horde_module,
        "cut_sample_stage",
        lambda *_args, **_kwargs: (OutputSpec(node="latent_output"),),
    )

    return horde.sample_stage(
        None,  # the stubbed materialization never reads the parameters
        positive_conditioning_bytes=b"p",
        negative_conditioning_bytes=b"n",
    )


def test_sample_stage_surfaces_the_truncation_on_its_return(monkeypatch) -> None:
    result = _sample_stage_with({SAMPLER_TRUNCATION_METADATA_KEY: _TRUNCATION}, monkeypatch)

    assert result.latent_bytes == b"LATENT"
    assert result.sampler_truncation == _TRUNCATION


def test_sample_stage_returns_no_truncation_when_the_sampler_ran_to_completion(monkeypatch) -> None:
    result = _sample_stage_with({}, monkeypatch)

    assert result.latent_bytes == b"LATENT"
    assert result.sampler_truncation is None


class _StubComfy:
    """Stands in for ``Comfy_Horde``, recording a truncation on the runs the test asks it to."""

    def __init__(self, truncating_runs: set[int]) -> None:
        self.truncating_runs = truncating_runs
        self.runs = 0

    def run_pipeline(self, *_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        self.runs += 1
        if self.runs in self.truncating_runs:
            _record_truncation(_TRUNCATION)
        return [{"imagedata": io.BytesIO(b"LATENT"), "type": "LATENT", "source_node": "latent_output"}]


def test_a_truncation_does_not_leak_into_the_next_stage_run() -> None:
    """Only the run that truncated carries the record; the bracket clears it for the next run."""
    backend = InProcessComfyBackend()
    backend._comfy = _StubComfy(truncating_runs={1})
    outputs = (OutputSpec(node="latent_output", kind=OutputKind.LATENT),)

    first = backend.run_pipeline({}, outputs=outputs)
    second = backend.run_pipeline({}, outputs=outputs)

    assert first[0].metadata[SAMPLER_TRUNCATION_METADATA_KEY] == _TRUNCATION
    assert SAMPLER_TRUNCATION_METADATA_KEY not in second[0].metadata


def test_a_truncation_recorded_before_a_run_starts_is_not_attributed_to_it() -> None:
    """A record left behind by an unbracketed caller is discarded when the next run begins."""
    begin_run_recording()
    _record_truncation(_TRUNCATION)
    backend = InProcessComfyBackend()
    backend._comfy = _StubComfy(truncating_runs=set())

    artifacts = backend.run_pipeline({}, outputs=(OutputSpec(node="latent_output", kind=OutputKind.LATENT),))

    assert SAMPLER_TRUNCATION_METADATA_KEY not in artifacts[0].metadata
