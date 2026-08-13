"""Pins the retention grant and the device-truth reading crossing the disaggregated sample stage (CPU only).

A sampler runs the same end-of-run eviction every other job does, so without a grant reaching it the
stage returns the card after every sample and the next same-model sample re-uploads the UNet. The grant
therefore has to be carried the whole way from the stage entry point to the layer that makes the eviction
decision, and the device-level free reading with it: the sample stage is where the UNet loads and where
the job's whole sampling activation lands, so it is the stage whose shortfall arithmetic most needs to be
computed against measured device truth rather than the process-local view.

No ComfyUI and no weights here: the backend (and, for the whole-chain test, the bridge) are stubbed and
the forwarding is observed directly.
"""

import io
from typing import Any

import pytest

from hordelib.execution.in_process import InProcessComfyBackend
from hordelib.execution.interface import OutputArtifact, OutputKind, OutputSpec
from hordelib.horde import HordeLib


class _RecordingBackend:
    """Records the keyword arguments each ``run_pipeline`` call carried."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def run_pipeline(self, _graph: dict, **kwargs: Any) -> list[OutputArtifact]:
        self.calls.append(kwargs)
        return [
            OutputArtifact(
                data=io.BytesIO(b"LATENT"),
                mime_type="application/octet-stream",
                kind=OutputKind.LATENT,
                source_node="latent_output",
                metadata={},
            ),
        ]


class _StubGraph:
    def to_api_dict(self) -> dict:
        return {}


class _StubTypedPayload:
    def solver_options(self) -> None:
        return None


def _sampler(monkeypatch: pytest.MonkeyPatch, backend: Any) -> HordeLib:
    """A HordeLib whose stage materialization is stubbed, so only the forwarding is under test."""
    import hordelib.horde as horde_module

    horde = object.__new__(HordeLib)
    horde.backend = backend
    monkeypatch.setattr(
        HordeLib,
        "_materialize_stage_graph",
        lambda _self, _params: (_StubGraph(), (), [], _StubTypedPayload(), None),
    )
    monkeypatch.setattr(
        horde_module,
        "cut_sample_stage",
        lambda *_args, **_kwargs: (OutputSpec(node="latent_output"),),
    )
    return horde


def test_an_ungranted_sample_stage_asks_for_no_deferral(monkeypatch: pytest.MonkeyPatch) -> None:
    """The default is the eviction: a stage nobody granted returns the card at the end of its run."""
    backend = _RecordingBackend()
    horde = _sampler(monkeypatch, backend)

    horde.sample_stage(None, positive_conditioning_bytes=b"p", negative_conditioning_bytes=b"n")

    assert backend.calls[0]["defer_vram_unload"] is False
    assert backend.calls[0]["device_free_truth_mb"] is None


def test_a_granted_sample_stage_defers_the_unload(monkeypatch: pytest.MonkeyPatch) -> None:
    """The grant reaches the backend, which is the only thing that suppresses the end-of-run eviction."""
    backend = _RecordingBackend()
    horde = _sampler(monkeypatch, backend)

    horde.sample_stage(
        None,
        positive_conditioning_bytes=b"p",
        negative_conditioning_bytes=b"n",
        defer_vram_unload=True,
    )

    assert backend.calls[0]["defer_vram_unload"] is True


def test_the_device_reading_reaches_the_sample_stages_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """A sampler told what the card really holds clamps its own view with it, as every other run does."""
    backend = _RecordingBackend()
    horde = _sampler(monkeypatch, backend)

    horde.sample_stage(
        None,
        positive_conditioning_bytes=b"p",
        negative_conditioning_bytes=b"n",
        device_free_truth_mb=1234.0,
    )

    assert backend.calls[0]["device_free_truth_mb"] == 1234.0


class _RecordingComfy:
    """Stands in for ``Comfy_Horde``, recording what the eviction decision would have been made on."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def run_pipeline(self, *_args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        self.calls.append(kwargs)
        return [{"imagedata": io.BytesIO(b"LATENT"), "type": "LATENT", "source_node": "latent_output"}]


@pytest.mark.parametrize("granted", [False, True], ids=["ungranted", "granted"])
def test_the_grant_reaches_the_layer_that_decides_the_eviction(
    monkeypatch: pytest.MonkeyPatch,
    granted: bool,
) -> None:
    """End to end over the real backend: the bridge is handed the same verdict the caller passed.

    The bridge's own ``defer_vram_unload`` handling is pinned by the end-of-job eviction tests; what this
    adds is that a sample stage reaches it at all, which is the whole of the gap a disaggregated sampler
    fell through.
    """
    comfy = _RecordingComfy()
    backend = InProcessComfyBackend()
    backend._comfy = comfy
    horde = _sampler(monkeypatch, backend)

    horde.sample_stage(
        None,
        positive_conditioning_bytes=b"p",
        negative_conditioning_bytes=b"n",
        defer_vram_unload=granted,
        device_free_truth_mb=2048.0,
    )

    assert comfy.calls[0]["defer_vram_unload"] is granted
    assert comfy.calls[0]["device_free_truth_mb"] == 2048.0
