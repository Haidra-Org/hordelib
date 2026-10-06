"""``HordeLib.post_process_chain`` and the unencoded artifact path, with no GPU and no model files."""

import io
import sys
import types
from collections.abc import Sequence
from typing import Any
from unittest.mock import MagicMock

import pytest
from PIL import Image

import hordelib.horde as horde_module
from hordelib.execution.in_process import InProcessComfyBackend
from hordelib.execution.interface import OutputArtifact, OutputSpec, UnencodedImageArtifact
from hordelib.execution.results import UI_ENTRY_IMAGE_KEY, UNENCODED_IMAGE_TYPE
from hordelib.horde import HordeLib
from hordelib.pipeline.payload_pp import FacefixPayload, UpscalePayload, order_post_processing

_UPSCALER = "RealESRGAN_x4plus"
_FACEFIXER = "GFPGAN"
_STRIP = "strip_background"
_OUTPUT_NODE = "output_image"


class _FakeComfy:
    """Stands in for the in-process ComfyUI runner, returning canned ui entries."""

    last_run_retained_weights_evicted = False

    def __init__(self, results: list[dict[str, Any]]) -> None:
        self._results = results

    def run_pipeline(self, *_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        return self._results


def _backend_returning(results: list[dict[str, Any]]) -> InProcessComfyBackend:
    backend = object.__new__(InProcessComfyBackend)
    backend._comfy = _FakeComfy(results)  # type: ignore[assignment]
    return backend


def _unencoded_entry() -> dict[str, Any]:
    return {
        UI_ENTRY_IMAGE_KEY: Image.new("RGB", (4, 4)),
        "type": UNENCODED_IMAGE_TYPE,
        "source_node": _OUTPUT_NODE,
    }


def _encoded_entry() -> dict[str, Any]:
    return {"imagedata": io.BytesIO(b"png"), "type": "PNG", "source_node": _OUTPUT_NODE}


def test_to_artifacts_builds_unencoded_artifact() -> None:
    artifacts = InProcessComfyBackend._to_artifacts([_unencoded_entry()], (OutputSpec(node=_OUTPUT_NODE),))
    assert len(artifacts) == 1
    assert isinstance(artifacts[0], UnencodedImageArtifact)
    assert artifacts[0].source_node == _OUTPUT_NODE
    assert artifacts[0].image.size == (4, 4)


def test_run_pipeline_raises_on_unencoded_entry() -> None:
    backend = _backend_returning([_unencoded_entry()])
    with pytest.raises(RuntimeError, match=_OUTPUT_NODE):
        backend.run_pipeline({}, outputs=(OutputSpec(node=_OUTPUT_NODE),))


def test_run_pipeline_unencoded_raises_on_encoded_entry() -> None:
    backend = _backend_returning([_encoded_entry()])
    with pytest.raises(RuntimeError, match=_OUTPUT_NODE):
        backend.run_pipeline_unencoded({}, outputs=(OutputSpec(node=_OUTPUT_NODE),))


def test_run_pipeline_unencoded_returns_images() -> None:
    backend = _backend_returning([_unencoded_entry()])
    artifacts = backend.run_pipeline_unencoded({}, outputs=(OutputSpec(node=_OUTPUT_NODE),))
    assert [type(a) for a in artifacts] == [UnencodedImageArtifact]


class _ChainHarness:
    """A ``HordeLib`` stand-in with a recording backend and the model-touching calls replaced."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.result_image = Image.new("RGB", (8, 8), "red")
        self.stripped_inputs: list[Image.Image] = []
        self.graph = MagicMock()
        self.graph.to_api_dict.return_value = {"graph": True}
        self.outputs = (OutputSpec(node=_OUTPUT_NODE),)
        self.compose = MagicMock(return_value=types.SimpleNamespace(graph=self.graph, outputs=self.outputs))
        self.backend = MagicMock()
        self.backend.run_pipeline_unencoded.return_value = [UnencodedImageArtifact(image=self.result_image)]
        self.events: list[str] = []
        self.backend.run_pipeline_unencoded.side_effect = self._record_run

        def _strip(image: Image.Image) -> Image.Image:
            self.events.append("strip")
            self.stripped_inputs.append(image)
            return image

        comfy_stub = types.ModuleType("hordelib.comfy_horde")
        comfy_stub.log_free_ram = lambda: None  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "hordelib.comfy_horde", comfy_stub)
        monkeypatch.setattr(horde_module, "compose_post_processing_chain", self.compose)
        monkeypatch.setattr(horde_module, "resolve_post_processing_model", lambda model: f"context:{model}")
        monkeypatch.setattr(horde_module, "ensure_feature_available", lambda _feature: None)
        monkeypatch.setattr(horde_module.ImageUtils, "strip_background", staticmethod(_strip))

    def _record_run(self, *_args: Any, **_kwargs: Any) -> list[UnencodedImageArtifact]:
        self.events.append("run")
        return [UnencodedImageArtifact(image=self.result_image)]

    def run(self, operations: Sequence[str], **kwargs: Any) -> horde_module.ResultingImageReturn:
        fake_self = types.SimpleNamespace(backend=self.backend)
        return HordeLib.post_process_chain(fake_self, Image.new("RGB", (8, 8)), operations, **kwargs)  # type: ignore[arg-type]


def test_chain_composes_once_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    harness = _ChainHarness(monkeypatch)
    operations = [_UPSCALER, _FACEFIXER]
    result = harness.run(operations)

    harness.compose.assert_called_once()
    stages = harness.compose.call_args.args[0]
    assert [payload.model for payload, _ in stages] == list(order_post_processing(operations))
    assert [context for _, context in stages] == [f"context:{payload.model}" for payload, _ in stages]
    harness.backend.run_pipeline_unencoded.assert_called_once_with({"graph": True}, outputs=harness.outputs)
    harness.backend.run_pipeline.assert_not_called()
    harness.graph.set_input.assert_called_once_with(f"{_OUTPUT_NODE}.encode_png", False)
    assert result.image is harness.result_image
    assert result.rawpng is None
    assert result.faults == []


def test_chain_applies_facefixer_strength(monkeypatch: pytest.MonkeyPatch) -> None:
    harness = _ChainHarness(monkeypatch)
    harness.run([_FACEFIXER, _UPSCALER], facefixer_strength=0.4)
    stages = harness.compose.call_args.args[0]
    facefix = [payload for payload, _ in stages if isinstance(payload, FacefixPayload)]
    assert [payload.strength for payload in facefix] == [0.4]
    assert any(isinstance(payload, UpscalePayload) for payload, _ in stages)


def test_chain_strips_background_after_graph(monkeypatch: pytest.MonkeyPatch) -> None:
    harness = _ChainHarness(monkeypatch)
    result = harness.run([_STRIP, _UPSCALER])
    assert harness.events == ["run", "strip"]
    assert harness.stripped_inputs == [harness.result_image]
    assert result.rawpng is None


def test_chain_skips_unknown_names(monkeypatch: pytest.MonkeyPatch) -> None:
    harness = _ChainHarness(monkeypatch)
    harness.run(["not_a_post_processor", _UPSCALER])
    stages = harness.compose.call_args.args[0]
    assert [payload.model for payload, _ in stages] == [_UPSCALER]


def test_strip_only_chain_never_runs_graph(monkeypatch: pytest.MonkeyPatch) -> None:
    harness = _ChainHarness(monkeypatch)
    harness.run([_STRIP])
    harness.compose.assert_not_called()
    harness.backend.run_pipeline_unencoded.assert_not_called()
    assert harness.events == ["strip"]


def test_unrecognised_only_chain_returns_source(monkeypatch: pytest.MonkeyPatch) -> None:
    harness = _ChainHarness(monkeypatch)
    source = Image.new("RGB", (8, 8))
    fake_self = types.SimpleNamespace(backend=harness.backend)
    result = HordeLib.post_process_chain(fake_self, source, ["nope"])  # type: ignore[arg-type]
    assert result.image is source
    harness.compose.assert_not_called()


def test_output_artifact_is_unchanged_for_encoded_entries() -> None:
    artifacts = InProcessComfyBackend._to_artifacts([_encoded_entry()], (OutputSpec(node=_OUTPUT_NODE),))
    assert [type(a) for a in artifacts] == [OutputArtifact]
