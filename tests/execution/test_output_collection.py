"""Unit tests for declared-output collection: artifact tagging and kind resolution."""

import io

from hordelib.execution.in_process import InProcessComfyBackend
from hordelib.execution.interface import DEFAULT_IMAGE_OUTPUTS, OutputKind, OutputSpec


def _image_entry(source_node: str | None) -> dict:
    return {"imagedata": io.BytesIO(b"\x89PNG..."), "type": "PNG", "source_node": source_node}


def test_artifacts_are_tagged_with_source_node_and_kind():
    outputs = (OutputSpec(node="output_image"),)

    artifacts = InProcessComfyBackend._to_artifacts([_image_entry("output_image")], outputs)

    assert len(artifacts) == 1
    assert artifacts[0].source_node == "output_image"
    assert artifacts[0].kind is OutputKind.IMAGE
    assert artifacts[0].mime_type == "image/png"


def test_artifact_from_undeclared_node_still_collected_with_image_kind():
    outputs = (OutputSpec(node="output_image"),)

    artifacts = InProcessComfyBackend._to_artifacts([_image_entry("stray_save_image")], outputs)

    assert len(artifacts) == 1
    assert artifacts[0].source_node == "stray_save_image"
    assert artifacts[0].kind is OutputKind.IMAGE


def test_entry_without_imagedata_is_skipped():
    artifacts = InProcessComfyBackend._to_artifacts([{"type": "PNG"}], DEFAULT_IMAGE_OUTPUTS)

    assert artifacts == []


def test_default_outputs_are_the_single_image_convention():
    assert DEFAULT_IMAGE_OUTPUTS == (OutputSpec(node="output_image", kind=OutputKind.IMAGE),)
