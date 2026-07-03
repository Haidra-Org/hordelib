"""Unit tests for run-result construction and modality-agnostic output-entry collection."""

import io

from hordelib.execution.comfy_events import ExecutionErrorEvent
from hordelib.execution.results import PipelineRunResult, collect_output_entries


def _png_bytes() -> io.BytesIO:
    return io.BytesIO(b"\x89PNG\r\n\x1a\n...")


def test_collects_image_entries_tagged_with_source_node() -> None:
    history_outputs = {
        "output_image": {"images": [{"imagedata": _png_bytes(), "type": "PNG"}]},
    }

    entries = collect_output_entries(history_outputs)

    assert len(entries) == 1
    assert entries[0]["source_node"] == "output_image"
    assert entries[0]["type"] == "PNG"
    assert isinstance(entries[0]["imagedata"], io.BytesIO)


def test_collection_is_ui_key_agnostic_for_future_modalities() -> None:
    # A future audio output node reuses the BytesIO entry contract under a different ui key;
    # collection must pick it up without knowing the key.
    history_outputs = {
        "output_audio": {"audio": [{"imagedata": _png_bytes(), "type": "FLAC"}]},
    }

    entries = collect_output_entries(history_outputs)

    assert len(entries) == 1
    assert entries[0]["source_node"] == "output_audio"
    assert entries[0]["type"] == "FLAC"


def test_multiple_nodes_and_batched_entries_preserve_order() -> None:
    history_outputs = {
        "output_image": {"images": [{"imagedata": _png_bytes(), "type": "PNG"}] * 2},
        "second_output": {"images": [{"imagedata": _png_bytes(), "type": "PNG"}]},
    }

    entries = collect_output_entries(history_outputs)

    assert [entry["source_node"] for entry in entries] == ["output_image", "output_image", "second_output"]


def test_entries_without_bytesio_are_skipped() -> None:
    history_outputs = {
        "output_image": {
            "images": [
                {"type": "PNG"},
                {"imagedata": "not bytes", "type": "PNG"},
                {"imagedata": _png_bytes(), "type": "PNG"},
            ],
        },
    }

    entries = collect_output_entries(history_outputs)

    assert len(entries) == 1


def test_non_list_ui_values_and_non_dict_payloads_are_ignored() -> None:
    history_outputs = {
        "output_image": {"images": [{"imagedata": _png_bytes(), "type": "PNG"}], "text_note": "hello"},
        "weird_node": "not a dict",
        "other_node": {"animated": False},
    }

    entries = collect_output_entries(history_outputs)

    assert len(entries) == 1
    assert entries[0]["source_node"] == "output_image"


def test_run_result_produced_nodes() -> None:
    result = PipelineRunResult(
        success=True,
        entries=[
            {"imagedata": _png_bytes(), "type": "PNG", "source_node": "output_image"},
            {"imagedata": _png_bytes(), "type": "PNG", "source_node": "second_output"},
        ],
    )

    assert result.produced_nodes == {"output_image", "second_output"}


def test_run_result_carries_typed_error() -> None:
    error = ExecutionErrorEvent(node_id="sampler", node_type="KSampler", exception_message="boom")

    result = PipelineRunResult(success=False, error=error)

    assert result.entries == []
    assert result.error is not None
    assert "sampler (KSampler)" in result.error.summary()
