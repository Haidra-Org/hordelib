"""Unit tests for the typed ComfyUI event layer (parsing recorded wire payloads)."""

import pytest

from hordelib.execution.comfy_events import (
    ComfyEventLabel,
    ExecutedEvent,
    ExecutingEvent,
    ExecutionCachedEvent,
    ExecutionErrorEvent,
    ExecutionInterruptedEvent,
    ExecutionStartEvent,
    ExecutionSuccessEvent,
    ProgressStateEvent,
    UnknownEvent,
    ValidationResult,
    parse_event,
)


def test_execution_start_parses() -> None:
    event = parse_event("execution_start", {"prompt_id": "abc", "timestamp": 123})

    assert isinstance(event, ExecutionStartEvent)
    assert event.prompt_id == "abc"


def test_execution_error_parses_full_payload() -> None:
    payload = {
        "prompt_id": "abc",
        "node_id": "sampler",
        "node_type": "KSampler",
        "executed": ["model_loader"],
        "exception_message": "out of memory",
        "exception_type": "torch.OutOfMemoryError",
        "traceback": ["line one", "line two"],
        "current_inputs": {},
        "current_outputs": [],
        "timestamp": 123,
    }

    event = parse_event("execution_error", payload)

    assert isinstance(event, ExecutionErrorEvent)
    assert event.node_id == "sampler"
    assert event.node_type == "KSampler"
    assert event.exception_type == "torch.OutOfMemoryError"
    assert "sampler (KSampler): torch.OutOfMemoryError: out of memory" == event.summary()


def test_execution_error_summary_with_minimal_payload() -> None:
    event = parse_event("execution_error", {})

    assert isinstance(event, ExecutionErrorEvent)
    assert event.summary() == "unknown node: Exception: "


def test_execution_interrupted_parses() -> None:
    payload = {"prompt_id": "abc", "node_id": "sampler", "node_type": "KSampler", "executed": []}

    event = parse_event("execution_interrupted", payload)

    assert isinstance(event, ExecutionInterruptedEvent)
    assert event.node_id == "sampler"


def test_executed_event_carries_output_dict() -> None:
    payload = {
        "node": "output_image",
        "display_node": "output_image",
        "output": {"images": [{"type": "PNG"}]},
        "prompt_id": "abc",
    }

    event = parse_event("executed", payload)

    assert isinstance(event, ExecutedEvent)
    assert event.node == "output_image"
    assert event.output is not None
    assert "images" in event.output


def test_executing_and_success_and_cached_parse() -> None:
    executing = parse_event("executing", {"node": "vae_decode", "display_node": "vae_decode", "prompt_id": "a"})
    success = parse_event("execution_success", {"prompt_id": "a"})
    cached = parse_event("execution_cached", {"nodes": ["model_loader"], "prompt_id": "a"})

    assert isinstance(executing, ExecutingEvent)
    assert executing.node == "vae_decode"
    assert isinstance(success, ExecutionSuccessEvent)
    assert isinstance(cached, ExecutionCachedEvent)
    assert cached.nodes == ["model_loader"]


def test_progress_state_parses_node_map() -> None:
    payload = {
        "prompt_id": "a",
        "nodes": {
            "sampler": {"value": 5, "max": 20, "state": "running", "node_id": "sampler"},
        },
    }

    event = parse_event("progress_state", payload)

    assert isinstance(event, ProgressStateEvent)
    assert "sampler" in event.nodes


def test_unknown_label_becomes_unknown_event() -> None:
    event = parse_event("brand_new_comfy_event", {"anything": 1})

    assert isinstance(event, UnknownEvent)
    assert event.label == "brand_new_comfy_event"
    assert event.data == {"anything": 1}


def test_malformed_payload_becomes_unknown_event_not_exception() -> None:
    # nodes must be a list; a dict payload fails validation but must not raise.
    event = parse_event("execution_cached", {"nodes": 42})

    assert isinstance(event, UnknownEvent)
    assert event.label == "execution_cached"


def test_extra_payload_keys_are_retained() -> None:
    event = parse_event("execution_success", {"prompt_id": "a", "future_field": "kept"})

    assert isinstance(event, ExecutionSuccessEvent)
    assert event.model_extra is not None
    assert event.model_extra.get("future_field") == "kept"


def test_every_label_has_a_typed_model() -> None:
    for label in ComfyEventLabel:
        event = parse_event(label.value, {})
        assert not isinstance(event, UnknownEvent), f"label {label} fell through to UnknownEvent"


class TestValidationResult:
    def test_from_comfy_maps_the_four_tuple(self) -> None:
        raw = (True, None, ["output_image"], {})

        validation = ValidationResult.from_comfy(raw)

        assert validation.is_valid is True
        assert validation.error is None
        assert validation.output_node_ids == ["output_image"]
        assert validation.node_errors == {}

    def test_from_comfy_maps_failure(self) -> None:
        error = {"type": "prompt_outputs_failed_validation", "message": "bad", "details": "d", "extra_info": {}}
        raw = (False, error, [], {"sampler": [{"type": "value_smaller_than_min"}]})

        validation = ValidationResult.from_comfy(raw)

        assert validation.is_valid is False
        assert validation.error == error
        assert "sampler" in validation.node_errors

    def test_from_comfy_rejects_drifted_arity(self) -> None:
        with pytest.raises(ValueError, match="contract has drifted"):
            ValidationResult.from_comfy((True, None, []))
