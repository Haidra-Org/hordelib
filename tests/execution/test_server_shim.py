"""Unit tests for the headless PromptServer shim."""

import PIL.Image
import pytest

from hordelib.execution.server_shim import HeadlessComfyServer

_PREVIEW_IMAGE_WITH_METADATA_EVENT: int = 4


def test_events_are_forwarded_to_the_listener() -> None:
    received: list[tuple[str, dict, str | None]] = []
    shim = HeadlessComfyServer(event_listener=lambda label, data, sid: received.append((label, data, sid)))

    shim.send_sync("execution_start", {"prompt_id": "a"}, "client-1")
    shim.send_sync("executing", {"node": "sampler"})

    assert received == [
        ("execution_start", {"prompt_id": "a"}, "client-1"),
        ("executing", {"node": "sampler"}, None),
    ]


def test_binary_websocket_events_are_accepted_and_discarded() -> None:
    received: list[tuple[str, dict, str | None]] = []
    shim = HeadlessComfyServer(event_listener=lambda label, data, sid: received.append((label, data, sid)))

    preview_image = PIL.Image.new("RGB", (1, 1))
    preview_payload = (
        ("JPEG", preview_image, None),
        {
            "node_id": "sampler",
            "prompt_id": "prompt",
            "display_node_id": "sampler",
            "parent_node_id": None,
            "real_node_id": "sampler",
        },
    )

    shim.send_sync(_PREVIEW_IMAGE_WITH_METADATA_EVENT, preview_payload, "client-1")

    assert received == []


def test_lifecycle_events_require_dictionary_payloads() -> None:
    shim = HeadlessComfyServer(event_listener=lambda label, data, sid: None)

    with pytest.raises(TypeError, match="must contain a dictionary payload"):
        shim.send_sync("execution_start", ("not", "a", "dictionary"))


def test_shim_exposes_the_pinned_executor_surface() -> None:
    # The attribute names ComfyUI's executor reads/writes headless; the live pin is the
    # strict fake in tests/test_comfy_contract_drift.py.
    shim = HeadlessComfyServer(event_listener=lambda label, data, sid: None)

    assert shim.client_id is None
    assert shim.last_node_id is None
    assert shim.sockets_metadata == {}
    assert callable(shim.send_sync)
    assert callable(shim.queue_updated)


def test_executor_writable_attributes_are_plain_fields() -> None:
    shim = HeadlessComfyServer(event_listener=lambda label, data, sid: None)

    shim.client_id = "client-1"
    shim.last_node_id = "sampler"

    assert shim.client_id == "client-1"
    assert shim.last_node_id == "sampler"
