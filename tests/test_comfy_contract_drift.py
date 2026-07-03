"""Pins the ComfyUI internals hordelib's bridge couples to, against the pinned checkout.

The bridge (``hordelib/comfy_horde.py`` and ``hordelib/execution/``) relies on a small set of
ComfyUI behaviors that are not covered by any formal API contract: the ``PromptExecutor``
result attributes, the event labels delivered through the duck-typed server's ``send_sync``,
the ``validate_prompt`` tuple shape, the signatures the monkeypatches wrap, and the
``folder_paths`` surface. When a ComfyUI version bump changes any of these, this module is
what fails first, with a named assertion instead of a silent behavior change deep inside a
GPU run.

Everything here runs on CPU; the mini-execution round trip uses ``EmptyImage`` feeding
``HordeImageOutput``, so it exercises the full executor path without loading any model.
"""

import asyncio
import inspect
import io
from typing import Any

import pytest

from hordelib.comfy_horde import Comfy_Horde

_KNOWN_EVENT_LABELS = {
    "execution_start",
    "execution_cached",
    "executing",
    "executed",
    "progress_state",
    "execution_error",
    "execution_interrupted",
    "execution_success",
}
"""Every event label ComfyUI's execution path can deliver to the server's ``send_sync``.

A label outside this set means ComfyUI grew a new event channel the bridge does not know
about; extend the typed event layer (``hordelib.execution.comfy_events``) before extending
this set.
"""

_EXPECTED_SERVER_SURFACE = frozenset({"client_id", "last_node_id", "sockets_metadata", "send_sync"})
"""The complete server surface ComfyUI's executor touches when running headless.

``client_id`` is read and written (``execute_async`` assigns it from ``extra_data``),
``last_node_id`` is written per node, ``send_sync`` receives every event, and
``sockets_metadata`` is read only when preview images are enabled (defined defensively).
"""


class _UnexpectedServerAccessError(AssertionError):
    """ComfyUI touched a server attribute outside the pinned headless surface."""


class _StrictRecordingServer:
    """A duck-typed PromptServer stand-in exposing exactly the pinned headless surface.

    Any attribute access outside ``_EXPECTED_SERVER_SURFACE`` raises, so growth in the
    surface ComfyUI expects from its server object is discovered here rather than as an
    AttributeError mid-run in production.
    """

    def __init__(self) -> None:
        self.client_id: str | None = None
        self.last_node_id: str | None = None
        self.sockets_metadata: dict[str, Any] = {}
        self.events: list[tuple[str, dict[str, Any], str | None]] = []

    def send_sync(self, label: str, data: dict[str, Any], sid: str | None = None) -> None:
        """Record an event delivered by the executor."""
        self.events.append((label, data, sid))

    def __getattr__(self, name: str) -> Any:
        raise _UnexpectedServerAccessError(
            f"ComfyUI accessed server attribute {name!r}, which is outside the pinned headless "
            f"server surface {sorted(_EXPECTED_SERVER_SURFACE)}. The executor's server contract has "
            "grown; extend the hordelib server shim (and this pin) deliberately.",
        )


_FAILING_NODE_CLASS_TYPE = "HordeDriftTestFailingNode"


class _FailingOutputNode:
    """An output node whose execution always raises, to pin the error-path payload shape."""

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:  # ComfyUI node contract requires this exact name
        """Return the ComfyUI input schema: a single required IMAGE input."""
        return {"required": {"images": ("IMAGE",)}}

    RETURN_TYPES: tuple = ()
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = "image"

    def run(self, images: Any) -> dict[str, Any]:
        """Raise unconditionally so the executor takes its error path."""
        raise RuntimeError("drift-test deliberate failure")


@pytest.fixture(scope="module")
def comfy_bridge(init_horde: None) -> Comfy_Horde:
    """A constructed bridge, ensuring custom nodes (HordeImageOutput) are registered."""
    return Comfy_Horde()


def _mini_graph(output_class_type: str = "HordeImageOutput") -> dict[str, Any]:
    """Create a CPU-only API-format graph: EmptyImage feeding a single output node."""
    return {
        "empty_image": {
            "class_type": "EmptyImage",
            "inputs": {"width": 64, "height": 64, "batch_size": 1, "color": 0},
        },
        "output_image": {
            "class_type": output_class_type,
            "inputs": {"images": ["empty_image", 0]},
        },
    }


def _build_executor(server: _StrictRecordingServer) -> Any:
    """Create a real PromptExecutor around the strict fake server, mirroring _get_executor."""
    import execution

    return execution.PromptExecutor(
        server,
        cache_type=execution.CacheType.CLASSIC,
        cache_args={"lru": 0, "ram": 0.0, "ram_inactive": 0.0},
    )


def _validate(graph: dict[str, Any]) -> tuple:
    """Run ComfyUI's async validate_prompt the same way the bridge does."""
    from execution import validate_prompt

    return asyncio.run(validate_prompt(1, graph, None))


class TestMiniExecutionRoundTrip:
    """The full executor path on CPU: validation shape, results, events, server surface."""

    def test_success_path_contract(self, comfy_bridge: Comfy_Horde) -> None:
        graph = _mini_graph()

        valid = _validate(graph)
        assert isinstance(valid, tuple)
        assert len(valid) == 4, "validate_prompt no longer returns a 4-tuple"
        is_valid, error, output_node_ids, node_errors = valid
        assert is_valid is True, f"mini graph failed validation: {error}"
        assert error is None
        assert output_node_ids == ["output_image"]
        assert isinstance(node_errors, dict)

        server = _StrictRecordingServer()
        executor = _build_executor(server)
        executor.execute(graph, "drift-test-prompt", {"client_id": "drift-test-client"}, output_node_ids)

        # The executor's post-run attributes are the bridge's output-retrieval channel.
        assert executor.success is True
        assert isinstance(executor.status_messages, list)
        history_result = getattr(executor, "history_result", None)
        assert history_result is not None, "PromptExecutor.history_result was not assigned after execute()"
        assert set(history_result) >= {"outputs", "meta"}

        output_ui = history_result["outputs"]["output_image"]
        image_entries = output_ui["images"]
        assert len(image_entries) == 1
        first_entry = image_entries[0]
        assert isinstance(first_entry["imagedata"], io.BytesIO), (
            "HordeImageOutput ui entries no longer carry an in-memory BytesIO; the file-less "
            "output contract has drifted (check enrich_output_with_assets behavior too)"
        )
        assert first_entry["type"] == "PNG"
        assert first_entry["imagedata"].getvalue().startswith(b"\x89PNG")

        assert "output_image" in history_result["meta"]

        observed_labels = {label for label, _, _ in server.events}
        unknown_labels = observed_labels - _KNOWN_EVENT_LABELS
        assert not unknown_labels, (
            f"ComfyUI emitted event label(s) {sorted(unknown_labels)} the bridge does not know about"
        )

        # Every event ComfyUI actually emitted must parse into a typed model, not UnknownEvent.
        from hordelib.execution.comfy_events import UnknownEvent, parse_event

        for event_label, event_data, _ in server.events:
            parsed = parse_event(event_label, event_data)
            assert not isinstance(parsed, UnknownEvent), (
                f"live event {event_label!r} fell through typed parsing: {event_data}"
            )
        assert "execution_start" in observed_labels
        assert "execution_success" in observed_labels
        assert "executed" in observed_labels

        # execute_async assigns client_id from extra_data; the executor nulls last_node_id at the end.
        assert server.client_id == "drift-test-client"
        assert server.last_node_id is None

    def test_error_path_contract(self, comfy_bridge: Comfy_Horde) -> None:
        import execution

        execution.nodes.NODE_CLASS_MAPPINGS[_FAILING_NODE_CLASS_TYPE] = _FailingOutputNode
        try:
            graph = _mini_graph(output_class_type=_FAILING_NODE_CLASS_TYPE)
            valid = _validate(graph)
            assert valid[0] is True, f"failing-node graph should validate cleanly: {valid[1]}"

            server = _StrictRecordingServer()
            executor = _build_executor(server)
            executor.execute(graph, "drift-test-error-prompt", {"client_id": "drift-test-client"}, valid[2])

            assert executor.success is False

            error_messages = [data for label, data, _ in server.events if label == "execution_error"]
            assert len(error_messages) == 1, "expected exactly one execution_error event"
            error_payload = error_messages[0]
            expected_error_keys = {
                "prompt_id",
                "node_id",
                "node_type",
                "executed",
                "exception_message",
                "exception_type",
                "traceback",
                "current_inputs",
                "current_outputs",
            }
            assert expected_error_keys <= set(error_payload), (
                f"execution_error payload lost key(s): {sorted(expected_error_keys - set(error_payload))}"
            )
            assert error_payload["node_id"] == "output_image"
            assert error_payload["node_type"] == _FAILING_NODE_CLASS_TYPE
            assert "drift-test deliberate failure" in error_payload["exception_message"]

            # history_result is still assigned on the handled-error path (the loop break falls
            # through to the assignment); the failed output node simply has no entry.
            history_result = getattr(executor, "history_result", None)
            assert history_result is not None
            assert "output_image" not in history_result["outputs"]

            # status_messages carries the same error payload for post-run retrieval.
            status_error_events = [data for event, data in executor.status_messages if event == "execution_error"]
            assert len(status_error_events) == 1
        finally:
            execution.nodes.NODE_CLASS_MAPPINGS.pop(_FAILING_NODE_CLASS_TYPE, None)

    def test_bridge_run_pipeline_round_trip(self, comfy_bridge: Comfy_Horde) -> None:
        """The full bridge path (validate, execute, history_result collection) on CPU."""
        results = comfy_bridge.run_pipeline(_mini_graph(), {})

        assert len(results) == 1
        entry = results[0]
        assert entry["source_node"] == "output_image"
        assert entry["type"] == "PNG"
        assert isinstance(entry["imagedata"], io.BytesIO)
        assert entry["imagedata"].getvalue().startswith(b"\x89PNG")

    def test_bridge_run_pipeline_error_raises_with_typed_summary(self, comfy_bridge: Comfy_Horde) -> None:
        """A failing node surfaces as the historical RuntimeError, now carrying error context."""
        import execution

        execution.nodes.NODE_CLASS_MAPPINGS[_FAILING_NODE_CLASS_TYPE] = _FailingOutputNode
        try:
            with pytest.raises(RuntimeError, match="Pipeline failed to run") as raised:
                comfy_bridge.run_pipeline(_mini_graph(output_class_type=_FAILING_NODE_CLASS_TYPE), {})
            assert "drift-test deliberate failure" in str(raised.value)
            assert "output_image" in str(raised.value)
        finally:
            execution.nodes.NODE_CLASS_MAPPINGS.pop(_FAILING_NODE_CLASS_TYPE, None)

    def test_cached_output_delivery_requires_client_id(self, comfy_bridge: Comfy_Horde) -> None:
        """Cached output nodes reach ui outputs only via _send_cached_ui, which needs client_id.

        The bridge always passes ``client_id`` in ``extra_data``; this pin documents why that
        must not change once output retrieval reads ``history_result``.
        """
        import execution

        send_cached_ui = execution._send_cached_ui
        signature = inspect.signature(send_cached_ui)
        assert list(signature.parameters) == [
            "server",
            "node_id",
            "display_node_id",
            "cached",
            "prompt_id",
            "ui_outputs",
        ]

        source = inspect.getsource(send_cached_ui)
        assert "client_id is None" in source, (
            "_send_cached_ui no longer early-returns on a missing client_id; "
            "re-verify the cached-output delivery path before trusting history_result for cached nodes"
        )


class TestV3CanaryNode:
    """Proves the comfy_api V3 extension path works in hordelib's headless embedding.

    hordelib policy: new nodes (especially new modalities) are written V3; the existing
    classic nodes stay classic. This canary run is what that policy rests on.
    """

    def test_v3_node_registered_via_comfy_entrypoint(self, comfy_bridge: Comfy_Horde) -> None:
        import execution

        assert "HordeV3CanaryOutput" in execution.nodes.NODE_CLASS_MAPPINGS, (
            "the V3 canary did not register; ComfyUI's comfy_entrypoint/ComfyExtension "
            "custom-node path no longer works headless"
        )

    def test_v3_output_node_round_trip(self, comfy_bridge: Comfy_Horde) -> None:
        """A V3 output node executes headless and honors the BytesIO ui-entry contract."""
        results = comfy_bridge.run_pipeline(_mini_graph(output_class_type="HordeV3CanaryOutput"), {})

        assert len(results) == 1
        entry = results[0]
        assert entry["source_node"] == "output_image"
        assert entry["type"] == "PNG"
        assert isinstance(entry["imagedata"], io.BytesIO)
        assert entry["imagedata"].getvalue().startswith(b"\x89PNG")


class TestProgressLifecyclePins:
    """Why the bridge keeps the global progress hook instead of the ProgressRegistry."""

    def test_reset_progress_state_discards_registered_handlers(self, init_horde: None) -> None:
        from comfy_execution import progress

        registry_before = progress.get_progress_state()
        handler = progress.CLIProgressHandler()
        progress.add_progress_handler(handler)
        assert handler.name in progress.get_progress_state().handlers

        from comfy_execution.graph import DynamicPrompt

        progress.reset_progress_state("drift-test", DynamicPrompt({}))

        registry_after = progress.get_progress_state()
        assert registry_after is not registry_before
        assert handler.name not in registry_after.handlers, (
            "reset_progress_state now preserves handlers; the ProgressRegistry may have become "
            "a viable persistent coupling point (revisit the global-hook decision)"
        )

    def test_global_progress_hook_seam_exists(self, init_horde: None) -> None:
        import comfy.utils

        assert callable(comfy.utils.set_progress_bar_global_hook)
        hook_params = list(inspect.signature(comfy.utils.set_progress_bar_global_hook).parameters)
        assert len(hook_params) == 1


class TestMonkeypatchSignaturePins:
    """The comfy signatures hordelib's policy monkeypatches wrap (see comfy_patches.py)."""

    def test_load_models_gpu_accepts_force_full_load(self, init_horde: None) -> None:
        from hordelib.execution.comfy_patches import _originals

        original_load_models_gpu = _originals.get("load_models_gpu")
        assert original_load_models_gpu is not None, "load_models_gpu monkeypatch was never installed"
        parameters = inspect.signature(original_load_models_gpu).parameters
        assert "force_full_load" in parameters
        assert "memory_required" in parameters

    def test_free_memory_accepts_positional_amount_and_device(self, init_horde: None) -> None:
        # Inspect the pristine handle the bridge captured at import time; the module attribute
        # is later wrapped by logfire instrumentation into an opaque (*args, **kwargs) signature.
        from hordelib import comfy_horde

        parameters = list(inspect.signature(comfy_horde._comfy_free_memory).parameters)
        assert parameters[:2] == ["memory_required", "device"]

    def test_model_patcher_load_signature(self, init_horde: None) -> None:
        from hordelib.execution.comfy_patches import _originals

        original_patcher_load = _originals.get("model_patcher_load")
        assert original_patcher_load is not None, "ModelPatcher.load monkeypatch was never installed"
        parameters = inspect.signature(original_patcher_load).parameters
        assert "full_load" in parameters

    def test_lora_calculate_weight_exists(self, init_horde: None) -> None:
        from hordelib.execution.comfy_patches import _originals

        original_calculate_weight = _originals.get("lora_calculate_weight")
        assert original_calculate_weight is not None, "calculate_weight monkeypatch was never installed"
        parameters = list(inspect.signature(original_calculate_weight).parameters)
        assert parameters[:3] == ["patches", "weight", "key"]

    def test_text_encoder_initial_device_patched(self, init_horde: None) -> None:
        from hordelib.execution.comfy_patches import _originals

        assert _originals.get("text_encoder_initial_device") is not None


class TestFolderPathsPins:
    """The folder_paths surface the bridge (and Phase 3's model_dirs) relies on."""

    def test_public_setter_api_exists(self, init_horde: None) -> None:
        import folder_paths

        assert callable(folder_paths.add_model_folder_path)
        assert callable(folder_paths.get_folder_paths)
        assert callable(folder_paths.get_full_path)
        assert callable(folder_paths.get_filename_list)

    def test_filename_list_cache_is_category_keyed_dict(self, init_horde: None) -> None:
        import folder_paths

        assert isinstance(folder_paths.filename_list_cache, dict), (
            "filename_list_cache is no longer a plain dict; update the embeddings cache "
            "invalidation in the bridge"
        )
