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

    def test_model_patcher_unpatch_model_signature(self, init_horde: None) -> None:
        from hordelib.execution.comfy_patches import _originals

        original_unpatch = _originals.get("model_patcher_unpatch_model")
        assert original_unpatch is not None, "ModelPatcher.unpatch_model monkeypatch was never installed"
        parameters = list(inspect.signature(original_unpatch).parameters)
        # The hijack passes (patcher, device_to, unpatch_weights) positionally and restores CPU weights only
        # for an unload to the offload device with weights unpatched, so both names must survive.
        assert parameters[:3] == ["self", "device_to", "unpatch_weights"]

    def test_lora_calculate_weight_exists(self, init_horde: None) -> None:
        from hordelib.execution.comfy_patches import _originals

        original_calculate_weight = _originals.get("lora_calculate_weight")
        assert original_calculate_weight is not None, "calculate_weight monkeypatch was never installed"
        parameters = list(inspect.signature(original_calculate_weight).parameters)
        assert parameters[:3] == ["patches", "weight", "key"]

    def test_text_encoder_initial_device_patched(self, init_horde: None) -> None:
        from hordelib.execution.comfy_patches import _originals

        assert _originals.get("text_encoder_initial_device") is not None

    def test_ksampler_factory_signature(self, init_horde: None) -> None:
        from hordelib.execution.comfy_patches import _originals

        original_ksampler = _originals.get("ksampler_factory")
        assert original_ksampler is not None, "ksampler monkeypatch was never installed"
        parameters = list(inspect.signature(original_ksampler).parameters)
        assert parameters[0] == "sampler_name"

    def test_adaptive_sampler_function_still_receives_the_schedule(self, init_horde: None) -> None:
        """The bound reads its nominal step count from ``sigmas``, which only this seam exposes.

        ``ksampler`` builds ``dpm_adaptive``'s sampler function as a closure that forwards only
        ``sigma_min``/``sigma_max`` onward, so a comfy change that stopped handing the sampler
        function the full schedule would silently remove the bound's only source of truth.
        """
        import comfy.samplers

        from hordelib.execution.adaptive_sampler_bound import ADAPTIVE_SAMPLER_NAME
        from hordelib.execution.comfy_patches import _originals

        original_ksampler = _originals["ksampler_factory"]
        stock_sampler = original_ksampler(ADAPTIVE_SAMPLER_NAME)

        assert isinstance(stock_sampler, comfy.samplers.KSAMPLER)
        parameters = list(inspect.signature(stock_sampler.sampler_function).parameters)
        assert parameters[:3] == ["model", "noise", "sigmas"]

    def test_adaptive_sampler_is_bounded_by_the_patched_factory(self, init_horde: None) -> None:
        import comfy.k_diffusion.sampling
        import comfy.samplers

        from hordelib.execution.adaptive_sampler_bound import (
            ADAPTIVE_SAMPLER_NAME,
            bounded_dpm_adaptive_sampler_function,
        )

        bounded = comfy.samplers.ksampler(ADAPTIVE_SAMPLER_NAME)
        assert bounded.sampler_function is bounded_dpm_adaptive_sampler_function

        # Every fixed-schedule sampler must be left exactly as comfy built it.
        assert comfy.samplers.ksampler("euler").sampler_function is comfy.k_diffusion.sampling.sample_euler

    def test_every_mapped_sampler_still_exists_in_comfy(self, init_horde: None) -> None:
        """A mapped sampler comfy no longer offers degrades silently, so pin the whole map.

        ``KSampler.__init__`` substitutes its first sampler for any name it does not recognise, and
        the payload validator clamps unknown *horde* names to the default. Between them, a comfy
        rename turns a requested sampler into a different one with no error anywhere, which the
        horde would keep advertising as supported.
        """
        import comfy.samplers

        from hordelib.pipeline.constants import SAMPLERS_MAP

        unknown = {
            horde_name: comfy_name
            for horde_name, comfy_name in SAMPLERS_MAP.items()
            if comfy_name not in comfy.samplers.SAMPLER_NAMES
        }
        assert unknown == {}, f"SAMPLERS_MAP targets samplers comfy does not offer: {unknown}"

    def test_every_scheduler_still_exists_in_comfy(self, init_horde: None) -> None:
        """The scheduler list is offered to callers verbatim, and comfy substitutes silently too."""
        import comfy.samplers

        from hordelib.pipeline.constants import SCHEDULERS, SIGMA_GENERATOR_SCHEDULES

        unknown = [
            name
            for name in SCHEDULERS
            if name not in comfy.samplers.SCHEDULER_NAMES and name not in SIGMA_GENERATOR_SCHEDULES
        ]
        assert unknown == [], f"SCHEDULERS lists schedulers comfy does not offer: {unknown}"

    def test_the_generator_schedules_are_still_unnameable_to_comfy(self, init_horde: None) -> None:
        """The two node-supplied schedules are exempt from the pin above only while comfy has no name.

        If comfy grows a handler for either, this package should stop computing it and pass the name
        through instead, so the exemption has to end the moment the reason for it does.
        """
        import comfy.samplers

        from hordelib.pipeline.constants import SIGMA_GENERATOR_SCHEDULES

        named_by_comfy = sorted(SIGMA_GENERATOR_SCHEDULES & set(comfy.samplers.SCHEDULER_NAMES))
        assert named_by_comfy == [], (
            f"comfy now resolves {named_by_comfy} itself; drop the sigma-generator override for it "
            "(hordelib.execution.sigma_schedules) and let calculate_sigmas handle the name"
        )

    def test_calculate_sigmas_signature(self, init_horde: None) -> None:
        """The sigma-generator patch replaces this function, so its arguments are the contract."""
        from hordelib.execution.comfy_patches import _originals

        original_calculate_sigmas = _originals.get("calculate_sigmas")
        assert original_calculate_sigmas is not None, "calculate_sigmas monkeypatch was never installed"
        parameters = list(inspect.signature(original_calculate_sigmas).parameters)
        assert parameters == ["model_sampling", "scheduler_name", "steps"]

    def test_the_ksampler_reads_calculate_sigmas_as_a_module_global(self, init_horde: None) -> None:
        """Patching the module attribute only reaches KSampler while it looks the function up there.

        A comfy refactor that bound the function into the class (or imported it into another module)
        would leave the patch installed and inert, so the graph would run an unrequested schedule.
        """
        import comfy.samplers

        source = inspect.getsource(comfy.samplers.KSampler.calculate_sigmas)
        assert "calculate_sigmas(self.model.get_model_object" in source, (
            "KSampler.calculate_sigmas no longer calls the module-level calculate_sigmas directly; "
            "re-verify the sigma-generator patch seam"
        )


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
            "filename_list_cache is no longer a plain dict; update the embeddings cache invalidation in the bridge"
        )


def _tiny_cpu_patcher() -> Any:
    """Build a ``ModelPatcher`` around a two-linear-layer module, resident and offloaded on the CPU.

    ``ModelPatcher`` only requires a ``torch.nn.Module``; everything the residency accounting touches
    (``model_loaded_weight_memory``, ``model_lowvram``, ``current_weight_patches_uuid``) is attached by its
    constructor. A module this small exercises the same code paths a checkpoint does, on CPU.
    """
    import comfy.model_patcher
    import torch

    model = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Linear(8, 8))
    cpu = torch.device("cpu")
    return comfy.model_patcher.ModelPatcher(model, load_device=cpu, offload_device=cpu)


def _strip_patched_weight_marks(patcher: Any) -> int:
    """Delete ``comfy_patched_weights`` from every module, as another patcher's unpatch would.

    ``ModelPatcher.unpatch_model`` with ``unpatch_weights=True`` deletes the attribute from every module of
    the model it shares. Returns how many modules carried the mark.
    """
    marked = 0
    for module in patcher.model.modules():
        if hasattr(module, "comfy_patched_weights"):
            del module.comfy_patched_weights
            marked += 1
    return marked


class TestModelPatcherResidencyPins:
    """The ``ModelPatcher`` residency accounting the worker's cost model and reclaim ladder assume.

    Three behaviors decide what an already-resident model costs to serve again: whether comfy can recognise
    an identical patch set (it cannot, ``add_patches`` rerolls ``patches_uuid`` with no content hash),
    whether an unpatch retracts the accounting unconditionally, and whether a partial unload that freed
    nothing still declares the model lowvram. The last is a latch: it disables ``partially_load``'s
    zero-cost fast return for every later load of that model.
    """

    def test_identical_patches_still_produce_distinct_uuids(self, init_horde: None) -> None:
        """Two clones given byte-identical patches at identical strength do not compare as matching.

        ``patches_uuid`` is a fresh uuid4 per ``add_patches`` call, so it identifies the call rather than
        the patch content. A LoRA-bearing job therefore mismatches the module's recorded
        ``current_weight_patches_uuid`` even when it applies exactly what is already baked in.
        """
        import torch

        patcher = _tiny_cpu_patcher()
        patch_payload = {"0.weight": torch.zeros(8, 8)}

        clone_a = patcher.clone()
        clone_b = patcher.clone()
        assert clone_a.patches_uuid == clone_b.patches_uuid, "a clone must inherit the source patch identity"
        assert patcher.clone_has_same_weights(clone_a), "an unmodified clone must compare as matching"

        assert clone_a.add_patches(patch_payload, 1.0) == ["0.weight"]
        assert clone_b.add_patches(patch_payload, 1.0) == ["0.weight"]

        assert clone_a.patches_uuid != clone_b.patches_uuid, (
            "add_patches now derives patches_uuid from the patch content; the LoRA reload cost the worker "
            "budgets for may no longer be paid on repeat jobs"
        )
        assert len(clone_a.patches) == len(clone_b.patches) == 1
        assert not clone_a.clone_has_same_weights(clone_b), (
            "clone_has_same_weights now recognises equal patch sets across clones; re-check whether a repeat "
            "LoRA job still forces a full unpatch and re-upload"
        )

    def test_unpatch_retracts_accounting_with_an_empty_backup(self, init_horde: None) -> None:
        """A weight-unpatch zeroes the loaded-weight accounting even when there is nothing to restore.

        The retraction is unconditional, so ``partially_load``'s unpatch step gives up the whole resident
        footprint and the following ``load`` re-uploads it, regardless of how little was actually patched.
        """
        import torch

        cpu = torch.device("cpu")
        patcher = _tiny_cpu_patcher()
        patcher.load(cpu, full_load=True)

        assert patcher.model.model_loaded_weight_memory > 0, "a full load must register a resident footprint"
        assert patcher.model.current_weight_patches_uuid == patcher.patches_uuid
        assert patcher.backup == {}, "an unpatched load must not have created weight backups"

        patcher.unpatch_model(cpu, unpatch_weights=True)

        assert patcher.model.model_loaded_weight_memory == 0, (
            "unpatch_model no longer retracts the loaded-weight accounting unconditionally; the cost model "
            "for a repeat LoRA job needs re-deriving"
        )
        assert patcher.model.current_weight_patches_uuid is None
        assert not any(hasattr(module, "comfy_patched_weights") for module in patcher.model.modules()), (
            "unpatch_model no longer clears comfy_patched_weights from the shared modules"
        )

    def test_zero_byte_partial_unload_still_latches_lowvram(self, init_horde: None) -> None:
        """Freeing nothing still sets ``model_lowvram``, because the flag is written outside the free loop.

        ``partially_unload`` only frees modules whose ``comfy_patched_weights`` is truthy, and another
        patcher's unpatch of the shared model deletes that attribute from all of them. The unload then walks
        the whole list, frees zero bytes, and still declares the model lowvram.
        """
        import torch

        cpu = torch.device("cpu")
        patcher = _tiny_cpu_patcher()
        patcher.load(cpu, full_load=True)
        resident_before = patcher.model.model_loaded_weight_memory
        assert resident_before > 0
        assert patcher.model.model_lowvram is False

        assert _strip_patched_weight_marks(patcher) > 0, "a full load must mark its modules as patched"

        freed = patcher.partially_unload(cpu, memory_to_free=1 << 40)

        assert freed == 0, (
            "partially_unload now frees modules whose comfy_patched_weights mark was removed; the "
            "zero-byte-unload latch this pins may no longer exist"
        )
        assert patcher.model.model_lowvram is True, (
            "partially_unload no longer sets model_lowvram unconditionally; the latch is gone and the "
            "worker's assumption that a no-op unload poisons later loads needs revisiting"
        )
        assert patcher.model.model_loaded_weight_memory == resident_before, (
            "the accounting moved without any weights being freed"
        )

    def test_the_lowvram_latch_costs_every_later_load(self, init_horde: None) -> None:
        """Before the latch ``partially_load`` returns free; after it, the same call re-walks ``load``.

        The fast return requires ``model_lowvram`` false and a nonzero resident footprint, so one zero-byte
        partial unload converts every later load of that model into a full re-walk.
        """
        import torch

        cpu = torch.device("cpu")
        patcher = _tiny_cpu_patcher()
        patcher.load(cpu, full_load=True)

        real_load = patcher.load
        load_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

        def counting_load(*args: Any, **kwargs: Any) -> Any:
            load_calls.append((args, kwargs))
            return real_load(*args, **kwargs)

        # An instance attribute is enough: partially_load reaches load through self.
        patcher.load = counting_load

        assert patcher.patches_uuid == patcher.model.current_weight_patches_uuid
        assert patcher.partially_load(cpu) == 0
        assert load_calls == [], (
            "partially_load no longer short-circuits for an already-resident, uuid-matching model; the "
            "zero-cost warm path this pins is gone"
        )

        _strip_patched_weight_marks(patcher)
        patcher.partially_unload(cpu, memory_to_free=1 << 40)
        assert patcher.model.model_lowvram is True

        patcher.partially_load(cpu)
        assert len(load_calls) == 1, (
            "a latched-lowvram model no longer pays a load() walk on partially_load; the latch's cost is no "
            "longer what the worker's reclaim ladder assumes"
        )

    def test_bypass_lora_loader_signature(self, init_horde: None) -> None:
        """The bypass LoRA loader, which applies a LoRA without touching base weights, still exists."""
        import comfy.sd

        assert callable(comfy.sd.load_bypass_lora_for_models), (
            "comfy.sd.load_bypass_lora_for_models is gone; the no-bake LoRA path it provides has to be "
            "re-sourced before anything can rely on it"
        )
        parameters = list(inspect.signature(comfy.sd.load_bypass_lora_for_models).parameters)
        assert parameters[:5] == ["model", "clip", "lora", "strength_model", "strength_clip"]


class _StubPatcher:
    """The ``.model`` attribute ``free_memory`` reaches through on a loaded entry."""

    def is_dynamic(self) -> bool:
        """Report a static model; dynamic models get their own on-demand freeing branch."""
        return False


class _StubUnloadableModel:
    """A ``LoadedModel`` stand-in exposing only what ``free_memory`` reads, recording its ask."""

    def __init__(self, device: Any) -> None:
        self.device = device
        self.currently_used = True
        self.model = _StubPatcher()
        self.unload_asks: list[float] = []

    def is_dead(self) -> bool:
        """Report the entry as live so ``free_memory`` considers it unloadable."""
        return False

    def model_memory(self) -> int:
        """Report a nonzero footprint, as a resident model would."""
        return 1 << 30

    def model_offloaded_memory(self) -> int:
        """Report nothing offloaded, i.e. the whole footprint is on the device."""
        return 0

    def model_unload(self, memory_to_free: float | None = None, unpatch_weights: bool = True) -> bool:
        """Record the amount comfy asked to free and decline, leaving the entry in place."""
        self.unload_asks.append(-1.0 if memory_to_free is None else float(memory_to_free))
        return False


class TestMemoryModeGatePins:
    """Pins the two ComfyUI memory-mode behaviors hordelib's end-of-job eviction is designed around.

    hordelib evicts explicitly at the end of a job (suppressed only by the host's retention grant),
    and relies on ComfyUI running with smart memory enabled so that a granted model survives the
    job's own later loads. Both halves of that rest on the behaviors pinned here.
    """

    def test_end_of_prompt_unload_is_gated_on_disable_smart_memory(
        self,
        comfy_bridge: Comfy_Horde,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The executor unloads everything at the end of a prompt if and only if the mode is set."""
        import comfy.model_management

        graph = _mini_graph()
        output_node_ids = _validate(graph)[2]

        calls: list[bool] = []
        observed: dict[bool, int] = {}
        for disable_smart_memory in (True, False):
            mode = disable_smart_memory
            monkeypatch.setattr(comfy.model_management, "DISABLE_SMART_MEMORY", mode)
            monkeypatch.setattr(comfy.model_management, "unload_all_models", lambda mode=mode: calls.append(mode))

            executor = _build_executor(_StrictRecordingServer())
            executor.execute(
                graph,
                f"drift-test-mode-{mode}",
                {"client_id": "drift-test-client"},
                output_node_ids,
            )
            assert executor.success is True

            observed[mode] = calls.count(mode)

        assert observed[True] == 1, (
            "the executor no longer unloads all models at the end of a prompt under "
            "DISABLE_SMART_MEMORY; the flag no longer returns the card, so re-derive where hordelib's "
            "end-of-job eviction and the worker's retention grant actuate"
        )
        assert observed[False] == 0, (
            "the executor now unloads all models at the end of a prompt with smart memory enabled; a "
            "retention grant can no longer survive a prompt, so the worker's cross-job residency "
            "assumption has to be re-derived"
        )

    def test_free_memory_frees_by_shortfall_only_with_smart_memory(
        self,
        init_horde: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """With smart memory on, ``free_memory`` asks for the shortfall; with it off, for everything.

        The shortfall branch is what lets a retained model survive the loads that follow it in the
        same job; the unconditional branch is why it cannot under ``--disable-smart-memory``.
        """
        import comfy.model_management as mm

        device = mm.get_torch_device()
        stub = _StubUnloadableModel(device)
        mm.current_loaded_models.append(stub)
        try:
            # Smart memory on, no shortfall: comfy computes a non-positive ask and unloads nothing.
            monkeypatch.setattr(mm, "DISABLE_SMART_MEMORY", False)
            mm.free_memory(0, device)
            assert stub.unload_asks == [], (
                "free_memory unloaded a model with smart memory on and no memory shortfall; it no longer "
                "frees by shortfall, so intra-job and cross-job residency can no longer be assumed"
            )

            # Smart memory on, real shortfall: the ask is the shortfall, not everything.
            requested = mm.get_free_memory(device) + (1 << 30)
            mm.free_memory(requested, device)
            assert len(stub.unload_asks) == 1, "free_memory no longer asks a live entry to unload on a shortfall"
            assert 0 < stub.unload_asks[0] <= requested, (
                f"free_memory asked to free {stub.unload_asks[0]} against a shortfall bounded by "
                f"{requested}; the shortfall computation has changed"
            )

            # Smart memory off: the ask is unbounded regardless of what is actually needed.
            stub.unload_asks.clear()
            monkeypatch.setattr(mm, "DISABLE_SMART_MEMORY", True)
            mm.free_memory(0, device)
            assert len(stub.unload_asks) == 1, (
                "free_memory no longer unloads unconditionally under DISABLE_SMART_MEMORY; the "
                "unload-everything behavior hordelib's regime notes describe is gone"
            )
            assert stub.unload_asks[0] > (1 << 40), (
                f"free_memory asked to free only {stub.unload_asks[0]} under DISABLE_SMART_MEMORY; the "
                "branch is no longer unconditional and hordelib's memory-mode reasoning needs re-deriving"
            )
        finally:
            mm.current_loaded_models[:] = [entry for entry in mm.current_loaded_models if entry is not stub]
