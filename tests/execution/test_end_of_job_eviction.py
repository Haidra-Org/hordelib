"""Pins the end-of-job VRAM eviction decision in ``Comfy_Horde._run_pipeline``.

The host's ``defer_vram_unload`` grant is the only thing that suppresses the eviction; with no
grant the bridge frees the card itself rather than relying on ComfyUI's memory mode to do it.
These run the CPU-only mini graph (EmptyImage into HordeImageOutput), so no model is loaded and
the evictor is observed through a stand-in rather than by its effect on a device.
"""

from typing import Any

import pytest

import hordelib.comfy_horde as comfy_horde
from hordelib.comfy_horde import Comfy_Horde


@pytest.fixture(scope="module")
def bridge(init_horde: None) -> Comfy_Horde:
    """A bridge with the default aggressive-unloading policy and custom nodes registered."""
    return Comfy_Horde()


def _mini_graph() -> dict[str, Any]:
    """Create a CPU-only API-format graph: EmptyImage feeding the horde output node."""
    return {
        "empty_image": {
            "class_type": "EmptyImage",
            "inputs": {"width": 64, "height": 64, "batch_size": 1, "color": 0},
        },
        "output_image": {
            "class_type": "HordeImageOutput",
            "inputs": {"images": ["empty_image", 0]},
        },
    }


def _record_evictor(monkeypatch: pytest.MonkeyPatch) -> list[None]:
    calls: list[None] = []
    monkeypatch.setattr(comfy_horde, "unload_all_models_vram", lambda: calls.append(None))
    return calls


def test_ungranted_job_evicts_once(bridge: Comfy_Horde, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _record_evictor(monkeypatch)

    bridge.run_pipeline(_mini_graph(), {})

    assert len(calls) == 1, "a run without a retention grant must free the card exactly once"


def test_granted_job_skips_eviction(bridge: Comfy_Horde, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _record_evictor(monkeypatch)

    bridge.run_pipeline(_mini_graph(), {}, defer_vram_unload=True)

    assert calls == [], "a retention grant must leave the model resident: no eviction at end of job"


def test_real_eviction_survives_a_model_free_run(bridge: Comfy_Horde) -> None:
    """The real evictor runs in the finally of every ungranted job, so it must tolerate a bare device.

    Nothing is loaded by the mini graph; a full free with no resident models has to be a no-op rather
    than an exception, or an eviction failure would surface as a failed job.
    """
    results = bridge.run_pipeline(_mini_graph(), {})

    assert len(results) == 1
    assert results[0]["imagedata"].getvalue().startswith(b"\x89PNG")


def test_non_aggressive_bridge_never_evicts(init_horde: None, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _record_evictor(monkeypatch)

    Comfy_Horde(aggressive_unloading=False).run_pipeline(_mini_graph(), {})

    assert calls == [], "a host that opted out of aggressive unloading owns eviction itself"


def _pin_device_residency(monkeypatch: pytest.MonkeyPatch, *, empty: bool) -> None:
    """Pin what the device holds at the end of a run, in place of a real ComfyUI residency."""
    monkeypatch.setattr(comfy_horde, "device_holds_no_loaded_model", lambda: empty)


def test_a_granted_run_reports_the_device_dropping_its_weights(
    bridge: Comfy_Horde,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A deferral that ends with an empty device kept nothing, and the run says so.

    ComfyUI frees other models on the device to fund an allocation, which takes a checkpoint held
    under a retention grant with it. The host predicts residency at dispatch and cannot observe that
    from outside the process, so a run that was granted the deferral has to report the divergence or
    the host charges and routes for weights the card does not have.
    """
    _pin_device_residency(monkeypatch, empty=True)

    bridge.run_pipeline(_mini_graph(), {}, defer_vram_unload=True)

    assert bridge.last_run_retained_weights_evicted is True


def test_a_granted_run_with_weights_still_resident_reports_nothing(
    bridge: Comfy_Horde,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ordinary granted run: the device still holds a model, so the grant was honoured."""
    _pin_device_residency(monkeypatch, empty=False)

    bridge.run_pipeline(_mini_graph(), {}, defer_vram_unload=True)

    assert bridge.last_run_retained_weights_evicted is False


def test_an_ungranted_run_never_reports_an_eviction(
    bridge: Comfy_Horde,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without a grant the run's own evictor empties the device, so an empty device says nothing."""
    _record_evictor(monkeypatch)
    _pin_device_residency(monkeypatch, empty=True)

    bridge.run_pipeline(_mini_graph(), {})

    assert bridge.last_run_retained_weights_evicted is False


def test_device_residency_counts_models_on_the_inference_device(monkeypatch: pytest.MonkeyPatch) -> None:
    """The reading is per device, and an entry whose device cannot be read counts as being on it.

    Reporting an eviction that did not happen costs the host a retained copy it then reloads, so the
    unrecognised case resolves toward "something is loaded".
    """

    class _Loaded:
        def __init__(self, device: object) -> None:
            self.device = device

    # The device accessor is bound onto the module by hordelib.initialise(), so it is absent in a
    # CPU-only test process; the reading has to answer without it either way.
    monkeypatch.setattr(comfy_horde, "_comfy_get_torch_device", lambda: "cuda:0", raising=False)

    monkeypatch.setattr(comfy_horde, "_comfy_current_loaded_models", [])
    assert comfy_horde.device_holds_no_loaded_model() is True

    monkeypatch.setattr(comfy_horde, "_comfy_current_loaded_models", [_Loaded("cpu")])
    assert comfy_horde.device_holds_no_loaded_model() is True

    monkeypatch.setattr(comfy_horde, "_comfy_current_loaded_models", [_Loaded("cuda:0")])
    assert comfy_horde.device_holds_no_loaded_model() is False

    monkeypatch.setattr(comfy_horde, "_comfy_current_loaded_models", [object()])
    assert comfy_horde.device_holds_no_loaded_model() is False


class _DeadRefLoadedModel:
    """A loaded-model entry whose patcher reference has been collected, as ComfyUI leaves them."""

    @property
    def model(self) -> None:
        return None


class _LiveLoadedModel:
    """A loaded-model entry that still answers for itself."""

    def __init__(self) -> None:
        self.model = object()


def test_a_full_unload_reports_what_the_device_gave_back(
    bridge: Comfy_Horde,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The unload is judged by its result: what came back, what is still listed, what it still holds.

    ComfyUI frees by walking its loaded-model list and skipping anything a live reference still pins, and
    it says nothing about what it skipped. A caller that reports the weights moved to host RAM on the
    strength of having asked keeps a ledger the card disagrees with for the rest of the session.
    """
    monkeypatch.setattr(comfy_horde, "_comfy_current_loaded_models", [_LiveLoadedModel()])
    monkeypatch.setattr(comfy_horde, "_remaining_loaded_weights_mb", lambda: 6800.0)

    result = comfy_horde.unload_all_models_vram()

    assert result.remaining_loaded_models == 1
    assert result.remaining_loaded_weights_mb == 6800.0
    assert result.complete is False, "weights still on the card cannot read as a completed unload"


def test_an_unload_that_emptied_the_device_reads_as_complete(
    bridge: Comfy_Horde,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nothing listed afterwards is the ordinary outcome and needs no weight reading to confirm."""
    monkeypatch.setattr(comfy_horde, "_comfy_current_loaded_models", [])

    result = comfy_horde.unload_all_models_vram()

    assert result.remaining_loaded_models == 0
    assert result.dead_model_refs_dropped == 0
    assert result.complete is True


def test_a_dead_reference_entry_is_dropped_and_counted(
    bridge: Comfy_Horde,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An entry whose patcher is gone is removed rather than left to make every unload read as incomplete.

    It pins nothing and answers nothing: every accessor on it raises, and no free can ever unload it. This
    is the same removal ComfyUI's own ``cleanup_models`` performs for its dead references.
    """
    loaded = [_DeadRefLoadedModel(), _DeadRefLoadedModel()]
    monkeypatch.setattr(comfy_horde, "_comfy_current_loaded_models", loaded)

    result = comfy_horde.unload_all_models_vram()

    assert result.dead_model_refs_dropped == 2
    assert loaded == []
    assert result.remaining_loaded_models == 0
    assert result.complete is True


def test_an_unreadable_weight_figure_is_not_read_as_an_empty_device(
    bridge: Comfy_Horde,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A figure this process cannot produce is None, and a caller must not take it for zero."""
    monkeypatch.setattr(comfy_horde, "_comfy_current_loaded_models", [_LiveLoadedModel()])
    monkeypatch.setattr(comfy_horde, "_remaining_loaded_weights_mb", lambda: None)

    result = comfy_horde.unload_all_models_vram()

    assert result.remaining_loaded_weights_mb is None
    assert result.complete is False


def test_a_small_remainder_does_not_condemn_an_unload(
    bridge: Comfy_Horde,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Residue well under any checkpoint is ordinary, so the verdict has a floor rather than a zero test."""
    monkeypatch.setattr(comfy_horde, "_comfy_current_loaded_models", [_LiveLoadedModel()])
    monkeypatch.setattr(comfy_horde, "_remaining_loaded_weights_mb", lambda: 8.0)

    result = comfy_horde.unload_all_models_vram()

    assert result.complete is True
