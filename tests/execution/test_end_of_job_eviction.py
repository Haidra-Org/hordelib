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
