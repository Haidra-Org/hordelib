"""Pins the clamp hordelib puts on ComfyUI's view of free VRAM during a pipeline run.

ComfyUI frees by shortfall against ``comfy.model_management.get_free_memory``, whose CUDA reading is
process-local: memory a sibling process holds is reported as free under WDDM, so the shortfall comes
out too small. A host that measures free VRAM at the device level passes that figure in as
``device_free_truth_mb`` and the scoped clamp lowers comfy's answer to it, less this process's own
allocator growth since the run started.

These are seam tests: the allocator and the device are stand-ins, so no GPU is required to observe
what comfy would read.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import pytest
import torch

import hordelib.comfy_horde as comfy_horde
from hordelib.comfy_horde import Comfy_Horde
from hordelib.execution import comfy_patches

MB = 1024 * 1024


@dataclass
class _DeviceSeam:
    """A stand-in CUDA device whose free-memory and allocator readings the test controls."""

    device: torch.device
    free_total: int
    free_torch: int
    reserved: int
    original: Any = None


@pytest.fixture
def seam(init_horde: None, monkeypatch: pytest.MonkeyPatch) -> Iterator[_DeviceSeam]:
    """A comfy memory surface reporting controllable readings for a CUDA device."""
    import comfy.model_management as model_management

    state = _DeviceSeam(
        device=torch.device("cuda", 0),
        free_total=4096 * MB,
        free_torch=0,
        reserved=1024 * MB,
    )

    def fake_get_free_memory(dev: Any = None, torch_free_too: bool = False) -> Any:
        if torch_free_too:
            return (state.free_total, state.free_torch)
        return state.free_total

    state.original = fake_get_free_memory
    monkeypatch.setattr(model_management, "get_free_memory", fake_get_free_memory)
    monkeypatch.setattr(model_management, "get_torch_device", lambda: state.device)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda dev=None: state.reserved)
    yield state


def _free_memory(dev: Any = None, torch_free_too: bool = False) -> Any:
    import comfy.model_management as model_management

    return model_management.get_free_memory(dev, torch_free_too)


def _current_reader() -> Any:
    import comfy.model_management as model_management

    return model_management.get_free_memory


def test_no_truth_figure_leaves_the_reading_alone(seam: _DeviceSeam) -> None:
    with comfy_patches.free_memory_view_clamped(None):
        assert _current_reader() is seam.original, "no truth figure must mean no interposition"
        assert _free_memory(seam.device) == seam.free_total

    assert _current_reader() is seam.original


def test_reading_is_clamped_to_truth_less_own_growth(seam: _DeviceSeam) -> None:
    with comfy_patches.free_memory_view_clamped(2048):
        assert _free_memory(seam.device) == 2048 * MB, "comfy's higher reading must lose to the host's figure"

        seam.reserved += 512 * MB
        assert _free_memory(seam.device) == 1536 * MB, "this process's own allocator growth must come off the top"

        seam.reserved -= 768 * MB
        assert _free_memory(seam.device) == 2048 * MB, "shrinking below the baseline must not credit free memory"


def test_comfy_reading_wins_when_it_is_lower(seam: _DeviceSeam) -> None:
    seam.free_total = 512 * MB

    with comfy_patches.free_memory_view_clamped(2048):
        assert _free_memory(seam.device) == 512 * MB


def test_reclaimable_pool_raises_the_ceiling(seam: _DeviceSeam) -> None:
    """Memory free inside torch's own pool is obtainable without evicting anything, so it counts.

    ComfyUI defines its own free total the same way (device free plus the allocator's cached-but-idle
    blocks). Leaving the term out would invent a shortfall at the decode-time load that follows
    sampling, when the cache is at its largest, and cost the resident diffusion model its residency.
    """
    seam.free_torch = 512 * MB

    with comfy_patches.free_memory_view_clamped(2048):
        assert _free_memory(seam.device) == 2560 * MB

        seam.reserved += 1024 * MB
        assert _free_memory(seam.device) == 1536 * MB


def test_growth_beyond_the_truth_figure_floors_at_the_reclaimable_pool(seam: _DeviceSeam) -> None:
    with comfy_patches.free_memory_view_clamped(1024):
        seam.reserved += 2048 * MB

        assert _free_memory(seam.device) == 0, "an exhausted budget reports no free memory, never a negative"

        seam.free_torch = 128 * MB
        assert _free_memory(seam.device) == 128 * MB, "what the allocator can hand back is still free"


def test_torch_free_pair_is_preserved(seam: _DeviceSeam) -> None:
    """The pair form keeps its shape; only the total is clamped, the torch-pool figure is reported as-is."""
    seam.free_torch = 512 * MB

    with comfy_patches.free_memory_view_clamped(2048):
        reading = _free_memory(seam.device, torch_free_too=True)

    assert isinstance(reading, tuple) and len(reading) == 2
    assert reading[0] == 2560 * MB
    assert reading[1] == 512 * MB


def test_default_device_is_clamped(seam: _DeviceSeam) -> None:
    """Comfy calls the reader with no device as well; that resolves to the torch device."""
    with comfy_patches.free_memory_view_clamped(2048):
        assert _free_memory() == 2048 * MB


def test_other_devices_pass_through(seam: _DeviceSeam) -> None:
    with comfy_patches.free_memory_view_clamped(2048):
        assert _free_memory(torch.device("cpu")) == seam.free_total


def test_cpu_torch_device_installs_no_clamp(
    init_horde: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import comfy.model_management as model_management

    original = model_management.get_free_memory
    monkeypatch.setattr(model_management, "get_torch_device", lambda: torch.device("cpu"))

    with comfy_patches.free_memory_view_clamped(2048):
        assert model_management.get_free_memory is original


def test_reading_is_restored_after_an_exception(seam: _DeviceSeam) -> None:
    with pytest.raises(RuntimeError), comfy_patches.free_memory_view_clamped(2048):
        assert _current_reader() is not seam.original
        raise RuntimeError("run failed")

    assert _current_reader() is seam.original, "a failed run must not leave comfy reading a clamped view"


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


def test_pipeline_run_scopes_the_clamp(init_horde: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """A run given a truth figure holds the clamp for its whole body and hands the reading back.

    The end-of-job evictor runs inside the clamped scope, which is where the reading in force during
    the run is observable without a model on a device.
    """
    import comfy.model_management as model_management

    before = model_management.get_free_memory
    device_is_cuda = torch.cuda.is_available() and model_management.get_torch_device().type == "cuda"

    observed: list[Any] = []
    monkeypatch.setattr(comfy_horde, "unload_all_models_vram", lambda: observed.append(_current_reader()))

    results = Comfy_Horde().run_pipeline(_mini_graph(), {}, device_free_truth_mb=1024)

    assert len(results) == 1
    assert model_management.get_free_memory is before, "the run must hand comfy's own reading back"
    assert observed and (observed[0] is not before) == device_is_cuda
