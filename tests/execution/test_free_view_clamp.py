"""Pins the clamp hordelib puts on ComfyUI's view of free VRAM during a pipeline run.

ComfyUI frees by shortfall against ``comfy.model_management.get_free_memory``, whose CUDA reading is
process-local: memory a sibling process holds is reported as free under WDDM, so the shortfall comes
out too small. A host that measures free VRAM at the device level passes that figure in as
``device_free_truth_mb`` and the scoped clamp lowers comfy's answer to it, less this process's own
allocator growth since the run started.

These are seam tests: the allocator and the device are stand-ins, so no GPU is required to observe
what comfy would read.
"""

import time
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
        assert _free_memory(seam.device) == 2304 * MB, "a release below the baseline must raise the ceiling"


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


def test_rebase_moves_the_ceiling_to_the_new_figure(seam: _DeviceSeam) -> None:
    """A grant-time figure replaces the entry figure, and growth before the rebase stops counting.

    The process's allocations before the rebase are already absent from the host's newer measurement, so
    charging them again would understate the card by exactly what this process holds.
    """
    seam.free_total = 16384 * MB

    with comfy_patches.free_memory_view_clamped(2048):
        seam.reserved += 512 * MB
        assert _free_memory(seam.device) == 1536 * MB

        assert comfy_patches.rebase_free_memory_view_clamp(6144) is True
        assert _free_memory(seam.device) == 6144 * MB, "the ceiling must follow the new figure"

        seam.reserved += 1024 * MB
        assert _free_memory(seam.device) == 5120 * MB, "growth must be measured from the rebase"


def test_release_below_the_rebase_baseline_raises_the_ceiling(seam: _DeviceSeam) -> None:
    """A model held at the rebase and unloaded afterwards is free on the card, so the ceiling gains it.

    The host's grant-time figure counted that model as used. Without the credit the clamp keeps reporting
    only the room beside it, and the next load seats partially on a card with room for all of it.
    """
    seam.free_total = 16384 * MB

    with comfy_patches.free_memory_view_clamped(2048):
        seam.reserved += 7680 * MB
        assert comfy_patches.rebase_free_memory_view_clamp(4608) is True
        assert _free_memory(seam.device) == 4608 * MB

        seam.reserved -= 7680 * MB
        assert _free_memory(seam.device) == 12288 * MB, "the released bytes must be added to the grant figure"

        seam.free_total = 10240 * MB
        assert _free_memory(seam.device) == 10240 * MB, "comfy's own total must still cap the credited ceiling"


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """A settable monotonic clock for the reservation checkpoints; the list's last value is the time now."""
    now = [100.0]
    monkeypatch.setattr(comfy_patches, "_monotonic", lambda: now[-1])
    return now


def _checkpoint(seam: _DeviceSeam, clock: list[float], *, at: float, reserved: int) -> None:
    """Move the allocator to ``reserved`` at time ``at`` and record the checkpoint there."""
    clock.append(at)
    seam.reserved = reserved
    assert comfy_patches.record_reservation_checkpoint() is True


def test_rebase_credits_a_release_the_figure_predates(seam: _DeviceSeam, clock: list[float]) -> None:
    """A figure read while the encoder was resident, consumed after its release, gains the release.

    The grant of a run job is written before the job's encode and consumed right after the encoder release,
    so the figure counts the encoder as used while the card has that room back.
    """
    seam.free_total = 16384 * MB
    seam.reserved = 7680 * MB
    with comfy_patches.free_memory_view_clamped(2048):
        _checkpoint(seam, clock, at=110.0, reserved=7680 * MB)  # after the encoder load
        _checkpoint(seam, clock, at=120.0, reserved=2560 * MB)  # after the encoder release

        assert comfy_patches.rebase_free_memory_view_clamp(2048, measured_at=115.0) is True
        assert _free_memory(seam.device) == (2048 + 5120) * MB, "the figure predates the release, so it is credited"


def test_rebase_takes_the_figure_as_measured_after_the_release(seam: _DeviceSeam, clock: list[float]) -> None:
    seam.free_total = 16384 * MB
    seam.reserved = 7680 * MB
    with comfy_patches.free_memory_view_clamped(2048):
        _checkpoint(seam, clock, at=110.0, reserved=7680 * MB)
        _checkpoint(seam, clock, at=120.0, reserved=2560 * MB)

        assert comfy_patches.rebase_free_memory_view_clamp(7168, measured_at=125.0) is True
        assert _free_memory(seam.device) == 7168 * MB, "a figure read after the release already counts it"


def test_rebase_charges_a_load_the_figure_predates(seam: _DeviceSeam, clock: list[float]) -> None:
    """A figure read before the staged upload still has that room free, so the upload comes off it."""
    seam.free_total = 16384 * MB
    seam.reserved = 2560 * MB
    with comfy_patches.free_memory_view_clamped(12288):
        _checkpoint(seam, clock, at=110.0, reserved=2560 * MB)  # after the encoder release
        _checkpoint(seam, clock, at=130.0, reserved=12800 * MB)  # after the staged upload

        assert comfy_patches.rebase_free_memory_view_clamp(12288, measured_at=120.0) is True
        assert _free_memory(seam.device) == (12288 - 10240) * MB


def test_rebase_near_a_checkpoint_takes_the_lower_reservation(seam: _DeviceSeam, clock: list[float]) -> None:
    """Within the clock's resolution of a checkpoint the figure may be from either side, so the safe side wins."""
    seam.free_total = 16384 * MB
    seam.reserved = 7680 * MB
    resolution = time.get_clock_info("monotonic").resolution
    with comfy_patches.free_memory_view_clamped(2048):
        _checkpoint(seam, clock, at=110.0, reserved=7680 * MB)
        _checkpoint(seam, clock, at=120.0, reserved=2560 * MB)

        assert comfy_patches.rebase_free_memory_view_clamp(2048, measured_at=120.0 - resolution / 2) is True
        assert _free_memory(seam.device) == 2048 * MB, "a release in progress at the measurement is not credited"

        seam.reserved = 2560 * MB
        _checkpoint(seam, clock, at=140.0, reserved=12800 * MB)
        assert comfy_patches.rebase_free_memory_view_clamp(12288, measured_at=140.0 + resolution / 2) is True
        assert _free_memory(seam.device) == (12288 - 10240) * MB, "a load in progress at the measurement is charged"


def test_rebase_older_than_every_checkpoint_takes_the_first(seam: _DeviceSeam, clock: list[float]) -> None:
    seam.free_total = 16384 * MB
    seam.reserved = 1024 * MB
    with comfy_patches.free_memory_view_clamped(4096):
        _checkpoint(seam, clock, at=110.0, reserved=6144 * MB)

        assert comfy_patches.rebase_free_memory_view_clamp(4096, measured_at=50.0) is True
        assert _free_memory(seam.device) == 0, "growth since the entry checkpoint is charged against the figure"


def test_rebase_without_a_measurement_time_counts_from_now(seam: _DeviceSeam, clock: list[float]) -> None:
    seam.free_total = 16384 * MB
    seam.reserved = 7680 * MB
    with comfy_patches.free_memory_view_clamped(2048):
        _checkpoint(seam, clock, at=110.0, reserved=7680 * MB)
        _checkpoint(seam, clock, at=120.0, reserved=2560 * MB)

        assert comfy_patches.rebase_free_memory_view_clamp(2048) is True
        assert _free_memory(seam.device) == 2048 * MB


def test_checkpoint_without_an_active_clamp_records_nothing(seam: _DeviceSeam) -> None:
    assert comfy_patches.record_reservation_checkpoint() is False


def test_rebase_without_an_active_clamp_is_a_no_op(seam: _DeviceSeam) -> None:
    assert comfy_patches.rebase_free_memory_view_clamp(6144) is False
    assert _current_reader() is seam.original

    with comfy_patches.free_memory_view_clamped(None):
        assert comfy_patches.rebase_free_memory_view_clamp(6144) is False, "no figure means no clamp to rebase"
        assert _free_memory(seam.device) == seam.free_total

    with comfy_patches.free_memory_view_clamped(2048):
        pass
    assert comfy_patches.rebase_free_memory_view_clamp(6144) is False, "a closed scope leaves nothing to rebase"


def test_rebase_keeps_the_clamp_when_the_allocator_is_unreadable(
    seam: _DeviceSeam,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with comfy_patches.free_memory_view_clamped(2048):
        monkeypatch.setattr(torch.cuda, "memory_reserved", _raise_runtime_error)
        assert comfy_patches.rebase_free_memory_view_clamp(6144) is False
        monkeypatch.setattr(torch.cuda, "memory_reserved", lambda dev=None: seam.reserved)
        assert _free_memory(seam.device) == 2048 * MB


def _raise_runtime_error(dev: Any = None) -> int:
    raise RuntimeError("allocator unreadable")


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
