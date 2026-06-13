"""Unit tests for the native progress hook plumbing (hook invoked directly, no comfy)."""

import pytest

from hordelib.execution import progress_hook
from hordelib.metrics import MetricsCollector
from hordelib.utils.ioredirect import ComfyUIProgress, ComfyUIProgressUnit


@pytest.fixture(autouse=True)
def isolated_hook_state(monkeypatch: pytest.MonkeyPatch) -> MetricsCollector:
    """Give each test a fresh collector and clean per-run hook state."""
    collector = MetricsCollector()
    monkeypatch.setattr("hordelib.metrics._collector", collector)
    progress_hook.set_run_progress_callback(None)
    return collector


def test_forwards_native_progress_to_run_callback(isolated_hook_state: MetricsCollector) -> None:
    received: list[ComfyUIProgress] = []
    progress_hook.set_run_progress_callback(lambda progress, _message: received.append(progress))

    progress_hook._native_hook(1, 20)
    progress_hook._native_hook(2, 20)

    assert len(received) == 2
    final = received[-1]
    assert final.source == "native"
    assert final.current_step == 2
    assert final.total_steps == 20
    assert final.percent == 10
    assert final.rate_unit == ComfyUIProgressUnit.ITERATIONS_PER_SECOND
    # First sample has no rate (-1.0); the second is computable.
    assert received[0].rate == -1.0
    assert final.rate > 0.0 or final.rate == -1.0


def test_feeds_metrics_collector(isolated_hook_state: MetricsCollector) -> None:
    progress_hook._native_hook(1, 4)
    progress_hook._native_hook(4, 4)

    stats = isolated_hook_state.snapshot_and_reset_job().sampling
    assert stats is not None
    assert stats.steps_completed == 4
    assert stats.total_steps == 4


def test_no_callback_registered_is_silent(isolated_hook_state: MetricsCollector) -> None:
    progress_hook._native_hook(1, 10)  # must not raise


def test_callback_exceptions_are_swallowed(isolated_hook_state: MetricsCollector) -> None:
    def _broken(_progress: ComfyUIProgress, _message: str) -> None:
        raise RuntimeError("boom")

    progress_hook.set_run_progress_callback(_broken)
    progress_hook._native_hook(1, 10)  # must not raise


def test_clearing_callback_stops_forwarding(isolated_hook_state: MetricsCollector) -> None:
    received: list[ComfyUIProgress] = []
    progress_hook.set_run_progress_callback(lambda progress, _message: received.append(progress))
    progress_hook._native_hook(1, 10)
    progress_hook.set_run_progress_callback(None)
    progress_hook._native_hook(2, 10)

    assert len(received) == 1
