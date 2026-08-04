"""Tests for the per-key log throttle and the ``send_sync`` receipt site that uses it.

ComfyUI drives ``Comfy_Horde.send_sync`` once per executed node and per progress tick, so its
receipt line dominated the child inference log at DEBUG. The throttle keeps one full-level line per
event label per interval and demotes the repeats, which these tests pin from both ends: the level
decision in isolation, and the levels the real ``send_sync`` actually emits.
"""

import threading
import types
import typing

import pytest
from loguru import logger

from hordelib.comfy_horde import Comfy_Horde
from hordelib.utils.logger import reset_log_throttle_state, throttled_log_level


@pytest.fixture(autouse=True)
def clear_throttle_schedule() -> typing.Generator[None, None, None]:
    """Give every case an empty schedule, since the throttle keeps process-global state."""
    reset_log_throttle_state()
    yield
    reset_log_throttle_state()


def test_the_first_call_on_a_key_uses_the_normal_level():
    assert throttled_log_level("site", 30.0, now=100.0) == "DEBUG"


def test_a_repeat_inside_the_interval_is_demoted():
    assert throttled_log_level("site", 30.0, now=100.0) == "DEBUG"

    assert throttled_log_level("site", 30.0, now=101.0) == "TRACE"
    assert throttled_log_level("site", 30.0, now=129.9) == "TRACE"


def test_the_normal_level_returns_once_the_interval_elapses():
    assert throttled_log_level("site", 30.0, now=100.0) == "DEBUG"
    assert throttled_log_level("site", 30.0, now=120.0) == "TRACE"

    assert throttled_log_level("site", 30.0, now=130.0) == "DEBUG"
    # The elapsed window restarts from the emission, not from the suppressed calls in between.
    assert throttled_log_level("site", 30.0, now=155.0) == "TRACE"
    assert throttled_log_level("site", 30.0, now=160.0) == "DEBUG"


def test_keys_keep_independent_schedules():
    assert throttled_log_level("first", 30.0, now=100.0) == "DEBUG"

    # A second key has its own schedule, so the first key's emission cannot mask it.
    assert throttled_log_level("second", 30.0, now=100.0) == "DEBUG"
    assert throttled_log_level("first", 30.0, now=101.0) == "TRACE"
    assert throttled_log_level("second", 30.0, now=101.0) == "TRACE"


def test_the_levels_are_caller_selectable():
    assert throttled_log_level("site", 30.0, normal_level="INFO", suppressed_level="DEBUG", now=100.0) == "INFO"
    assert throttled_log_level("site", 30.0, normal_level="INFO", suppressed_level="DEBUG", now=101.0) == "DEBUG"


def test_reset_clears_every_schedule():
    assert throttled_log_level("first", 30.0, now=100.0) == "DEBUG"
    assert throttled_log_level("second", 30.0, now=100.0) == "DEBUG"
    assert throttled_log_level("first", 30.0, now=101.0) == "TRACE"

    reset_log_throttle_state()

    assert throttled_log_level("first", 30.0, now=101.0) == "DEBUG"
    assert throttled_log_level("second", 30.0, now=101.0) == "DEBUG"


def test_concurrent_callers_on_one_key_emit_exactly_once():
    """ComfyUI can reach a throttled site from more than one thread, so the schedule needs a lock.

    Every thread reads the same injected clock, so a correct throttle hands the normal level to
    exactly one of them. A read-then-write race would let several through.
    """
    levels: list[str] = []
    levels_lock = threading.Lock()
    start = threading.Barrier(16)

    def call_site() -> None:
        start.wait()
        level = throttled_log_level("contended", 30.0, now=100.0)
        with levels_lock:
            levels.append(level)

    threads = [threading.Thread(target=call_site) for _ in range(16)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(levels) == 16
    assert levels.count("DEBUG") == 1
    assert levels.count("TRACE") == 15


def _send_sync_receiver() -> typing.Any:
    """Build the minimal stand-in ``Comfy_Horde.send_sync`` needs.

    ``send_sync`` reads only the outbound callback and the throttle interval off ``self``, so the
    real method runs against this without a ComfyUI runtime or a GPU.
    """
    return types.SimpleNamespace(
        _comfyui_callback=None,
        _CALLBACK_LOG_INTERVAL_SECONDS=Comfy_Horde._CALLBACK_LOG_INTERVAL_SECONDS,
    )


def _receipt_levels_for(events: list[tuple[str, dict]]) -> list[str]:
    """Return the level of each "ComfyUI callback" receipt emitted while replaying ``events``."""
    captured: list[str] = []
    sink_id = logger.add(
        lambda message: captured.append(message.record["level"].name),
        level="TRACE",
        filter=lambda record: record["message"] == "ComfyUI callback",
        catch=False,
    )
    try:
        receiver = _send_sync_receiver()
        for label, data in events:
            Comfy_Horde.send_sync(receiver, label, data, "client-1")
    finally:
        logger.remove(sink_id)
    return captured


def test_send_sync_demotes_repeat_receipts_for_one_label():
    # The 30s interval means every repeat within a test run falls inside the window.
    levels = _receipt_levels_for([("executing", {"node": "sampler"})] * 5)

    assert levels == ["DEBUG", "TRACE", "TRACE", "TRACE", "TRACE"]


def test_send_sync_surfaces_each_event_label_once():
    """A label that fires rarely must not be masked by whichever label floods the channel."""
    levels = _receipt_levels_for(
        [
            ("executing", {"node": "sampler"}),
            ("executing", {"node": "vae"}),
            ("progress", {"value": 1, "max": 20}),
            ("progress", {"value": 2, "max": 20}),
            ("execution_success", {"prompt_id": "a"}),
        ],
    )

    assert levels == ["DEBUG", "TRACE", "DEBUG", "TRACE", "DEBUG"]
