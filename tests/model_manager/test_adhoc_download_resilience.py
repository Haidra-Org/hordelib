"""Unit tests for the ad-hoc downloader's disk-full and wait-timeout handling (no comfy, no GPU).

Both behaviours protect callers that block in ``wait_for_downloads`` / ``reset_adhoc_cache`` while a
worker thread holds a download record. A full disk that is retried, or a ``timeout=0`` that is treated
as "wait forever", turns a transient hiccup into an unbounded stall that the embedding worker grades as
a hung process.
"""

from __future__ import annotations

import builtins
import errno
import threading
import time
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

import hordelib.model_manager.civitai_adhoc as civitai_adhoc
from hordelib.model_manager.civitai_adhoc import DownloadTarget, _QueuedDownload
from hordelib.model_manager.lora import LoraModelManager


def _bare_manager(tmp_path: Any, target: DownloadTarget | None) -> LoraModelManager:
    """Build a LoraModelManager wired only for ``_process_download`` (no real init/network/cache)."""
    manager = object.__new__(LoraModelManager)
    manager.model_folder_path = str(tmp_path)
    manager._civitai_api_token = None
    manager.read_only = False
    manager.eviction_pins = set()
    manager.total_retries_attempted = 0
    manager.stop_downloading = False
    manager.min_free_disk_mb = 0
    manager._download_mutex = threading.Lock()
    manager._known_bad_versions = {}
    manager._download_generation = 0
    manager._prepare_download = Mock(return_value=target)  # pyrefly: ignore
    manager._existing_file_matches = Mock(return_value=False)  # pyrefly: ignore
    manager.is_model_url_from_civitai = Mock(return_value=False)  # pyrefly: ignore
    manager._commit_download = Mock()  # pyrefly: ignore
    manager.save_reference_to_disk = Mock()  # pyrefly: ignore
    manager._evict_adhoc_over_limit = Mock()  # pyrefly: ignore
    manager.is_default_cache_full = Mock(return_value=False)  # pyrefly: ignore
    manager._record_download_event = Mock()  # pyrefly: ignore
    manager._metric_download_duration = Mock()  # pyrefly: ignore
    manager._metric_network_errors = Mock()  # pyrefly: ignore
    manager._metric_retries = Mock()  # pyrefly: ignore
    return manager


def test_enospc_write_is_terminal_and_not_retried(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """A full-disk write must fail fast: one download attempt, recorded as a failure, no retries.

    The buggy path caught ENOSPC in the generic handler and retried ``MAX_RETRIES`` times, re-downloading
    the whole weight file each time while pinning the worker thread; the caller's drains blocked the
    entire budget. A full disk does not heal between attempts, so the download is terminal.
    """
    target = DownloadTarget(filename="big.safetensors", url="http://example/big", size_mb=200, sha256=None)
    manager = _bare_manager(tmp_path, target)

    get_calls = {"n": 0}

    def _fake_get(url: str, timeout: float | None = None) -> Any:
        get_calls["n"] += 1
        return SimpleNamespace(content=b"x" * 1024, url=url, raise_for_status=lambda: None)

    monkeypatch.setattr(civitai_adhoc.requests, "get", _fake_get)

    real_open = builtins.open

    def _open_no_space(file: Any, *args: Any, **kwargs: Any) -> Any:
        # Only the weight-file write hits the full disk; everything else (logging, etc.) is unaffected.
        if str(file).endswith("big.safetensors"):
            raise OSError(errno.ENOSPC, "No space left on device")
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", _open_no_space)

    record = SimpleNamespace(name="big lora")
    manager._process_download(_QueuedDownload(record=record), civitai_adhoc.logger.bind(manager="lora"))

    assert get_calls["n"] == 1, "ENOSPC must not be retried (each retry re-downloads the whole file)"
    manager._record_download_event.assert_called_once()  # pyrefly: ignore
    assert manager._record_download_event.call_args.kwargs["success"] is False  # pyrefly: ignore


def test_non_disk_oserror_still_retries(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """A non disk-full OS error keeps the original retry behaviour (it may be transient)."""
    target = DownloadTarget(filename="big.safetensors", url="http://example/big", size_mb=200, sha256=None)
    manager = _bare_manager(tmp_path, target)
    manager.RETRY_DELAY = 0.0  # don't sleep between retries in the test

    get_calls = {"n": 0}

    def _fake_get(url: str, timeout: float | None = None) -> Any:
        get_calls["n"] += 1
        return SimpleNamespace(content=b"x" * 1024, url=url, raise_for_status=lambda: None)

    monkeypatch.setattr(civitai_adhoc.requests, "get", _fake_get)

    real_open = builtins.open

    def _open_eacces(file: Any, *args: Any, **kwargs: Any) -> Any:
        if str(file).endswith("big.safetensors"):
            raise OSError(errno.EACCES, "Permission denied")
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", _open_eacces)

    record = SimpleNamespace(name="big lora")
    manager._process_download(_QueuedDownload(record=record), civitai_adhoc.logger.bind(manager="lora"))

    assert get_calls["n"] == manager.MAX_RETRIES + 1, "a non disk-full OS error should exhaust the retry budget"


def test_wait_for_downloads_zero_timeout_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    """``wait_for_downloads(0)`` must honour the 0s budget and raise, not block forever.

    The truthiness guard treated ``timeout=0`` as ``None`` (wait forever), so a caller asking for a
    bounded drain could hang on a wedged background download.
    """
    manager = object.__new__(LoraModelManager)
    manager.THREAD_WAIT_TIME = 0.01
    manager.are_downloads_complete = Mock(return_value=False)  # pyrefly: ignore

    start = time.perf_counter()
    with pytest.raises(TimeoutError):
        manager.wait_for_downloads(0)
    assert time.perf_counter() - start < 2.0, "a 0s budget must not block"


def test_wait_for_downloads_none_timeout_waits_then_returns(monkeypatch: pytest.MonkeyPatch) -> None:
    """``timeout=None`` keeps the wait-forever contract: it returns once downloads complete."""
    manager = object.__new__(LoraModelManager)
    manager.THREAD_WAIT_TIME = 0.01
    calls = {"n": 0}

    def _complete_after_a_few() -> bool:
        calls["n"] += 1
        return calls["n"] >= 3

    manager.are_downloads_complete = _complete_after_a_few  # pyrefly: ignore

    manager.wait_for_downloads(None)  # must not raise

    assert calls["n"] >= 3


def _http_error(status_code: int) -> civitai_adhoc.requests.HTTPError:
    """Build an ``HTTPError`` whose ``.response.status_code`` the handler inspects."""
    error = civitai_adhoc.requests.HTTPError(f"{status_code}")
    error.response = SimpleNamespace(status_code=status_code)  # type: ignore[assignment]
    return error


def test_download_404_is_terminal_after_one_confirm_retry(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """A 404 on the signed download URL must fail fast (one confirm retry) and remember the version.

    The buggy path treated a deterministic download-endpoint 404 like a transient error and burned the
    full ``MAX_RETRIES`` ladder (~33s of sleeping) while the job blocked in ``download_aux_models``,
    repeating that waste on every later job that requested the same dead model.
    """
    target = DownloadTarget(
        filename="dead.safetensors",
        url="http://example/dead",
        size_mb=1,
        sha256=None,
        version_key="299617",
    )
    manager = _bare_manager(tmp_path, target)
    manager.NOTFOUND_CONFIRM_DELAY = 0.0

    get_calls = {"n": 0}

    def _raise_404() -> None:
        raise _http_error(404)

    def _fake_get(url: str, timeout: float | None = None) -> Any:
        get_calls["n"] += 1
        return SimpleNamespace(content=b"", url=url, raise_for_status=_raise_404)

    monkeypatch.setattr(civitai_adhoc.requests, "get", _fake_get)

    record = SimpleNamespace(name="dead lora")
    manager._process_download(_QueuedDownload(record=record), civitai_adhoc.logger.bind(manager="lora"))

    assert get_calls["n"] == 2, "a download 404 should get exactly one confirm retry, not the full ladder"
    manager._record_download_event.assert_called_once()  # pyrefly: ignore
    assert manager._record_download_event.call_args.kwargs["success"] is False  # pyrefly: ignore
    assert manager._is_version_bad("299617"), "the dead version must be remembered for later jobs"


def test_known_bad_version_is_skipped_without_network(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """A version already in the negative cache must be skipped before any network call."""
    target = DownloadTarget(
        filename="dead.safetensors",
        url="http://example/dead",
        size_mb=1,
        sha256=None,
        version_key="299617",
    )
    manager = _bare_manager(tmp_path, target)
    manager._mark_version_bad("299617")

    get_calls = {"n": 0}

    def _fake_get(url: str, timeout: float | None = None) -> Any:
        get_calls["n"] += 1
        return SimpleNamespace(content=b"", url=url, raise_for_status=lambda: None)

    monkeypatch.setattr(civitai_adhoc.requests, "get", _fake_get)

    record = SimpleNamespace(name="dead lora")
    manager._process_download(_QueuedDownload(record=record), civitai_adhoc.logger.bind(manager="lora"))

    assert get_calls["n"] == 0, "a known-bad version must be skipped before any network call"
    manager._record_download_event.assert_called_once()  # pyrefly: ignore
    assert manager._record_download_event.call_args.kwargs["success"] is False  # pyrefly: ignore


def test_negative_cache_expires_and_prunes(monkeypatch: pytest.MonkeyPatch) -> None:
    """A negative-cache entry must expire after its TTL (so a restored model self-heals) and be pruned."""
    manager = object.__new__(LoraModelManager)
    manager._download_mutex = threading.Lock()
    manager._known_bad_versions = {}
    manager.NEGATIVE_CACHE_TTL = 10

    clock = {"t": 1000.0}
    monkeypatch.setattr(civitai_adhoc.time, "monotonic", lambda: clock["t"])

    manager._mark_version_bad("299617")
    assert manager._is_version_bad("299617") is True

    clock["t"] += 11  # advance past the TTL
    assert manager._is_version_bad("299617") is False
    assert "299617" not in manager._known_bad_versions, "an expired entry must be pruned on access"


def test_cancel_mid_retry_ladder_stops_further_attempts(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """A download whose job is cancelled mid-retry must abandon the ladder, not pin the thread.

    This is what frees the shared pool when a caller gives up on a stalled aux download: without it the
    ReadTimeout ladder burns its full ``MAX_RETRIES`` budget while the next job waits behind it.
    """
    target = DownloadTarget(
        filename="slow.safetensors",
        url="http://example/slow",
        size_mb=1,
        sha256=None,
        version_key="555",
    )
    manager = _bare_manager(tmp_path, target)
    manager.RETRY_DELAY = 0.0

    get_calls = {"n": 0}

    def _fake_get(url: str, timeout: float | None = None) -> Any:
        get_calls["n"] += 1
        # Simulate a concurrent cancel_active_downloads() landing after the second attempt.
        if get_calls["n"] == 2:
            manager._download_generation += 1
        raise civitai_adhoc.requests.exceptions.ReadTimeout("read timed out")

    monkeypatch.setattr(civitai_adhoc.requests, "get", _fake_get)

    record = SimpleNamespace(name="slow lora")
    manager._process_download(_QueuedDownload(record=record, generation=0), civitai_adhoc.logger.bind(manager="lora"))

    assert get_calls["n"] == 2, "a cancelled download must stop retrying instead of exhausting the ladder"
    assert manager._record_download_event.call_args.kwargs["success"] is False  # pyrefly: ignore
    assert not manager._is_version_bad("555"), "a cancel is not a terminal failure and must not poison the cache"


def test_stale_generation_download_skips_network(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """A queued download already behind the current generation is abandoned before any network call."""
    target = DownloadTarget(filename="x.safetensors", url="http://example/x", size_mb=1, sha256=None, version_key="9")
    manager = _bare_manager(tmp_path, target)
    manager._download_generation = 3  # a cancel happened before this record was picked up

    get_calls = {"n": 0}

    def _fake_get(url: str, timeout: float | None = None) -> Any:
        get_calls["n"] += 1
        return SimpleNamespace(content=b"weights", url=url, raise_for_status=lambda: None)

    monkeypatch.setattr(civitai_adhoc.requests, "get", _fake_get)

    record = SimpleNamespace(name="stale lora")
    manager._process_download(_QueuedDownload(record=record, generation=0), civitai_adhoc.logger.bind(manager="lora"))

    assert get_calls["n"] == 0, "a stale-generation download must not touch the network"


def test_cancel_active_downloads_bumps_generation_and_clears_queue() -> None:
    """cancel_active_downloads bumps the generation and drops queued work without killing the pool."""
    from collections import deque

    manager = object.__new__(LoraModelManager)
    manager._download_mutex = threading.Lock()
    manager._download_generation = 0
    manager._stop_all_threads = False
    manager._download_queue = deque(
        [_QueuedDownload(record=SimpleNamespace(name="a"), generation=0)],
    )
    manager._metric_queue_size = Mock()  # pyrefly: ignore
    manager.METRIC_PREFIX = "lora"

    manager.cancel_active_downloads()

    assert manager._download_generation == 1
    assert len(manager._download_queue) == 0
    assert manager._stop_all_threads is False, "cancel must not permanently shut the pool down"


def test_read_timeout_still_retries_and_is_not_cached(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """A transient ReadTimeout must keep its full retry budget and must not poison the negative cache."""
    target = DownloadTarget(
        filename="slow.safetensors",
        url="http://example/slow",
        size_mb=1,
        sha256=None,
        version_key="2493924",
    )
    manager = _bare_manager(tmp_path, target)
    manager.RETRY_DELAY = 0.0

    get_calls = {"n": 0}

    def _fake_get(url: str, timeout: float | None = None) -> Any:
        get_calls["n"] += 1
        if get_calls["n"] == 1:
            raise civitai_adhoc.requests.exceptions.ReadTimeout("read timed out")
        return SimpleNamespace(content=b"weights", url=url, raise_for_status=lambda: None)

    monkeypatch.setattr(civitai_adhoc.requests, "get", _fake_get)

    record = SimpleNamespace(name="slow lora")
    manager._process_download(_QueuedDownload(record=record), civitai_adhoc.logger.bind(manager="lora"))

    assert get_calls["n"] == 2, "a transient ReadTimeout must be retried, then succeed"
    assert not manager._is_version_bad("2493924"), "a transient failure must not poison the negative cache"
    manager._record_download_event.assert_called_once()  # pyrefly: ignore
    assert manager._record_download_event.call_args.kwargs["success"] is True  # pyrefly: ignore
