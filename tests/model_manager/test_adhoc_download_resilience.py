"""Unit tests for the ad-hoc downloader's disk-full and wait-timeout handling (no comfy, no GPU).

Both behaviours protect callers that block in ``wait_for_downloads`` / ``reset_adhoc_cache`` while a
worker thread holds a download record. A full disk that is retried, or a ``timeout=0`` that is treated
as "wait forever", turns a transient hiccup into an unbounded stall that the embedding worker grades as
a hung process.
"""

from __future__ import annotations

import builtins
import errno
import hashlib
import os
import threading
import time
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

import hordelib.model_manager.civitai_adhoc as civitai_adhoc
from hordelib.model_manager.civitai_adhoc import DownloadTarget, _QueuedDownload
from hordelib.model_manager.lora import LoraModelManager


class _FakeStreamResponse:
    """Minimal stand-in for a streamed ``requests`` response used by the download tests.

    Supports the subset the streaming download path exercises: use as a context manager,
    ``raise_for_status``, a ``url``, a ``headers`` mapping (for ``Content-Length``), and
    ``iter_content`` yielding the body in ``chunk_size`` slices.
    """

    def __init__(
        self,
        content: bytes = b"",
        url: str = "",
        raise_for_status: Any = None,
        headers: dict[str, str] | None = None,
        chunk_size: int | None = None,
    ) -> None:
        self._content = content
        self.url = url
        self._raise = raise_for_status
        self.headers = headers if headers is not None else {"Content-Length": str(len(content))}
        self._chunk_size = chunk_size

    def raise_for_status(self) -> None:
        if self._raise is not None:
            self._raise()

    def iter_content(self, chunk_size: int = 1) -> Any:
        step = self._chunk_size or chunk_size or 1
        for start in range(0, len(self._content), step):
            yield self._content[start : start + step]

    def __enter__(self) -> _FakeStreamResponse:
        return self

    def __exit__(self, *exc: Any) -> bool:
        return False


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
    manager._download_connections = 1  # keep these single-stream tests off the segmented fast path
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

    def _fake_get(url: str, timeout: float | None = None, stream: bool = False) -> Any:
        get_calls["n"] += 1
        return _FakeStreamResponse(content=b"x" * 1024, url=url, raise_for_status=lambda: None)

    monkeypatch.setattr(civitai_adhoc.requests, "get", _fake_get)

    real_open = builtins.open

    def _open_no_space(file: Any, *args: Any, **kwargs: Any) -> Any:
        # Only the weight-file write hits the full disk; everything else (logging, etc.) is unaffected.
        # The weight bytes are staged to a "big.safetensors.tmp-*" sibling before being renamed into
        # place, so the disk-full failure surfaces on that temp write.
        if ".safetensors" in str(file) and "wb" in args:
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

    def _fake_get(url: str, timeout: float | None = None, stream: bool = False) -> Any:
        get_calls["n"] += 1
        return _FakeStreamResponse(content=b"x" * 1024, url=url, raise_for_status=lambda: None)

    monkeypatch.setattr(civitai_adhoc.requests, "get", _fake_get)

    real_open = builtins.open

    def _open_eacces(file: Any, *args: Any, **kwargs: Any) -> Any:
        if ".safetensors" in str(file) and "wb" in args:
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

    def _fake_get(url: str, timeout: float | None = None, stream: bool = False) -> Any:
        get_calls["n"] += 1
        return _FakeStreamResponse(content=b"", url=url, raise_for_status=_raise_404)

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

    def _fake_get(url: str, timeout: float | None = None, stream: bool = False) -> Any:
        get_calls["n"] += 1
        return _FakeStreamResponse(content=b"", url=url, raise_for_status=lambda: None)

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

    def _fake_get(url: str, timeout: float | None = None, stream: bool = False) -> Any:
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

    def _fake_get(url: str, timeout: float | None = None, stream: bool = False) -> Any:
        get_calls["n"] += 1
        return _FakeStreamResponse(content=b"weights", url=url, raise_for_status=lambda: None)

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


def test_successful_download_lands_weight_and_sidecar_with_no_temp_residue(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A successful download leaves the final weight file, its sidecar, and no leftover temp file."""
    target = DownloadTarget(filename="ok.safetensors", url="http://example/ok", size_mb=1, sha256=None)
    manager = _bare_manager(tmp_path, target)

    payload = b"weight-bytes"

    def _fake_get(url: str, timeout: float | None = None, stream: bool = False) -> Any:
        return _FakeStreamResponse(content=payload, url=url, raise_for_status=lambda: None)

    monkeypatch.setattr(civitai_adhoc.requests, "get", _fake_get)

    record = SimpleNamespace(name="ok lora")
    manager._process_download(_QueuedDownload(record=record), civitai_adhoc.logger.bind(manager="lora"))

    assert (tmp_path / "ok.safetensors").read_bytes() == payload
    sidecar = (tmp_path / "ok.sha256").read_text()
    assert sidecar.split()[0] == hashlib.sha256(payload).hexdigest()
    assert sidecar.split()[1] == "*ok.safetensors"
    assert list(tmp_path.glob("*.tmp-*")) == [], "a successful download must not leave a temp file behind"
    manager._record_download_event.assert_called_once()  # pyrefly: ignore
    assert manager._record_download_event.call_args.kwargs["success"] is True  # pyrefly: ignore


def test_replace_failure_leaves_no_partial_final_file(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """When the rename into place fails, no partial final file and no leaked temp file remain."""
    target = DownloadTarget(filename="fail.safetensors", url="http://example/fail", size_mb=1, sha256=None)
    manager = _bare_manager(tmp_path, target)
    manager.RETRY_DELAY = 0.0

    def _fake_get(url: str, timeout: float | None = None, stream: bool = False) -> Any:
        return _FakeStreamResponse(content=b"weight-bytes", url=url, raise_for_status=lambda: None)

    monkeypatch.setattr(civitai_adhoc.requests, "get", _fake_get)

    def _boom_replace(src: Any, dst: Any) -> None:
        raise OSError(errno.EIO, "I/O error")

    monkeypatch.setattr(civitai_adhoc.os, "replace", _boom_replace)

    record = SimpleNamespace(name="fail lora")
    manager._process_download(_QueuedDownload(record=record), civitai_adhoc.logger.bind(manager="lora"))

    assert not (tmp_path / "fail.safetensors").exists(), "a failed replace must not leave a partial final file"
    assert list(tmp_path.glob("*.tmp-*")) == [], "a failed replace must not leak a temp file"
    manager._record_download_event.assert_called_once()  # pyrefly: ignore
    assert manager._record_download_event.call_args.kwargs["success"] is False  # pyrefly: ignore


def test_weight_is_exposed_only_by_the_atomic_replace(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """The final weight name appears only via os.replace; the bytes are staged to a temp sibling first.

    A reader listing the folder mid-write can therefore never observe a torn file at the final path:
    at the instant the rename runs the final path does not yet exist, and the source is a temp file.
    """
    target = DownloadTarget(filename="atomic.safetensors", url="http://example/atomic", size_mb=1, sha256=None)
    manager = _bare_manager(tmp_path, target)

    payload = b"weight-bytes"

    def _fake_get(url: str, timeout: float | None = None, stream: bool = False) -> Any:
        return _FakeStreamResponse(content=payload, url=url, raise_for_status=lambda: None)

    monkeypatch.setattr(civitai_adhoc.requests, "get", _fake_get)

    weight_path = str(tmp_path / "atomic.safetensors")
    real_replace = os.replace
    replace_calls: list[tuple[str, str]] = []

    def _spy_replace(src: Any, dst: Any) -> None:
        if str(dst) == weight_path:
            assert not os.path.exists(weight_path), "the final weight path must not exist before the atomic replace"
            assert ".tmp-" in str(src), "the weight bytes must be staged to a temp file before being renamed"
        replace_calls.append((str(src), str(dst)))
        real_replace(src, dst)

    monkeypatch.setattr(civitai_adhoc.os, "replace", _spy_replace)

    record = SimpleNamespace(name="atomic lora")
    manager._process_download(_QueuedDownload(record=record), civitai_adhoc.logger.bind(manager="lora"))

    weight_replaces = [call for call in replace_calls if call[1] == weight_path]
    assert len(weight_replaces) == 1, "the weight file must be exposed by exactly one atomic replace"
    assert (tmp_path / "atomic.safetensors").read_bytes() == payload


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

    def _fake_get(url: str, timeout: float | None = None, stream: bool = False) -> Any:
        get_calls["n"] += 1
        if get_calls["n"] == 1:
            raise civitai_adhoc.requests.exceptions.ReadTimeout("read timed out")
        return _FakeStreamResponse(content=b"weights", url=url, raise_for_status=lambda: None)

    monkeypatch.setattr(civitai_adhoc.requests, "get", _fake_get)

    record = SimpleNamespace(name="slow lora")
    manager._process_download(_QueuedDownload(record=record), civitai_adhoc.logger.bind(manager="lora"))

    assert get_calls["n"] == 2, "a transient ReadTimeout must be retried, then succeed"
    assert not manager._is_version_bad("2493924"), "a transient failure must not poison the negative cache"
    manager._record_download_event.assert_called_once()  # pyrefly: ignore
    assert manager._record_download_event.call_args.kwargs["success"] is True  # pyrefly: ignore


def test_streamed_download_uses_temp_replace_and_reports_monotonic_progress(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A weight is streamed chunk-by-chunk to a temp file, atomically renamed, with rising progress.

    The transfer must never buffer the whole file: each chunk is written and hashed as it arrives, the
    per-record ``progress_callback`` sees monotonically increasing ``downloaded_bytes`` up to the total,
    and the final weight name appears only via ``os.replace`` from a temp sibling.
    """
    target = DownloadTarget(filename="stream.safetensors", url="http://example/stream", size_mb=1, sha256=None)
    manager = _bare_manager(tmp_path, target)

    payload = b"ABCDEFGHIJ"  # ten bytes, streamed two at a time -> five chunks

    def _fake_get(url: str, timeout: float | None = None, stream: bool = False) -> Any:
        return _FakeStreamResponse(content=payload, url=url, raise_for_status=lambda: None, chunk_size=2)

    monkeypatch.setattr(civitai_adhoc.requests, "get", _fake_get)

    weight_path = str(tmp_path / "stream.safetensors")
    real_replace = os.replace
    weight_replaced_from: list[str] = []

    def _spy_replace(src: Any, dst: Any) -> None:
        if str(dst) == weight_path:
            assert ".tmp-" in str(src), "the weight bytes must be staged to a temp file before the rename"
            assert not os.path.exists(weight_path), "the final weight path must not exist before the atomic replace"
            weight_replaced_from.append(str(src))
        real_replace(src, dst)

    monkeypatch.setattr(civitai_adhoc.os, "replace", _spy_replace)

    progress: list[tuple[int, int]] = []
    record = SimpleNamespace(name="stream lora")
    queued = _QueuedDownload(record=record, progress_callback=lambda done, total: progress.append((done, total)))
    manager._process_download(queued, civitai_adhoc.logger.bind(manager="lora"))

    assert (tmp_path / "stream.safetensors").read_bytes() == payload
    assert len(weight_replaced_from) == 1, "the weight file must be exposed by exactly one atomic replace"
    assert list(tmp_path.glob("*.tmp-*")) == [], "a successful stream must not leave a temp file behind"

    downloaded_values = [done for done, _total in progress]
    assert downloaded_values == sorted(downloaded_values), "progress must be reported monotonically"
    assert all(total == len(payload) for _done, total in progress), "total must be the Content-Length"
    assert downloaded_values[-1] == len(payload), "the final progress must equal the full size"
    assert len(downloaded_values) > 1, "a chunked transfer must report progress more than once"
    assert queued.success is True
    assert queued.completion_event.is_set()


def test_progress_total_is_zero_when_content_length_absent(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """When the server reports no ``Content-Length`` the callback receives ``0`` as the total."""
    target = DownloadTarget(filename="nolen.safetensors", url="http://example/nolen", size_mb=1, sha256=None)
    manager = _bare_manager(tmp_path, target)

    def _fake_get(url: str, timeout: float | None = None, stream: bool = False) -> Any:
        return _FakeStreamResponse(content=b"XYZ", url=url, raise_for_status=lambda: None, headers={})

    monkeypatch.setattr(civitai_adhoc.requests, "get", _fake_get)

    totals: list[int] = []
    queued = _QueuedDownload(
        record=SimpleNamespace(name="nolen lora"),
        progress_callback=lambda _done, total: totals.append(total),
    )
    manager._process_download(queued, civitai_adhoc.logger.bind(manager="lora"))

    assert totals and all(total == 0 for total in totals), "an unknown Content-Length must surface as total 0"
    assert queued.success is True


def test_per_record_event_fires_while_a_sibling_is_still_in_flight(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One record's completion event fires as soon as it finishes, even while a slow sibling downloads.

    This is what lets ``fetch_adhoc_lora``/``fetch_adhoc_ti`` wait on their own record: the whole pool is
    still busy (``are_downloads_complete`` is False), yet the finished record's waiter is already released.
    """
    manager = _bare_manager(tmp_path, target=None)
    manager._existing_file_matches = Mock(return_value=True)  # pyrefly: ignore  # short-circuit to success

    release_sibling = threading.Event()

    def _prepare(record: Any) -> DownloadTarget:
        if record.name == "slow":
            release_sibling.wait(5)
        return DownloadTarget(filename=f"{record.name}.safetensors", url="http://x", size_mb=1, sha256=None)

    manager._prepare_download = _prepare  # pyrefly: ignore

    fast_q = _QueuedDownload(record=SimpleNamespace(name="fast"))
    slow_q = _QueuedDownload(record=SimpleNamespace(name="slow"))
    logger = civitai_adhoc.logger.bind(manager="lora")
    slow_thread = threading.Thread(target=manager._process_download, args=(slow_q, logger), daemon=True)
    fast_thread = threading.Thread(target=manager._process_download, args=(fast_q, logger), daemon=True)
    slow_thread.start()
    fast_thread.start()

    assert fast_q.completion_event.wait(3) is True, "the finished record's event must fire promptly"
    assert fast_q.success is True
    assert not slow_q.completion_event.is_set(), "the slow sibling must still be in flight"

    release_sibling.set()
    assert slow_q.completion_event.wait(5) is True
    slow_thread.join(5)
    fast_thread.join(5)


def test_per_record_wait_times_out_without_raising_and_without_the_sibling(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Waiting on a record's event honours the bounded timeout, returns False, and never raises.

    ``fetch_adhoc_*`` rely on this: a wait that elapses returns ``None`` rather than raising, and it does
    not block on any other in-flight download.
    """
    manager = _bare_manager(tmp_path, target=None)

    never_release = threading.Event()

    def _prepare(record: Any) -> DownloadTarget:
        never_release.wait()  # deliberately never released within the test
        return DownloadTarget(filename="stuck.safetensors", url="http://x", size_mb=1, sha256=None)

    manager._prepare_download = _prepare  # pyrefly: ignore

    stuck_q = _QueuedDownload(record=SimpleNamespace(name="stuck"))
    logger = civitai_adhoc.logger.bind(manager="lora")
    worker = threading.Thread(target=manager._process_download, args=(stuck_q, logger), daemon=True)
    worker.start()

    start = time.perf_counter()
    completed = stuck_q.completion_event.wait(0.3)
    elapsed = time.perf_counter() - start

    assert completed is False, "a bounded wait on an unfinished record must time out, not raise"
    assert 0.25 < elapsed < 3.0, "the wait must honour its own timeout, not block on the in-flight download"

    never_release.set()
    worker.join(5)


def test_completion_event_set_on_retries_exhausted(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """A download that exhausts its retry budget still signals its completion event as a failure."""
    target = DownloadTarget(filename="flaky.safetensors", url="http://example/flaky", size_mb=1, sha256=None)
    manager = _bare_manager(tmp_path, target)
    manager.RETRY_DELAY = 0.0

    def _fake_get(url: str, timeout: float | None = None, stream: bool = False) -> Any:
        raise civitai_adhoc.requests.ConnectionError("connection refused")

    monkeypatch.setattr(civitai_adhoc.requests, "get", _fake_get)

    queued = _QueuedDownload(record=SimpleNamespace(name="flaky lora"))
    manager._process_download(queued, civitai_adhoc.logger.bind(manager="lora"))

    assert queued.completion_event.is_set(), "an exhausted retry ladder must still release the waiter"
    assert queued.success is False


def test_completion_event_set_on_no_room(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """A download refused for lack of disk room signals its completion event as a failure, no network."""
    target = DownloadTarget(filename="huge.safetensors", url="http://example/huge", size_mb=999999, sha256=None)
    manager = _bare_manager(tmp_path, target)
    manager._ensure_room_for_download = Mock(return_value=False)  # pyrefly: ignore
    manager.disk_free_mb = Mock(return_value=10.0)  # pyrefly: ignore

    get_calls = {"n": 0}

    def _fake_get(url: str, timeout: float | None = None, stream: bool = False) -> Any:
        get_calls["n"] += 1
        return _FakeStreamResponse(content=b"x", url=url, raise_for_status=lambda: None)

    monkeypatch.setattr(civitai_adhoc.requests, "get", _fake_get)

    queued = _QueuedDownload(record=SimpleNamespace(name="huge lora"))
    manager._process_download(queued, civitai_adhoc.logger.bind(manager="lora"))

    assert get_calls["n"] == 0, "a no-room skip must not touch the network"
    assert queued.completion_event.is_set(), "a no-room skip must still release the waiter"
    assert queued.success is False


def test_cancellation_stops_stream_at_chunk_boundary_and_signals_event(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A generation bump mid-stream abandons the transfer at the next chunk, leaving no partial file.

    This frees the shared pool the instant a caller gives up, rather than transferring the whole weight
    to a file nobody is waiting for; the completion event is still signalled so the (former) waiter is
    released and the negative cache is not poisoned (a cancel is not a terminal failure).
    """
    target = DownloadTarget(
        filename="cancel.safetensors",
        url="http://example/cancel",
        size_mb=1,
        sha256=None,
        version_key="4242",
    )
    manager = _bare_manager(tmp_path, target)

    payload = b"ABCDEFGHIJ"  # streamed two bytes at a time

    def _fake_get(url: str, timeout: float | None = None, stream: bool = False) -> Any:
        return _FakeStreamResponse(content=payload, url=url, raise_for_status=lambda: None, chunk_size=2)

    monkeypatch.setattr(civitai_adhoc.requests, "get", _fake_get)

    queued = _QueuedDownload(record=SimpleNamespace(name="cancel lora"), generation=0)

    progress_calls: list[tuple[int, int]] = []

    def _bump_after_first(done: int, total: int) -> None:
        progress_calls.append((done, total))
        # Simulate cancel_active_downloads() landing after the first chunk is written.
        manager._download_generation = 1

    queued.progress_callback = _bump_after_first
    manager._process_download(queued, civitai_adhoc.logger.bind(manager="lora"))

    assert len(progress_calls) == 1, "the stream must stop at the chunk boundary after the cancel, not run on"
    assert not (tmp_path / "cancel.safetensors").exists(), "a cancelled stream must not leave a final weight file"
    assert list(tmp_path.glob("*.tmp-*")) == [], "a cancelled stream must not leak a temp file"
    assert queued.completion_event.is_set(), "a cancelled download must still release the waiter"
    assert queued.success is False
    assert not manager._is_version_bad("4242"), "a cancel is not a terminal failure and must not poison the cache"
