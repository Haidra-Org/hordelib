"""Unit tests for the ad-hoc downloader's disk-full and wait-timeout handling (no comfy, no GPU).

Both behaviours protect callers that block in ``wait_for_downloads`` / ``reset_adhoc_cache`` while a
worker thread holds a download record. A full disk that is retried, or a ``timeout=0`` that is treated
as "wait forever", turns a transient hiccup into an unbounded stall that the embedding worker grades as
a hung process.
"""

from __future__ import annotations

import builtins
import errno
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
    manager.total_retries_attempted = 0
    manager.stop_downloading = False
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
