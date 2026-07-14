"""Unit tests for the ad-hoc downloader's multi-connection segmented fast path (no comfy, no GPU).

A large ad-hoc LoRA is fetched over several concurrent ranged connections by reusing the checkpoint
engine's ``_segmented_download``, while the existing single-stream transfer stays intact as the fallback
for small files, range-refusing servers, and any environment where the engine helper is unavailable.

These tests exercise the civitai_adhoc boundary: they monkeypatch ``civitai_adhoc._segmented_download``
(the engine has its own tests) and assert which transfer path ran, commit vs no-commit, the file left on
disk, and the terminal outcome.
"""

from __future__ import annotations

import hashlib
import threading
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

import hordelib.model_manager.civitai_adhoc as civitai_adhoc
from hordelib.model_manager.civitai_adhoc import DownloadTarget, _QueuedDownload
from hordelib.model_manager.lora import LoraModelManager


class _FakeStreamResponse:
    """Minimal stand-in for a streamed ``requests`` response used by the single-stream fallback tests."""

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
    """Build a LoraModelManager wired only for ``_process_download`` (no real init/network/cache).

    Mirrors the resilience-suite harness. The segmented path additionally needs ``_download_connections``,
    which the real ``__init__`` sets but ``object.__new__`` does not, so every test sets it explicitly.
    """
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


def _outcome(destination: Any, *, success: bool, payload: bytes) -> Any:
    """Land a file (and its sidecar) at *destination* like the engine does, returning its ``DownloadOutcome``."""
    destination.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    destination.with_suffix(".sha256").write_text(f"{digest} *{destination.name}")
    return civitai_adhoc.DownloadOutcome(
        success=success,
        final_path=destination,
        bytes_written=len(payload),
        sha256=digest,
    )


def test_segmented_success_commits_without_single_stream(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """A successful segmented transfer commits the download and never touches the single-stream path."""
    target = DownloadTarget(filename="big.safetensors", url="http://example/big", size_mb=200, sha256=None)
    manager = _bare_manager(tmp_path, target)
    manager._download_connections = 4

    payload = b"segmented-weight-bytes"
    destination = tmp_path / "big.safetensors"

    def _fake_segmented(url: str, dest: Any, **kwargs: Any) -> Any:
        return _outcome(dest, success=True, payload=payload)

    monkeypatch.setattr(civitai_adhoc, "_segmented_download", _fake_segmented)

    def _no_single_stream(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the single-stream path must not run when segmentation succeeds")

    monkeypatch.setattr(civitai_adhoc.requests, "get", _no_single_stream)

    queued = _QueuedDownload(record=SimpleNamespace(name="big lora"))
    manager._process_download(queued, civitai_adhoc.logger.bind(manager="lora"))

    assert queued.success is True
    manager._commit_download.assert_called_once()  # pyrefly: ignore
    assert manager._commit_download.call_args.kwargs["downloaded"] is True  # pyrefly: ignore
    assert destination.read_bytes() == payload
    assert manager._record_download_event.call_args.kwargs["success"] is True  # pyrefly: ignore


def test_segmented_none_falls_back_to_single_stream(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """When the engine reports the file unsuitable (returns None), the single-stream path completes it."""
    target = DownloadTarget(filename="small.safetensors", url="http://example/small", size_mb=1, sha256=None)
    manager = _bare_manager(tmp_path, target)
    manager._download_connections = 4

    def _fake_segmented(url: str, dest: Any, **kwargs: Any) -> Any:
        return None

    monkeypatch.setattr(civitai_adhoc, "_segmented_download", _fake_segmented)

    payload = b"single-stream-body"

    def _fake_get(url: str, timeout: float | None = None, stream: bool = False) -> Any:
        return _FakeStreamResponse(content=payload, url=url, raise_for_status=lambda: None)

    monkeypatch.setattr(civitai_adhoc.requests, "get", _fake_get)

    queued = _QueuedDownload(record=SimpleNamespace(name="small lora"))
    manager._process_download(queued, civitai_adhoc.logger.bind(manager="lora"))

    assert queued.success is True
    assert (tmp_path / "small.safetensors").read_bytes() == payload, "the single-stream body must have landed"
    manager._commit_download.assert_called_once()  # pyrefly: ignore
    assert manager._commit_download.call_args.kwargs["downloaded"] is True  # pyrefly: ignore


def test_connections_one_skips_segmentation(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """With a single configured connection the engine helper is never invoked; single-stream serves."""
    target = DownloadTarget(filename="one.safetensors", url="http://example/one", size_mb=200, sha256=None)
    manager = _bare_manager(tmp_path, target)
    manager._download_connections = 1

    def _must_not_segment(url: str, dest: Any, **kwargs: Any) -> Any:
        raise AssertionError("segmentation must be skipped when only one connection is configured")

    monkeypatch.setattr(civitai_adhoc, "_segmented_download", _must_not_segment)

    payload = b"single-connection-body"

    def _fake_get(url: str, timeout: float | None = None, stream: bool = False) -> Any:
        return _FakeStreamResponse(content=payload, url=url, raise_for_status=lambda: None)

    monkeypatch.setattr(civitai_adhoc.requests, "get", _fake_get)

    queued = _QueuedDownload(record=SimpleNamespace(name="one lora"))
    manager._process_download(queued, civitai_adhoc.logger.bind(manager="lora"))

    assert queued.success is True
    assert (tmp_path / "one.safetensors").read_bytes() == payload


def test_segmented_cancellation_returns_false(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """A cancel raised through the progress hook aborts the segmented transfer without committing."""
    target = DownloadTarget(
        filename="cancel.safetensors",
        url="http://example/cancel",
        size_mb=200,
        sha256=None,
        version_key="4242",
    )
    manager = _bare_manager(tmp_path, target)
    manager._download_connections = 4

    def _fake_segmented(url: str, dest: Any, **kwargs: Any) -> Any:
        raise civitai_adhoc._DownloadCancelled

    monkeypatch.setattr(civitai_adhoc, "_segmented_download", _fake_segmented)

    queued = _QueuedDownload(record=SimpleNamespace(name="cancel lora"), generation=0)
    manager._process_download(queued, civitai_adhoc.logger.bind(manager="lora"))

    assert queued.success is False
    manager._commit_download.assert_not_called()  # pyrefly: ignore
    assert manager._record_download_event.call_args.kwargs["success"] is False  # pyrefly: ignore
    assert not manager._is_version_bad("4242"), "a cancel is not a terminal failure and must not poison the cache"


def test_segmented_hash_mismatch_discards_and_does_not_commit(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A segmented outcome whose hash never matches exhausts retries, commits nothing, leaves no weight file."""
    target = DownloadTarget(
        filename="bad.safetensors",
        url="http://example/bad",
        size_mb=200,
        sha256="deadbeef",
    )
    manager = _bare_manager(tmp_path, target)
    manager._download_connections = 4
    manager.RETRY_DELAY = 0.0

    destination = tmp_path / "bad.safetensors"
    call_count = {"n": 0}

    def _fake_segmented(url: str, dest: Any, **kwargs: Any) -> Any:
        call_count["n"] += 1
        return _outcome(dest, success=False, payload=b"corrupt-bytes")

    monkeypatch.setattr(civitai_adhoc, "_segmented_download", _fake_segmented)

    queued = _QueuedDownload(record=SimpleNamespace(name="bad lora"))
    manager._process_download(queued, civitai_adhoc.logger.bind(manager="lora"))

    assert queued.success is False
    assert call_count["n"] == manager.MAX_RETRIES + 1, "a mismatch must retry the full ladder, then fail"
    manager._commit_download.assert_not_called()  # pyrefly: ignore
    assert not destination.exists(), "a discarded mismatched download must leave no weight file behind"
    assert not destination.with_suffix(".sha256").exists(), "the sidecar of a discarded download must be removed"
    assert manager._record_download_event.call_args.kwargs["success"] is False  # pyrefly: ignore


def test_adhoc_connections_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """The env override parses to an int, floors at 1, and falls back to the default when unset or invalid."""
    monkeypatch.delenv("AIWORKER_LORA_DOWNLOAD_CONNECTIONS", raising=False)
    assert civitai_adhoc._adhoc_connections_from_env() == civitai_adhoc.DEFAULT_ADHOC_DOWNLOAD_CONNECTIONS

    monkeypatch.setenv("AIWORKER_LORA_DOWNLOAD_CONNECTIONS", "6")
    assert civitai_adhoc._adhoc_connections_from_env() == 6

    monkeypatch.setenv("AIWORKER_LORA_DOWNLOAD_CONNECTIONS", "0")
    assert civitai_adhoc._adhoc_connections_from_env() == 1, "the connection count must floor at 1"

    monkeypatch.setenv("AIWORKER_LORA_DOWNLOAD_CONNECTIONS", "not-an-int")
    assert civitai_adhoc._adhoc_connections_from_env() == civitai_adhoc.DEFAULT_ADHOC_DOWNLOAD_CONNECTIONS
