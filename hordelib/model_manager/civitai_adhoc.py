"""Shared engine for the CivitAI-sourced ad-hoc model managers (LoRA and Textual Inversion).

Both the LoRA and Textual Inversion managers discover model metadata from CivitAI, download weight
files on demand into a size-bounded local cache, verify checksums, and evict least-recently-used
entries. That machinery — the retrying metadata fetch, the threaded download workers, the queue and
thread pool, the fuzzy name index, the cache-size accounting and eviction, and the on-disk
persistence — lives here once. :class:`CivitaiAdhocModelManager` is generic over the typed record it
manages and exposes a handful of hooks for the two genuine points of divergence:

* how a CivitAI API item is parsed into a record (LoRA tracks multiple versions; TI is single-version);
* how a record's weight file is located for download (TI resolves its URL via the Hordeling API); and
* how the per-file cache entries are enumerated for accounting and eviction.

Subclasses provide those hooks plus a small amount of category-specific configuration; everything else
is inherited. See :class:`~hordelib.model_manager.lora.LoraModelManager` and
:class:`~hordelib.model_manager.ti.TextualInversionModelManager`.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import re
import threading
import time
import uuid
from abc import abstractmethod
from collections import deque
from collections.abc import Iterable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from datetime import datetime
from multiprocessing.synchronize import Lock as multiprocessing_lock
from pathlib import Path
from typing import Any, override

import logfire
import requests
from fuzzywuzzy import fuzz
from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORY
from horde_model_reference.model_reference_records import GenericModelRecord
from loguru import logger

from hordelib.consts import MODEL_CATEGORY_NAMES
from hordelib.metrics import DownloadCategory, DownloadEvent, get_metrics_collector
from hordelib.model_manager.base import BaseModelManager

TESTS_ONGOING = os.getenv("TESTS_ONGOING", "0") == "1"

TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"
"""The strftime/strptime format used for all cache timestamps persisted to disk."""

A1111_TRIGGER_PATTERN = re.compile(r"<(?:lora|ti):(.*):.*>")
"""Matches A1111-style inline trigger syntax so it can be reduced to the bare trigger word."""


def now_timestamp() -> str:
    """Return the current local time formatted with :data:`TIMESTAMP_FORMAT`."""
    return datetime.now().strftime(TIMESTAMP_FORMAT)


def timestamp_to_datetime(value: str | None) -> datetime | None:
    """Parse a :data:`TIMESTAMP_FORMAT` string into a ``datetime``, or ``None`` if absent/invalid."""
    if not value:
        return None
    try:
        return datetime.strptime(value, TIMESTAMP_FORMAT)
    except ValueError:
        return None


def normalise_a1111_triggers(triggers: list[str]) -> list[str]:
    """Return *triggers* with any A1111-style ``<lora:name:weight>`` entries reduced to ``name``."""
    return [
        A1111_TRIGGER_PATTERN.sub(r"\1", trigger) if A1111_TRIGGER_PATTERN.match(trigger) else trigger
        for trigger in triggers
    ]


@dataclass(frozen=True)
class CacheEntry:
    """Represents one downloaded weight file for cache accounting and eviction.

    A LoRA contributes one entry per known version; a textual inversion contributes a single entry.
    """

    model_key: str
    """The reference key of the owning model."""
    version_key: str | None
    """The version id for LoRA entries, or ``None`` for single-version models."""
    filename: str
    """The on-disk weight filename, relative to the manager's model folder."""
    size_mb: float
    """The entry's size in megabytes (normalised across managers for comparable accounting)."""
    adhoc: bool
    """Whether the entry was fetched ad-hoc and is therefore subject to eviction."""
    last_used: datetime | None
    """When the entry was last used, used to choose eviction victims (``None`` sorts as oldest)."""


@dataclass
class DownloadTarget:
    """Represents the concrete weight file a download worker should fetch for a record."""

    filename: str
    """The on-disk filename to write, relative to the manager's model folder."""
    url: str
    """The fully qualified download URL (a CivitAI API token is appended later if applicable)."""
    size_mb: float
    """The expected size in megabytes, for progress/metrics."""
    sha256: str | None = None
    """The expected SHA256 checksum, or ``None`` to accept whatever is downloaded."""
    version_key: str | None = None
    """The version id this target corresponds to, for multi-version (LoRA) records."""


class SkipDownload(Exception):
    """Raised by :meth:`CivitaiAdhocModelManager._prepare_download` to abandon a queued download.

    Unlike a network error (which is retried), this signals a definitive "give up on this one"
    decision, e.g. the remote reports the file is permanently unavailable.
    """


@dataclass
class _QueuedDownload[R]:
    """An item on the download queue: the record to download plus contextual logging metadata."""

    record: R
    context: dict = field(default_factory=dict)


class CivitaiAdhocModelManager[RecordT: GenericModelRecord](
    BaseModelManager[RecordT],
):
    """Shared base for managers that download and cache CivitAI models on demand.

    Subclass Integration:
        Concrete managers set the class attributes (:attr:`MODEL_CATEGORY`, :attr:`RECORD_TYPE`,
        :attr:`METRIC_PREFIX`, size limits, fuzzy threshold) and implement the abstract hooks
        (:meth:`_parse_civitai_item`, :meth:`_iter_cache_entries`, :meth:`_prepare_download`,
        :meth:`_commit_download`, :meth:`_rebuild_indices`, :meth:`_delete_model_entry`). Everything
        else — metadata fetching, the thread pool, queueing, cache accounting/eviction, persistence —
        is inherited.

    Thread Safety:
        Downloads run on a pool of daemon threads. ``_mutex`` guards mutations of the in-memory
        reference and indices; ``_file_mutex``/``_file_lock`` guard on-disk reads and writes (the
        latter may be a multiprocessing lock shared across worker processes).
    """

    MODEL_CATEGORY: MODEL_REFERENCE_CATEGORY
    """The ``horde_model_reference`` category this manager serves."""
    METRIC_PREFIX: str
    """The logfire metric/log namespace for this manager (e.g. ``"lora"``)."""

    MAX_RETRIES: int = 10 if not TESTS_ONGOING else 3
    """The maximum number of attempts for a metadata fetch or file download."""
    MAX_DOWNLOAD_THREADS: int = 5 if not TESTS_ONGOING else 75
    """The size of the download worker thread pool."""
    RETRY_DELAY: float = 3 if not TESTS_ONGOING else 0.2
    """The delay between retries, in seconds."""
    REQUEST_METADATA_TIMEOUT: float = 35
    """The no-data timeout for a metadata request, in seconds."""
    REQUEST_DOWNLOAD_TIMEOUT: float = 10 if not TESTS_ONGOING else 1
    """The no-data timeout for a download request, in seconds."""
    THREAD_WAIT_TIME: float = 0.1
    """The poll interval while waiting on the download queue/threads, in seconds."""
    FUZZ_THRESHOLD: int = 90
    """The minimum fuzzy-match ratio for a name to be accepted as a reference key."""

    def __init__(
        self,
        *,
        model_category_name: MODEL_CATEGORY_NAMES,
        models_db_path: str | Path,
        civitai_api_token: str | None = None,
        download_reference: bool = False,
        multiprocessing_lock: multiprocessing_lock | None = None,
        reference_backups: bool | None = None,
        max_top_disk: int,
        max_adhoc_disk: int,
    ) -> None:
        """Create an ad-hoc CivitAI model manager.

        Args:
            model_category_name: The hordelib category enum for this manager.
            models_db_path: Path to the JSON file persisting this manager's reference cache.
            civitai_api_token: Optional CivitAI API token, appended to CivitAI download URLs.
            download_reference: Unused for ad-hoc managers (the reference is built from downloads);
                accepted for signature parity with :class:`BaseModelManager`.
            multiprocessing_lock: Optional cross-process lock guarding on-disk reference writes.
            reference_backups: Whether to write timestamped backups on each save. When ``None``,
                backups follow whether a multiprocessing lock was supplied.
            max_top_disk: The default-set cache budget, in megabytes.
            max_adhoc_disk: The ad-hoc cache budget, in megabytes.
        """
        self._max_top_disk = max_top_disk
        self.max_adhoc_disk = max_adhoc_disk

        self._data: dict | None = None
        self._next_page_url: str | None = None
        self._mutex = threading.Lock()
        self._file_mutex = threading.Lock()
        self._download_mutex = threading.Lock()

        self._using_multiprocessing = (
            reference_backups if reference_backups is not None else multiprocessing_lock is not None
        )
        self._file_lock: AbstractContextManager[Any] = multiprocessing_lock or nullcontext()

        self._file_count = 0
        self._download_threads: dict[int, dict] = {}
        self._download_queue: deque[_QueuedDownload[RecordT]] = deque()
        self._thread: threading.Thread | None = None
        self.stop_downloading = True
        self.nsfw = True
        self._adhoc_reset_thread: threading.Thread | None = None
        self._stop_all_threads = False
        self._index_ids: dict[int, str] = {}
        self._index_orig_names: dict[str, str] = {}
        self.total_retries_attempted = 0

        self._init_metrics()

        super().__init__(
            model_category_name=model_category_name,
            download_reference=download_reference,
            models_db_path=models_db_path,
            civitai_api_token=civitai_api_token,
        )

    def _init_metrics(self) -> None:
        """Create the logfire metrics for this manager under its :attr:`METRIC_PREFIX` namespace."""
        prefix = self.METRIC_PREFIX
        self._metric_download_duration = logfire.metric_histogram(f"{prefix}.download.duration_s", unit="s")
        self._metric_metadata_duration = logfire.metric_histogram(f"{prefix}.metadata_fetch.duration_ms", unit="ms")
        self._metric_retries = logfire.metric_counter(f"{prefix}.download.retries", unit="1")
        self._metric_network_errors = logfire.metric_counter(f"{prefix}.network.errors", unit="1")
        self._metric_queue_size = logfire.metric_gauge(f"{prefix}.queue.size", unit="1")
        self._metric_active_threads = logfire.metric_gauge(f"{prefix}.threads.active", unit="1")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @override
    def load_model_database(self) -> None:
        """Load the cached reference from disk, dropping entries whose weight file is missing."""
        manager_logger = logger.bind(manager=self.METRIC_PREFIX)
        os.makedirs(self.model_folder_path, exist_ok=True)
        os.makedirs(self.models_db_path.parent, exist_ok=True)

        if not self.models_db_path.exists():
            manager_logger.info("adhoc.reference_missing", database_name=self.models_db_name)
            self.model_reference = {}
            self.save_reference_to_disk()
            return

        with self._file_mutex, self._file_lock:
            try:
                raw = json.loads(self.models_db_path.read_text())
            except json.JSONDecodeError:
                raw = self._load_reference_backup()

        self.model_reference = self._deserialise_reference(raw)
        self._prune_missing_files()
        self._rebuild_indices()
        self.available_models = list(self.model_reference.keys())
        self.save_reference_to_disk()
        manager_logger.info("adhoc.reference_loaded", model_count=len(self.model_reference))

    def _load_reference_backup(self) -> dict:
        """Return the newest readable reference backup as raw JSON, or an empty mapping."""
        for backup in self.get_all_backup_files():
            try:
                return json.loads(backup.read_text())
            except (json.JSONDecodeError, FileNotFoundError):
                logger.bind(manager=self.METRIC_PREFIX).warning("adhoc.backup_unreadable", backup_path=str(backup))
        logger.bind(manager=self.METRIC_PREFIX).error("adhoc.reference_load_failed", database_name=self.models_db_name)
        return {}

    def _deserialise_reference(self, raw: dict) -> dict[str, RecordT]:
        """Validate raw JSON into typed records, skipping entries that fail validation."""
        reference: dict[str, RecordT] = {}
        record_cls: type[RecordT] = self.RECORD_TYPE
        for model_key, record_data in raw.items():
            try:
                reference[model_key] = record_cls.model_validate(record_data)
            except Exception as parse_error:  # pydantic.ValidationError and friends
                logger.bind(manager=self.METRIC_PREFIX).warning(
                    "adhoc.record_invalid",
                    model_key=model_key,
                    error=str(parse_error),
                )
        return reference

    def _prune_missing_files(self) -> None:
        """Drop cache entries whose weight file is no longer present on disk."""
        for entry in list(self._iter_cache_entries()):
            if not os.path.exists(os.path.join(self.model_folder_path, entry.filename)):
                self._drop_entry(entry.model_key, entry.version_key)

    def save_reference_to_disk(self) -> None:
        """Persist the in-memory reference to disk, optionally writing a timestamped backup first."""
        serialised = {key: record.model_dump(mode="json") for key, record in self.model_reference.items()}
        payload = json.dumps(serialised, indent=4)
        os.makedirs(self.models_db_path.parent, exist_ok=True)
        with self._file_mutex, self._file_lock:
            if self._using_multiprocessing:
                backup_path = f"{self.models_db_path}-backup-{uuid.uuid4().hex[:8]}.json"
                with open(backup_path, "w", encoding="utf-8", errors="ignore") as backup_file:
                    backup_file.write(payload)
            with open(self.models_db_path, "w", encoding="utf-8", errors="ignore") as outfile:
                outfile.write(payload)
            if self._using_multiprocessing:
                self.cleanup_reference_backup_files()

    def reload_reference_from_disk(self) -> None:
        """Reload the reference from disk and rebuild the in-memory indices."""
        with self._file_mutex, self._file_lock:
            raw = json.loads(self.models_db_path.read_text())
        self.model_reference = self._deserialise_reference(raw)
        self._rebuild_indices()

    def get_all_backup_files(self) -> list[Path]:
        """Return existing reference backup files, newest first."""
        pattern = f"{self.models_db_path}-backup-*.json"
        backups = glob.glob(pattern)
        try:
            backups.sort(key=os.path.getmtime, reverse=True)
        except FileNotFoundError:
            logger.warning("A reference backup file was removed by another process while sorting")
        return [Path(backup) for backup in backups]

    def cleanup_reference_backup_files(self, keep_latest: int = 3) -> None:
        """Delete all but the *keep_latest* most recent reference backup files."""
        for stale_backup in self.get_all_backup_files()[keep_latest:]:
            try:
                stale_backup.unlink()
            except FileNotFoundError:
                logger.warning("Expected to delete backup but it was already gone: {}", stale_backup)
            except OSError as delete_error:
                logger.warning("Error deleting backup file {}: {}", stale_backup, delete_error)

    @override
    def download_model_reference(self) -> dict:
        """Clear the reference and persist the empty state (the cache is rebuilt from downloads)."""
        self.model_reference = {}
        self.save_reference_to_disk()
        return {}

    # ------------------------------------------------------------------
    # CivitAI metadata fetching
    # ------------------------------------------------------------------

    def _fetch_civitai_json(self, url: str) -> dict | None:
        """Return the parsed JSON body of a CivitAI API GET, retrying transient failures.

        Returns ``None`` on a definitive failure (client error, exhausted retries, or repeated
        invalid JSON), mirroring CivitAI's quirk of returning 500 for impossible ids and HTML for
        some error pages.
        """
        fetch_logger = logger.bind(manager=self.METRIC_PREFIX)
        start_time = time.perf_counter()
        truncated_url = url[:100] + "..." if len(url) > 100 else url

        retries = 0
        while retries <= self.MAX_RETRIES:
            response = None
            try:
                timeout = self.REQUEST_METADATA_TIMEOUT if len(url) < 200 else self.REQUEST_METADATA_TIMEOUT * 1.5
                response = requests.get(url, timeout=timeout)
                response.raise_for_status()
                result = response.json()
                self._metric_metadata_duration.record((time.perf_counter() - start_time) * 1000)
                return result
            except (
                requests.HTTPError,
                requests.ConnectionError,
                requests.Timeout,
                json.JSONDecodeError,
            ) as fetch_error:
                error_type = type(fetch_error).__name__
                self._metric_network_errors.add(1, {"error_type": error_type})
                status_code = response.status_code if response is not None else None
                fetch_logger.warning(
                    "adhoc.metadata_fetch_error",
                    error_type=error_type,
                    url=truncated_url,
                    attempt=retries + 1,
                    status_code=status_code,
                )

                # 401/404 and 5xx above 500 will not resolve by retrying; give up immediately.
                if status_code in (401, 404) or (status_code is not None and status_code > 500):
                    return None
                # CivitAI's 500s and HTML-instead-of-JSON pages rarely clear quickly; burn most retries.
                if status_code == 500 or isinstance(fetch_error, json.JSONDecodeError):
                    retries += 3
                # A timeout already cost us REQUEST_METADATA_TIMEOUT seconds; don't keep waiting.
                if response is None:
                    retries += 5

                retries += 1
                self.total_retries_attempted += 1
                self._metric_retries.add(1, {"reason": error_type})
                if retries > self.MAX_RETRIES:
                    fetch_logger.error("adhoc.metadata_fetch_failed", total_attempts=retries)
                    return None
                time.sleep(self.RETRY_DELAY)
            except Exception as unexpected_error:
                fetch_logger.error("adhoc.metadata_fetch_exception", error=str(unexpected_error))
                return None
        return None

    # ------------------------------------------------------------------
    # Download queue and worker threads
    # ------------------------------------------------------------------

    def _enqueue_download(self, record: RecordT, context: dict | None = None) -> None:
        """Ensure the worker pool is running and append *record* to the download queue."""
        with self._download_mutex:
            if len(self._download_threads) < self.MAX_DOWNLOAD_THREADS:
                while len(self._download_threads) < self.MAX_DOWNLOAD_THREADS:
                    thread_number = len(self._download_threads)
                    worker = threading.Thread(target=self._download_thread, daemon=True, args=(thread_number,))
                    self._download_threads[thread_number] = {"thread": worker, "record": None}
                    worker.start()
                self._metric_active_threads.set(len(self._download_threads))

            self._download_queue.append(_QueuedDownload(record=record, context=context or {}))
            self._metric_queue_size.set(len(self._download_queue))

    def _download_thread(self, thread_number: int) -> None:
        """Worker loop: dequeue records and download their weight files until stopped.

        Failures are tolerated: CivitAI downloads fail often, and there is no value in blocking the
        pool on any single file, so the worker logs and moves on.
        """
        thread_logger = logger.bind(manager=self.METRIC_PREFIX, thread_number=thread_number)
        while True:
            if self._stop_all_threads:
                return
            try:
                queued = self._download_queue.popleft()
                self._download_threads[thread_number]["record"] = queued.record
                self._metric_queue_size.set(len(self._download_queue))
            except IndexError:
                self._download_threads[thread_number]["record"] = None
                time.sleep(self.THREAD_WAIT_TIME)
                continue

            self._process_download(queued, thread_logger)

    def _record_download_event(
        self,
        record: RecordT,
        *,
        success: bool,
        size_bytes: int,
        duration_seconds: float,
        retries: int,
    ) -> None:
        """Feed one finished download attempt into the in-process metrics collector."""
        megabytes = size_bytes / (1024 * 1024)
        category: DownloadCategory = (
            "lora" if self.METRIC_PREFIX == "lora" else "ti" if self.METRIC_PREFIX == "ti" else "other"
        )
        get_metrics_collector().record_download(
            DownloadEvent(
                name=record.name,
                category=category,
                size_bytes=size_bytes,
                duration_seconds=duration_seconds,
                megabytes_per_second=(megabytes / duration_seconds) if duration_seconds > 0 and success else 0.0,
                retries=retries,
                success=success,
                timestamp=time.time(),
            ),
        )

    def _process_download(self, queued: _QueuedDownload, thread_logger) -> None:
        """Download a single queued record's weight file, retrying transient failures."""
        record = queued.record
        download_logger = thread_logger.bind(model_name=record.name)

        overall_start = time.perf_counter()
        retries = 0
        while retries <= self.MAX_RETRIES:
            try:
                target = self._prepare_download(record)
                if target is None:
                    return
                filepath = os.path.join(self.model_folder_path, target.filename)
                hashpath = f"{os.path.splitext(filepath)[0]}.sha256"

                if self._existing_file_matches(filepath, hashpath, target.sha256):
                    self._commit_download(record, target, downloaded=False)
                    if self.is_default_cache_full():
                        self.stop_downloading = True
                    self.save_reference_to_disk()
                    return

                download_start = time.perf_counter()
                download_url = target.url
                if self._civitai_api_token and self.is_model_url_from_civitai(download_url):
                    download_url += f"{'&' if '?' in download_url else '?'}token={self._civitai_api_token}"

                response = requests.get(download_url, timeout=self.REQUEST_DOWNLOAD_TIMEOUT)
                response.raise_for_status()
                if "reason=download-auth" in response.url:
                    download_logger.error("adhoc.download_auth_redirect", response_url=response.url)
                    self._record_download_event(
                        record,
                        success=False,
                        size_bytes=0,
                        duration_seconds=time.perf_counter() - overall_start,
                        retries=retries,
                    )
                    return

                sha256 = hashlib.sha256(response.content).hexdigest()
                if target.sha256 and sha256.lower() != target.sha256.lower():
                    download_logger.warning(
                        "adhoc.download_hash_mismatch",
                        expected=target.sha256[:16],
                        actual=sha256[:16],
                        attempt=retries + 1,
                    )
                    retries += 1
                    self.total_retries_attempted += 1
                    if retries > self.MAX_RETRIES:
                        self._record_download_event(
                            record,
                            success=False,
                            size_bytes=0,
                            duration_seconds=time.perf_counter() - overall_start,
                            retries=retries,
                        )
                        return
                    time.sleep(self.RETRY_DELAY)
                    continue

                with open(filepath, "wb") as weight_file:
                    weight_file.write(response.content)
                with open(hashpath, "w") as hash_file:
                    hash_file.write(f"{sha256} *{target.filename}")

                self._commit_download(record, target, downloaded=True)
                self._metric_download_duration.record(time.perf_counter() - download_start)
                self._record_download_event(
                    record,
                    success=True,
                    size_bytes=len(response.content),
                    duration_seconds=time.perf_counter() - download_start,
                    retries=retries,
                )
                self._evict_adhoc_over_limit()
                self.save_reference_to_disk()
                download_logger.info("adhoc.download_success", filename=target.filename)
                return
            except SkipDownload:
                return
            except (requests.HTTPError, requests.ConnectionError, requests.Timeout, json.JSONDecodeError) as net_error:
                error_type = type(net_error).__name__
                self._metric_network_errors.add(1, {"error_type": error_type})
                self._metric_retries.add(1, {"reason": error_type})
                status_code = getattr(getattr(net_error, "response", None), "status_code", None)
                download_logger.warning("adhoc.download_network_error", error_type=error_type, status_code=status_code)
                if isinstance(net_error, requests.HTTPError) and status_code in (401, 403):
                    self._record_download_event(
                        record,
                        success=False,
                        size_bytes=0,
                        duration_seconds=time.perf_counter() - overall_start,
                        retries=retries,
                    )
                    return
            except Exception as fatal_error:
                self._metric_network_errors.add(1, {"error_type": "fatal"})
                download_logger.error("adhoc.download_fatal_error", error=str(fatal_error))

            retries += 1
            self.total_retries_attempted += 1
            if retries > self.MAX_RETRIES:
                download_logger.error("adhoc.download_max_retries")
                self._record_download_event(
                    record,
                    success=False,
                    size_bytes=0,
                    duration_seconds=time.perf_counter() - overall_start,
                    retries=retries,
                )
                return
            time.sleep(self.RETRY_DELAY)

    def _existing_file_matches(self, filepath: str, hashpath: str, expected_sha256: str | None) -> bool:
        """Return whether a previously downloaded file is present and matches *expected_sha256*."""
        if not (os.path.exists(filepath) and os.path.exists(hashpath)):
            return False
        try:
            with open(hashpath) as hash_file:
                stored_hash = hash_file.read().split()[0]
        except (IndexError, OSError) as read_error:
            logger.bind(manager=self.METRIC_PREFIX).error(
                "adhoc.hash_read_failed", hashpath=hashpath, error=str(read_error)
            )
            return False
        return not expected_sha256 or stored_hash.lower() == expected_sha256.lower()

    # ------------------------------------------------------------------
    # Completion tracking
    # ------------------------------------------------------------------

    def wait_for_downloads(self, timeout: float | None = None) -> None:
        """Block until the download queue drains and all workers are idle.

        Raises:
            TimeoutError: If *timeout* seconds elapse before downloads complete.
        """
        waited = 0.0
        while not self.are_downloads_complete():
            time.sleep(self.THREAD_WAIT_TIME)
            waited += self.THREAD_WAIT_TIME
            if timeout and waited > timeout:
                raise TimeoutError(f"{self.METRIC_PREFIX} downloads exceeded specified timeout ({timeout})")

    def are_downloads_complete(self) -> bool:
        """Return whether the metadata thread, all workers, and the queue are all idle."""
        if self._thread and self._thread.is_alive():
            return False
        if not self.are_download_threads_idle():
            return False
        if len(self._download_queue) > 0:
            return False
        return self.stop_downloading

    def are_download_threads_idle(self) -> bool:
        """Return whether every worker thread is currently idle (holding no record)."""
        return all(worker["record"] is None for worker in self._download_threads.values())

    def is_adhoc_reset_complete(self) -> bool:
        """Return whether the background ad-hoc reset thread has finished (or never ran)."""
        return not (self._adhoc_reset_thread and self._adhoc_reset_thread.is_alive())

    def wait_for_adhoc_reset(self, timeout: float = 15) -> None:
        """Block until the background ad-hoc reset completes.

        Raises:
            TimeoutError: If *timeout* seconds elapse first.
        """
        waited = 0.0
        while not self.is_adhoc_reset_complete():
            time.sleep(self.THREAD_WAIT_TIME)
            waited += self.THREAD_WAIT_TIME
            if timeout and waited > timeout:
                raise TimeoutError(f"{self.METRIC_PREFIX} adhoc reset exceeded specified timeout ({timeout})")

    def stop_all(self) -> None:
        """Signal all worker threads to stop at their next iteration."""
        self._stop_all_threads = True

    # ------------------------------------------------------------------
    # Cache accounting and eviction (generic over CacheEntry)
    # ------------------------------------------------------------------

    def find_adhoc_models(self) -> set[str]:
        """Return the reference keys of models that have at least one ad-hoc cache entry."""
        return {entry.model_key for entry in self._iter_cache_entries() if entry.adhoc}

    def calculate_downloaded_cache_mb(self, *, adhoc_only: bool = False, default_only: bool = False) -> float:
        """Return the total cached size in megabytes, optionally restricted to ad-hoc/default entries."""
        adhoc_keys = self.find_adhoc_models()
        total = 0.0
        for entry in self._iter_cache_entries():
            if default_only and entry.model_key in adhoc_keys:
                continue
            if adhoc_only and entry.model_key not in adhoc_keys:
                continue
            total += entry.size_mb
        return total

    def calculate_adhoc_cache(self) -> float:
        """Return the total size in megabytes of ad-hoc cache entries."""
        return self.calculate_downloaded_cache_mb(adhoc_only=True)

    def calculate_default_cache(self) -> float:
        """Return the total size in megabytes of default (non-ad-hoc) cache entries."""
        return self.calculate_downloaded_cache_mb(default_only=True)

    def is_default_cache_full(self) -> bool:
        """Return whether the default cache has reached its budget."""
        return self.calculate_default_cache() >= self._max_top_disk * 1024

    def is_adhoc_cache_full(self) -> bool:
        """Return whether the ad-hoc cache has reached its budget."""
        return self.calculate_adhoc_cache() >= self.max_adhoc_disk * 1024

    def amount_of_adhoc_to_delete(self) -> int:
        """Return how many ad-hoc entries to evict: one, plus one more per 4GB over budget."""
        if not self.is_adhoc_cache_full():
            return 0
        return 1 + int((self.calculate_adhoc_cache() - self.max_adhoc_disk * 1024) / 4096)

    def find_oldest_adhoc_entry(self) -> CacheEntry | None:
        """Return the least-recently-used ad-hoc cache entry, or ``None`` if there are none."""
        oldest: CacheEntry | None = None
        for entry in self._iter_cache_entries():
            if not entry.adhoc:
                continue
            if oldest is None or (entry.last_used or datetime.now()) < (oldest.last_used or datetime.now()):
                oldest = entry
        return oldest

    def delete_oldest(self) -> None:
        """Evict the least-recently-used ad-hoc cache entry, if any."""
        oldest = self.find_oldest_adhoc_entry()
        if oldest is None:
            return
        self._delete_model_entry(oldest.model_key, oldest.version_key)

    def _evict_adhoc_over_limit(self) -> None:
        """Evict ad-hoc entries until the ad-hoc cache is back within budget."""
        for _eviction in range(self.amount_of_adhoc_to_delete()):
            self.delete_oldest()

    # ------------------------------------------------------------------
    # Unused-file cleanup (shared)
    # ------------------------------------------------------------------

    def find_unused_files(self) -> set[str]:
        """Return weight filenames present on disk that are not referenced by any record."""
        known = {entry.filename for entry in self._iter_cache_entries()}
        unused = set()
        for weight_file in glob.glob(f"{self.model_folder_path}/*.safetensors"):
            filename = os.path.basename(weight_file)
            if filename not in known:
                unused.add(filename)
        return unused

    def delete_unused_models(self, timeout: float = 0) -> set[str]:
        """Delete on-disk weight files not referenced by any record once downloads are complete.

        Raises:
            TimeoutError: If downloads do not complete within *timeout* seconds.
        """
        waited = 0.0
        while not self.are_downloads_complete():
            if waited >= timeout:
                raise TimeoutError(
                    f"Waiting for {self.METRIC_PREFIX} downloads exceeded specified timeout ({timeout})"
                )
            waited += 0.2
            time.sleep(0.2)
        unused = self.find_unused_files()
        for filename in unused:
            self._delete_weight_files(filename)
        return unused

    def _delete_weight_files(self, filename: str) -> None:
        """Delete a weight file and its companion ``.sha256`` from the model folder, if present."""
        weight_path = os.path.join(self.model_folder_path, filename)
        if not os.path.exists(weight_path):
            logger.bind(manager=self.METRIC_PREFIX).warning("adhoc.delete_file_missing", filename=filename)
            return
        os.remove(weight_path)
        hash_path = f"{os.path.splitext(weight_path)[0]}.sha256"
        if os.path.exists(hash_path):
            os.remove(hash_path)
        logger.bind(manager=self.METRIC_PREFIX).info("adhoc.delete_file_removed", filename=filename)

    # ------------------------------------------------------------------
    # Provider integration
    # ------------------------------------------------------------------

    def current_records(self) -> dict[str, RecordT]:
        """Return a snapshot of the current reference, for the CivitAI provider to expose."""
        return dict(self.model_reference)

    # ------------------------------------------------------------------
    # Shared fuzzy lookup
    # ------------------------------------------------------------------

    def _fuzzy_find_key(self, name: int | str) -> str | None:
        """Return the reference key best matching *name* (by id, exact, substring, then fuzzy)."""
        if isinstance(name, int) or name.isdigit():
            return self._index_ids.get(int(name))
        if name in self.model_reference:
            return name
        lowered = name.lower()
        if lowered in self._index_orig_names:
            return self._index_orig_names[lowered]
        from hordelib.utils.sanitizer import Sanitizer

        if Sanitizer.has_unicode(name):
            for orig_name, key in self._index_orig_names.items():
                if name in orig_name:
                    return key
            # Unicode names are transliterated in the keys, so a miss here is a definitive miss.
            return None
        for key in self.model_reference:
            if name in key:
                return key
        for key, record in self.model_reference.items():
            if fuzz.ratio(name, key) > self.FUZZ_THRESHOLD:
                return key
            if fuzz.ratio(name, self._orig_name(record)) > self.FUZZ_THRESHOLD:
                return key
        return None

    def _orig_name(self, record: RecordT) -> str:
        """Return the record's original (unsanitised) CivitAI name, used for fuzzy matching.

        The CivitAI records carry ``orig_name``; ``getattr`` keeps this tolerant of the generic
        ``GenericModelRecord`` bound and falls back to the reference name.
        """
        return getattr(record, "orig_name", "") or record.name

    # ------------------------------------------------------------------
    # Abstract hooks
    # ------------------------------------------------------------------

    RECORD_TYPE: type[RecordT]
    """The concrete record class this manager validates and stores."""

    @abstractmethod
    def _parse_civitai_item(self, item: dict, *, adhoc: bool = False) -> RecordT | None:
        """Return a validated record for a CivitAI API item, or ``None`` if it should be rejected."""

    @abstractmethod
    def _iter_cache_entries(self) -> Iterable[CacheEntry]:
        """Yield a :class:`CacheEntry` for each downloaded weight file across all records."""

    @abstractmethod
    def _prepare_download(self, record: RecordT) -> DownloadTarget | None:
        """Return the weight file to download for *record*, or ``None`` to skip it.

        May raise :class:`SkipDownload` to abandon the download or a ``requests`` error to retry.
        """

    @abstractmethod
    def _commit_download(self, record: RecordT, target: DownloadTarget, *, downloaded: bool) -> None:
        """Record a successful download/verification of *target* into the reference and indices."""

    @abstractmethod
    def _rebuild_indices(self) -> None:
        """Rebuild ``_index_ids``/``_index_orig_names`` (and any subclass indices) from the reference."""

    @abstractmethod
    def _drop_entry(self, model_key: str, version_key: str | None) -> None:
        """Remove a cache entry from the in-memory reference and indices only (no disk I/O)."""

    @abstractmethod
    def _delete_model_entry(self, model_key: str, version_key: str | None) -> None:
        """Remove a single cache entry (a version for LoRA, the whole model for TI) and persist."""
