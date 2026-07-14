"""The Textual Inversion (embedding) model manager: ad-hoc download and caching of CivitAI TIs.

A thin specialisation of :class:`~hordelib.model_manager.civitai_adhoc.CivitaiAdhocModelManager` for
the ``ti`` category. Textual inversions are simpler than LoRAs (a single version per model) but resolve
their actual download URL and checksum through the Hordeling embedding API at download time rather than
from the CivitAI metadata. Cache accounting and eviction are inherited from the base — unlike the
previous bespoke implementation, the TI cache is now genuinely size-bounded.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from enum import auto
from multiprocessing.synchronize import Lock as multiprocessing_lock
from typing import override

import logfire
import requests
from fuzzywuzzy import fuzz
from horde_model_reference import horde_model_reference_paths
from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORY
from horde_model_reference.model_reference_records import DownloadRecord
from strenum import StrEnum

from hordelib.model_manager.civitai_adhoc import (
    CacheEntry,
    CivitaiAdhocModelManager,
    DownloadTarget,
    SkipDownload,
    normalise_a1111_triggers,
    now_timestamp,
    timestamp_to_datetime,
)
from hordelib.model_manager.civitai_records import HordeTextualInversionModelRecord
from hordelib.utils.sanitizer import Sanitizer

MAX_ADHOC_TI_SIZE_KB = 20000
"""Textual inversions larger than this (~20MB) are not downloaded ad-hoc."""

TI_CACHE_SIZE_DEFAULT_MB = 1024
"""The default cache budget in megabytes; embeddings are tiny, so this rarely binds."""


class TIRejectionReason(StrEnum):
    """Reasons a textual inversion is permanently rejected from ad-hoc download."""

    NOT_FOUND = auto()
    """No embedding matched the request (CivitAI returned no items, or Hordeling reported a 404)."""
    UNEXPECTED_TYPE = auto()
    """Hordeling reported the resolved file is the wrong type or its hash does not match."""
    INVALID = auto()
    """The embedding's metadata is invalid or incomplete (impossible id, missing checksum, etc.)."""


class TextualInversionModelManager(CivitaiAdhocModelManager[HordeTextualInversionModelRecord]):
    """Downloads and caches CivitAI textual inversions (a single version per embedding)."""

    TI_API = "https://civitai.com/api/v1/models?types=TextualInversion&sort=Highest%20Rated&primaryFileOnly=true"
    """The CivitAI search endpoint used for name-based ad-hoc lookups."""
    HORDELING_API = "https://hordeling.aihorde.net/api/v1/embedding"
    """The Hordeling endpoint that resolves an embedding id to its download URL and checksum."""

    RECORD_TYPE = HordeTextualInversionModelRecord
    MODEL_CATEGORY = MODEL_REFERENCE_CATEGORY.ti
    METRIC_PREFIX = "ti"
    MAX_DOWNLOAD_THREADS = 3
    REQUEST_METADATA_TIMEOUT = 30
    REQUEST_DOWNLOAD_TIMEOUT = 30
    THREAD_WAIT_TIME = 2
    FUZZ_THRESHOLD = 80

    def __init__(
        self,
        download_reference: bool = False,
        multiprocessing_lock: multiprocessing_lock | None = None,
        civitai_api_token: str | None = None,
        *,
        read_only: bool = False,
        **kwargs,
    ) -> None:
        """Create the textual inversion manager.

        Args:
            download_reference: Accepted for parity; ad-hoc managers build their reference on demand.
            multiprocessing_lock: Optional cross-process lock guarding on-disk reference writes.
            civitai_api_token: Optional CivitAI API token.
            read_only: When ``True``, the manager is a pure reader that never writes, downloads, or
                evicts; mutating calls raise
                :class:`~hordelib.model_manager.civitai_adhoc.ReadOnlyModelManagerError`.
            **kwargs: Ignored; accepted for uniform construction across managers.
        """
        # Terminal Hordeling rejections observed by download workers in _prepare_download, keyed by
        # CivitAI id and drained on read so a reason from one fetch cannot leak into a later one.
        self._adhoc_rejection_reasons: dict[int, TIRejectionReason] = {}
        models_db_path = horde_model_reference_paths.legacy_path.joinpath("ti.json").resolve()
        super().__init__(
            model_category=MODEL_REFERENCE_CATEGORY.ti,
            models_db_path=models_db_path,
            civitai_api_token=civitai_api_token,
            download_reference=download_reference,
            multiprocessing_lock=multiprocessing_lock,
            max_top_disk=TI_CACHE_SIZE_DEFAULT_MB,
            max_adhoc_disk=TI_CACHE_SIZE_DEFAULT_MB,
            read_only=read_only,
        )

    # ------------------------------------------------------------------
    # Reference lookups
    # ------------------------------------------------------------------

    def fuzzy_find_ti_key(self, ti_name: str | int) -> str | None:
        """Return the reference key best matching *ti_name* (case-insensitive), or ``None``."""
        if isinstance(ti_name, int) or ti_name.isdigit():
            return self._fuzzy_find_key(ti_name)
        return self._fuzzy_find_key(ti_name.lower().strip())

    @override
    def get_model_reference_info(self, model_name: str | int) -> HordeTextualInversionModelRecord | None:
        """Return the record for *model_name*, or ``None`` if not found."""
        ti_key = self.fuzzy_find_ti_key(model_name)
        return self.model_reference.get(ti_key) if ti_key else None

    def get_ti_name(self, model_name: str | int) -> str | None:
        """Return the embedding's display name for *model_name*, or ``None`` if not found."""
        record = self.get_model_reference_info(model_name)
        return record.name if record else None

    def get_ti_id(self, model_name: str | int) -> int | None:
        """Return the CivitAI id for *model_name*, or ``None`` if not found."""
        record = self.get_model_reference_info(model_name)
        return record.civitai_id if record else None

    def get_ti_filename(self, model_name: str | int) -> str | None:
        """Return the weight filename for *model_name*, or ``None`` if not found."""
        record = self.get_model_reference_info(model_name)
        return record.filename if record else None

    def get_ti_triggers(self, model_name: str | int) -> list[str] | None:
        """Return the trigger list for *model_name* (empty if none), or ``None`` if not found."""
        record = self.get_model_reference_info(model_name)
        if not record:
            return None
        # Return a copy so callers cannot mutate the cached list.
        return list(record.trigger) if record.trigger else []

    def find_ti_trigger(self, model_name: str | int, trigger_search: str) -> str | None:
        """Return the trigger best matching *trigger_search* for *model_name*, or ``None``."""
        triggers = self.get_ti_triggers(model_name)
        if triggers is None:
            return None
        if trigger_search.lower() in [trigger.lower() for trigger in triggers]:
            return trigger_search
        for trigger in triggers:
            if trigger_search.lower() in trigger.lower():
                return trigger
        for trigger in triggers:
            if fuzz.ratio(trigger_search.lower(), trigger.lower()) > 65:
                return trigger
        return None

    def is_local_model(self, model_name: str | int) -> bool:
        """Return whether *model_name* resolves to a known embedding."""
        return self.fuzzy_find_ti_key(model_name) is not None

    @override
    def get_available_models(self) -> list[str]:
        """Return the reference keys of all known embeddings."""
        return list(self.model_reference.keys())

    def touch_ti(self, ti_name: str | int) -> None:
        """Mark *ti_name* as used now (in memory)."""
        record = self.get_model_reference_info(ti_name)
        if record:
            record.last_used = now_timestamp()

    def do_baselines_match(self, ti_name: str | int, model_details) -> bool:
        """Return whether the embedding's baseline is compatible (currently always ``True``)."""
        return True  # FIXME: baseline matching disabled pending normalised baseline data.

    # ------------------------------------------------------------------
    # CivitAI parsing
    # ------------------------------------------------------------------

    @override
    def _parse_civitai_item(self, item: dict, *, adhoc: bool = False) -> HordeTextualInversionModelRecord | None:
        """Return a :class:`HordeTextualInversionModelRecord` for a CivitAI item, or ``None``."""
        try:
            version_data = item.get("modelVersions", {})[0]
        except IndexError:
            version_data = {}
        triggers = normalise_a1111_triggers(version_data.get("trainedWords", []))

        primary_file = next((f for f in version_data.get("files", []) if f.get("primary", False)), None)
        if primary_file is None:
            return None

        try:
            size_kb = round(primary_file.get("sizeKB", 0))
        except TypeError:
            size_kb = 24  # Common embedding size; not critical when CivitAI omits it.

        civitai_id = item.get("id", 0)
        nsfw = item.get("nsfw", True)
        record = HordeTextualInversionModelRecord(
            name=Sanitizer.sanitise_model_name(item.get("name", "")),
            civitai_id=civitai_id,
            orig_name=item.get("name", ""),
            filename=f"{civitai_id}.safetensors",
            url=primary_file.get("downloadUrl", ""),
            sha256=primary_file.get("hashes", {}).get("SHA256"),
            size_kb=size_kb,
            nsfw=nsfw,
            trigger=triggers,
            base_model=version_data.get("baseModel", "SD 1.5"),
            version_id=version_data.get("id", 0),
            adhoc=adhoc,
        )

        if adhoc and not record.sha256:
            return None
        if not record.filename or not record.url:
            return None
        if adhoc and record.size_kb > MAX_ADHOC_TI_SIZE_KB:
            return None
        if adhoc and record.nsfw and not self.nsfw:
            return None

        self._sync_top_level(record)
        return record

    def _sync_top_level(self, record: HordeTextualInversionModelRecord) -> None:
        """Mirror the embedding's details onto its canonical provider-facing fields."""
        record.version = str(record.version_id)
        record.baseline = record.base_model
        record.config.download = [
            DownloadRecord(
                file_name=record.filename,
                file_url=record.url,
                sha256sum=record.sha256 or "FIXME",
            ),
        ]

    # ------------------------------------------------------------------
    # Download hooks
    # ------------------------------------------------------------------

    def _record_rejection_reason(self, civitai_id: int, reason: TIRejectionReason) -> None:
        """Remember a terminal rejection *reason* for *civitai_id* so the fetch caller can report it."""
        with self._mutex:
            self._adhoc_rejection_reasons[civitai_id] = reason

    def _drain_rejection_reason(self, civitai_id: int) -> TIRejectionReason | None:
        """Return and forget the terminal rejection reason recorded for *civitai_id*, if any."""
        with self._mutex:
            return self._adhoc_rejection_reasons.pop(civitai_id, None)

    @override
    def _prepare_download(self, record: HordeTextualInversionModelRecord) -> DownloadTarget | None:
        """Resolve *record*'s download URL and checksum via the Hordeling API.

        Raises:
            SkipDownload: If Hordeling reports the embedding is unavailable or mismatched.
            requests.HTTPError: For transient Hordeling errors, so the base retries.
        """
        response = requests.get(f"{self.HORDELING_API}/{record.civitai_id}", timeout=5)
        if not response.ok:
            if response.status_code == 404:
                self._record_rejection_reason(record.civitai_id, TIRejectionReason.NOT_FOUND)
                raise SkipDownload
            payload = response.json()
            message = payload.get("message", "")
            message = message.lower() if isinstance(message, str) else ""
            if "unexpected type" in message or "hash" in message:
                self._record_rejection_reason(record.civitai_id, TIRejectionReason.UNEXPECTED_TYPE)
                raise SkipDownload
            response.raise_for_status()  # Transient (e.g. 500); let the base retry.

        payload = response.json()
        sha256 = payload.get("sha256") or record.sha256
        return DownloadTarget(
            filename=record.filename,
            url=payload["url"],
            size_mb=record.size_kb / 1024,
            sha256=sha256,
        )

    @override
    def _commit_download(
        self,
        record: HordeTextualInversionModelRecord,
        target: DownloadTarget,
        *,
        downloaded: bool,
    ) -> None:
        """Add *record* to the reference and indices, recording the resolved checksum."""
        with self._mutex:
            if target.sha256:
                record.sha256 = target.sha256
            if downloaded:
                record.last_used = now_timestamp()
            record.last_checked = now_timestamp()
            self._sync_top_level(record)

            ti_key = record.name.lower().strip()
            self.model_reference[ti_key] = record
            self._index_ids[record.civitai_id] = ti_key
            self._index_orig_names[record.orig_name.lower().strip()] = ti_key

    @override
    def _rebuild_indices(self) -> None:
        """Rebuild the id and original-name indices from the reference."""
        self._index_ids = {}
        self._index_orig_names = {}
        for ti_key, record in self.model_reference.items():
            self._index_ids[record.civitai_id] = ti_key
            self._index_orig_names[record.orig_name.lower().strip()] = ti_key

    @override
    def _iter_cache_entries(self) -> Iterable[CacheEntry]:
        """Yield one :class:`CacheEntry` per known embedding."""
        for ti_key, record in self.model_reference.items():
            yield CacheEntry(
                model_key=ti_key,
                version_key=None,
                filename=record.filename,
                size_mb=record.size_kb / 1024,
                adhoc=record.adhoc,
                last_used=timestamp_to_datetime(record.last_used),
            )

    @override
    def _drop_entry(self, model_key: str, version_key: str | None) -> None:
        """Remove an embedding from memory and indices."""
        record = self.model_reference.get(model_key)
        if record is None:
            return
        self._index_ids.pop(record.civitai_id, None)
        self._index_orig_names.pop(record.orig_name.lower().strip(), None)
        del self.model_reference[model_key]

    @override
    def _delete_model_entry(self, model_key: str, version_key: str | None) -> None:
        """Delete an embedding's weight file, forget it, and persist.

        Raises:
            ReadOnlyModelManagerError: If the manager is read-only.
        """
        self._ensure_writable()
        record = self.model_reference.get(model_key)
        if record is None:
            return
        self._delete_weight_files(record.filename)
        self._drop_entry(model_key, None)
        self.save_reference_to_disk()

    def delete_ti(self, ti_name: str) -> None:
        """Delete the embedding keyed by *ti_name* and persist."""
        self._delete_model_entry(ti_name, None)

    def ensure_ti_deleted(self, ti_name: str | int) -> None:
        """Delete the embedding matching *ti_name*, if present."""
        ti_key = self.fuzzy_find_ti_key(ti_name)
        if ti_key:
            self._delete_model_entry(ti_key, None)

    # ------------------------------------------------------------------
    # Ad-hoc fetch
    # ------------------------------------------------------------------

    @logfire.instrument("ti.fetch_adhoc", extract_args=True)
    def fetch_adhoc_ti(
        self,
        ti_name: str | int,
        timeout: float = 15,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> str | None:
        """Ensure an embedding is available, downloading it from CivitAI/Hordeling on demand.

        Blocks until this embedding's own download finishes (unrelated queued downloads no longer extend
        the wait) and returns the reference key on success, or ``None`` if it could not be found or
        downloaded, or if *timeout* elapsed before the download finished (raising nothing).

        Args:
            ti_name: The CivitAI id or name to resolve.
            timeout: Seconds to wait for the download to finish.
            progress_callback: Optional hook invoked as ``(downloaded_bytes, total_bytes)`` per streamed
                chunk while the weight file is written; ``total_bytes`` is ``0`` when unknown.

        Raises:
            ReadOnlyModelManagerError: If the manager is read-only.
        """
        ti_key, _ = self.fetch_adhoc_ti_with_reason(
            ti_name=ti_name,
            timeout=timeout,
            progress_callback=progress_callback,
        )
        return ti_key

    @logfire.instrument("ti.fetch_adhoc_with_reason", extract_args=True)
    def fetch_adhoc_ti_with_reason(
        self,
        ti_name: str | int,
        timeout: float = 15,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> tuple[
        str | None,
        TIRejectionReason | None,
    ]:
        """Ensure an embedding is available, downloading it from CivitAI/Hordeling on demand.

        Behaves like :meth:`fetch_adhoc_ti`, additionally returning why a request produced no key. A
        successful fetch yields ``(reference_key, None)``. A transient failure (metadata fetch failure,
        an elapsed *timeout*, or a plain download failure) yields ``(None, None)``. A permanent upstream
        rejection yields ``(None, reason)``.

        Args:
            ti_name: The CivitAI id or name to resolve.
            timeout: Seconds to wait for the download to finish.
            progress_callback: Optional hook invoked as ``(downloaded_bytes, total_bytes)`` per streamed
                chunk while the weight file is written; ``total_bytes`` is ``0`` when unknown.

        Raises:
            ReadOnlyModelManagerError: If the manager is read-only.
        """
        self._ensure_writable()
        if isinstance(ti_name, int) or str(ti_name).isdigit():
            # CivitAI returns 500 (not 404) for ids beyond its id space; reject impossible ids early.
            if int(ti_name) >= 2**32:
                return None, TIRejectionReason.INVALID
            url = f"https://civitai.com/api/v1/models/{ti_name}"
        else:
            url = f"{self.TI_API}&nsfw={str(self.nsfw).lower()}&query={ti_name}"

        data = self._fetch_civitai_json(url)
        if not data:
            # A None body conflates a genuine "not found" with an exhausted-retry transient, so it stays
            # reason-less rather than asserting a terminal verdict the metadata layer cannot distinguish.
            return None, None
        if "items" in data:
            if len(data["items"]) == 0:
                return None, TIRejectionReason.NOT_FOUND
            record = self._parse_civitai_item(data["items"][0], adhoc=True)
        else:
            record = self._parse_civitai_item(data, adhoc=True)
        if not record:
            return None, TIRejectionReason.INVALID

        existing_key = self.fuzzy_find_ti_key(record.civitai_id)
        if existing_key:
            return existing_key, None

        # Clear any stale terminal reason for this id so this attempt reports only its own outcome.
        self._drain_rejection_reason(record.civitai_id)
        queued = self._enqueue_download(record, progress_callback=progress_callback)
        # Wait on this record's own completion rather than the whole pool going idle, so unrelated queued
        # downloads can no longer blow this bounded wait. On timeout/failure re-probe the reference so a
        # concurrent fetch that already landed this embedding is still reported.
        queued.completion_event.wait(timeout)
        if queued.success:
            return record.name.lower(), None
        reason = self._drain_rejection_reason(record.civitai_id)
        if reason is not None:
            return None, reason
        existing_key = self.fuzzy_find_ti_key(record.civitai_id)
        if existing_key:
            return existing_key, None
        return None, None
