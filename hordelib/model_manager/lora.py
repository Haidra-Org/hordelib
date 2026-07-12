"""The LoRA model manager: ad-hoc download and caching of CivitAI LoRAs.

A thin specialisation of :class:`~hordelib.model_manager.civitai_adhoc.CivitaiAdhocModelManager` for
the ``lora`` category. Everything generic (metadata fetching, the download thread pool, cache
accounting and eviction, persistence) is inherited; this module adds only what is specific to LoRAs:
their multiple-versions-per-model shape, the per-version index, the 400MB ad-hoc size cap, the
default-LoRA seeding flow, and the ad-hoc fetch/metadata-match logic.
"""

from __future__ import annotations

import os
import threading
import time
from datetime import datetime
from multiprocessing.synchronize import Lock as multiprocessing_lock
from typing import override

import logfire
from fuzzywuzzy import fuzz
from horde_model_reference import horde_model_reference_paths
from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORY
from horde_model_reference.model_reference_records import DownloadRecord
from loguru import logger

import hordelib.exceptions as he
from hordelib.model_manager.civitai_adhoc import (
    DEFAULT_MIN_FREE_DISK_MB,
    TESTS_ONGOING,
    CacheEntry,
    CivitaiAdhocModelManager,
    DownloadTarget,
    normalise_a1111_triggers,
    now_timestamp,
    timestamp_to_datetime,
)
from hordelib.model_manager.civitai_records import HordeLoraModelRecord, LoraVersionEntry
from hordelib.utils.sanitizer import Sanitizer

AIWORKER_LORA_CACHE_SIZE_DEFAULT = 10 * 1024  # 10GB
"""The default ad-hoc LoRA cache budget in megabytes when ``AIWORKER_LORA_CACHE_SIZE`` is unset."""

MAX_ADHOC_LORA_SIZE_MB = 400
"""LoRAs larger than this are not downloaded ad-hoc (defaults are exempt)."""


class LoraModelManager(CivitaiAdhocModelManager[HordeLoraModelRecord]):
    """Downloads and caches CivitAI LoRAs, tracking multiple versions per LoRA."""

    LORA_DEFAULTS = (
        "https://raw.githubusercontent.com/Haidra-Org/AI-Horde-image-model-reference/refs/heads/qwen/lora.json"
    )
    """The URL of the curated default-LoRA id list seeded on first download."""
    LORA_API = "https://civitai.com/api/v1/models?types=LORA&sort=Highest%20Rated&primaryFileOnly=true"
    """The CivitAI search endpoint used for name-based ad-hoc lookups."""

    RECORD_TYPE = HordeLoraModelRecord
    MODEL_CATEGORY = MODEL_REFERENCE_CATEGORY.lora
    METRIC_PREFIX = "lora"
    REQUEST_METADATA_TIMEOUT = 35  # CivitAI is slow on multi-model metadata requests.

    def __init__(
        self,
        download_reference: bool = False,
        allowed_top_lora_storage: int = 10240 if not TESTS_ONGOING else 1024,
        allowed_adhoc_lora_storage: int = AIWORKER_LORA_CACHE_SIZE_DEFAULT,
        download_wait: bool = False,
        multiprocessing_lock: multiprocessing_lock | None = None,
        civitai_api_token: str | None = None,
        reference_backups: bool | None = None,
        *,
        read_only: bool = False,
    ) -> None:
        """Create the LoRA manager.

        Args:
            download_reference: Accepted for parity; ad-hoc managers build their reference on demand.
            allowed_top_lora_storage: The default-set cache budget, in megabytes.
            allowed_adhoc_lora_storage: The ad-hoc cache budget, in megabytes. Overridden by the
                ``AIWORKER_LORA_CACHE_SIZE`` environment variable when set.
                The ``AIWORKER_LORA_MIN_DISK_FREE_MB`` environment variable likewise overrides the
                free-space floor below which ad-hoc downloads evict (then refuse) to spare the disk.
            download_wait: Whether :meth:`download_default_models` blocks until downloads complete.
            multiprocessing_lock: Optional cross-process lock guarding on-disk reference writes.
            civitai_api_token: Optional CivitAI API token.
            reference_backups: Whether to write timestamped reference backups on each save.
            read_only: When ``True``, the manager is a pure reader that never writes, downloads, or
                evicts; mutating calls raise :class:`~hordelib.model_manager.civitai_adhoc.ReadOnlyModelManagerError`.
        """
        self._index_version_ids: dict[str, str] = {}
        self._default_lora_ids: list = []
        self._download_wait = download_wait

        adhoc_storage = allowed_adhoc_lora_storage
        env_cache_size = os.getenv("AIWORKER_LORA_CACHE_SIZE")
        if env_cache_size is not None:
            try:
                adhoc_storage = int(env_cache_size)
            except (ValueError, TypeError):
                logger.bind(manager="lora").warning("lora.env_cache_size_invalid", raw_value=env_cache_size)
                adhoc_storage = AIWORKER_LORA_CACHE_SIZE_DEFAULT

        min_free_disk_mb = DEFAULT_MIN_FREE_DISK_MB
        env_min_free = os.getenv("AIWORKER_LORA_MIN_DISK_FREE_MB")
        if env_min_free is not None:
            try:
                min_free_disk_mb = int(env_min_free)
            except (ValueError, TypeError):
                logger.bind(manager="lora").warning("lora.env_min_free_invalid", raw_value=env_min_free)
                min_free_disk_mb = DEFAULT_MIN_FREE_DISK_MB

        models_db_path = horde_model_reference_paths.legacy_path.joinpath("lora.json").resolve()
        super().__init__(
            model_category=MODEL_REFERENCE_CATEGORY.lora,
            models_db_path=models_db_path,
            civitai_api_token=civitai_api_token,
            download_reference=download_reference,
            multiprocessing_lock=multiprocessing_lock,
            reference_backups=reference_backups,
            max_top_disk=allowed_top_lora_storage,
            max_adhoc_disk=adhoc_storage,
            min_free_disk_mb=min_free_disk_mb,
            read_only=read_only,
        )

    def ensure_is_version(self, lora_version: int | str) -> str | None:
        """Return *lora_version* as a string (JSON keys must be strings), or ``None`` if unusable."""
        if not isinstance(lora_version, (str, int)):
            return None
        return str(lora_version)

    def find_latest_version(self, record: HordeLoraModelRecord | None) -> str | None:
        """Return the highest non-EarlyAccess version id for *record*, or ``None``."""
        if record is None:
            return None
        usable = [
            int(version_id) for version_id, version in record.versions.items() if version.availability != "EarlyAccess"
        ]
        if not usable:
            return None
        return str(max(usable))

    def get_latest_version(self, record: HordeLoraModelRecord | None) -> LoraVersionEntry | None:
        """Return the latest usable :class:`LoraVersionEntry` for *record*, or ``None``."""
        if record is None:
            return None
        version_id = self.find_latest_version(record)
        return record.versions.get(version_id) if version_id is not None else None

    def find_lora_key_by_version(self, lora_version: int | str) -> str | None:
        """Return the reference key owning *lora_version*, or ``None``."""
        version = self.ensure_is_version(lora_version)
        return self._index_version_ids.get(version) if version is not None else None

    def fuzzy_find_lora_key(self, lora_name: int | str) -> str | None:
        """Return the reference key best matching *lora_name* (by id, exact, substring, or fuzzy)."""
        return self._fuzzy_find_key(lora_name)

    @override
    def get_model_reference_info(
        self,
        model_name: str | int,
        is_version: bool = False,
    ) -> HordeLoraModelRecord | None:
        """Return the record for *model_name* (or version id when *is_version*), or ``None``."""
        with self._mutex:
            if is_version:
                lora_key = self.find_lora_key_by_version(model_name)
                if lora_key is None or lora_key not in self.model_reference:
                    return None
                return self.model_reference[lora_key]
            lora_key = self.fuzzy_find_lora_key(model_name)
            if not lora_key:
                return None
            return self.model_reference.get(lora_key)

    def get_lora_name(self, model_name: str | int, is_version: bool = False) -> str | None:
        """Return the canonical reference key for *model_name*, or ``None`` if not found."""
        record = self.get_model_reference_info(model_name, is_version)
        return record.name if record else None

    def get_lora_filename(self, model_name: str | int, is_version: bool = False) -> str | None:
        """Return the weight filename for *model_name*, or ``None`` if not found."""
        record = self.get_model_reference_info(model_name, is_version)
        if not record:
            return None
        if is_version:
            version = record.versions.get(self.ensure_is_version(model_name) or "")
            return version.filename if version else None
        latest = self.get_latest_version(record)
        return latest.filename if latest else None

    def get_lora_triggers(self, model_name: str | int, is_version: bool = False) -> list[str] | None:
        """Return the trigger list for *model_name* (empty if none), or ``None`` if not found."""
        record = self.get_model_reference_info(model_name, is_version)
        if not record:
            return None
        if is_version:
            version = record.versions.get(self.ensure_is_version(model_name) or "")
            triggers = version.triggers if version else None
        else:
            latest = self.get_latest_version(record)
            triggers = latest.triggers if latest else None
        # Return a copy so callers cannot mutate the cached list.
        return list(triggers) if triggers else []

    def find_lora_trigger(self, model_name: str | int, trigger_search: str, is_version: bool = False) -> str | None:
        """Return the trigger best matching *trigger_search* for *model_name*, or ``None``."""
        triggers = self.get_lora_triggers(model_name, is_version)
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

    def calculate_downloaded_loras(self) -> float:
        """Return the total size in megabytes of all downloaded LoRA versions."""
        return self.calculate_downloaded_cache_mb()

    @override
    def is_model_available(self, model_name: str | int) -> bool:
        """Return whether *model_name* resolves to a known LoRA (and mark it recently used)."""
        found = self.fuzzy_find_lora_key(model_name)
        if found is None:
            return False
        self._touch_lora(found)
        return True

    def is_lora_available(self, lora_name: str | int, timeout: float = 45, is_version: bool = False) -> bool:
        """Return whether *lora_name* is already known locally (by version or fuzzy name)."""
        if is_version:
            return self.ensure_is_version(lora_name) in self._index_version_ids
        return self.is_model_available(lora_name)

    def _touch_lora(self, lora_name: str | int, is_version: bool = False) -> None:
        """Mark the latest (or specified) version of *lora_name* as used now, and persist.

        In read-only mode the in-memory timestamp is still updated but nothing is written to disk, so
        availability checks stay pure reads.
        """
        record = self.get_model_reference_info(lora_name, is_version)
        if not record:
            return
        version_id = self.ensure_is_version(lora_name) if is_version else self.find_latest_version(record)
        if version_id is None or version_id not in record.versions:
            return
        record.versions[version_id].last_used = now_timestamp()
        if not self.read_only:
            self.save_reference_to_disk()

    def get_lora_last_use(self, lora_name: str | int, is_version: bool = False) -> datetime | None:
        """Return when the latest (or specified) version of *lora_name* was last used, or ``None``."""
        record = self.get_model_reference_info(lora_name, is_version)
        if not record:
            return None
        version_id = self.ensure_is_version(lora_name) if is_version else self.find_latest_version(record)
        if version_id is None or version_id not in record.versions:
            return None
        last_used = record.versions[version_id].last_used
        if not last_used:
            record.versions[version_id].last_used = now_timestamp()
            return datetime.now()
        return timestamp_to_datetime(last_used)

    def do_baselines_match(self, lora_name: str | int, model_details, is_version: bool = False) -> bool:
        """Return whether the LoRA's baseline is compatible with the model (currently always ``True``)."""
        return True  # FIXME: baseline matching disabled pending normalised baseline data.

    @override
    def _parse_civitai_item(self, item: dict, *, adhoc: bool = False) -> HordeLoraModelRecord | None:
        """Return a single-version :class:`HordeLoraModelRecord` for a CivitAI item, or ``None``."""
        parse_logger = logger.bind(manager="lora", adhoc=adhoc)
        if "modelId" in item:
            version_data = item
            lora_id = item["modelId"]
            if lora_id in self._index_ids:
                existing = self.model_reference[self._index_ids[lora_id]]
                lora_name = existing.orig_name
                lora_nsfw = existing.nsfw
            else:
                model_data = self._fetch_civitai_json(f"https://civitai.com/api/v1/models/{lora_id}")
                if model_data is None:
                    return None
                lora_name = model_data.get("name", "")
                lora_nsfw = model_data.get("nsfw", True)
        else:
            try:
                version_data = item.get("modelVersions", {})[0]
            except IndexError:
                version_data = {}
            lora_name = item.get("name", "")
            lora_id = item.get("id", 0)
            lora_nsfw = item.get("nsfw", True)

        version_id = self.ensure_is_version(version_data.get("id", 0))
        if version_id is None:
            return None
        triggers = normalise_a1111_triggers(version_data.get("trainedWords", []))

        primary_file = next(
            (
                f
                for f in version_data.get("files", [])
                if f.get("primary", False) and f.get("name", "").endswith(".safetensors")
            ),
            None,
        )
        if primary_file is None:
            return None

        try:
            size_mb = round(primary_file.get("sizeKB", 0) / 1024)
        except TypeError:
            size_mb = 144  # Common LoRA size; not critical when CivitAI omits it.

        lora_key = Sanitizer.sanitise_model_name(lora_name).lower().strip()
        version = LoraVersionEntry(
            filename=f"{Sanitizer.sanitise_filename(lora_name)[0:128]}_{version_data.get('id', 0)}.safetensors",
            url=primary_file.get("downloadUrl", ""),
            version_id=version_id,
            lora_key=lora_key,
            sha256=primary_file.get("hashes", {}).get("SHA256"),
            adhoc=adhoc,
            size_mb=size_mb,
            triggers=triggers,
            base_model=version_data.get("baseModel", "SD 1.5"),
            availability=version_data.get("availability", "Public"),
        )

        if adhoc and not version.sha256:
            return None
        if not version.filename or not version.url:
            return None
        if adhoc and version.size_mb > MAX_ADHOC_LORA_SIZE_MB and lora_id not in self._default_lora_ids:
            parse_logger.debug("lora.parse_rejected_size", size_mb=version.size_mb)
            return None
        if adhoc and lora_nsfw and not self.nsfw:
            parse_logger.debug("lora.parse_rejected_sfw_worker")
            return None

        record = HordeLoraModelRecord(
            name=lora_key,
            civitai_id=lora_id,
            orig_name=lora_name,
            nsfw=lora_nsfw,
            versions={version_id: version},
        )
        self._sync_top_level(record)
        return record

    def _sync_top_level(self, record: HordeLoraModelRecord) -> None:
        """Mirror the latest version's details onto the record's canonical provider-facing fields."""
        latest = self.get_latest_version(record) or next(iter(record.versions.values()), None)
        if latest is None:
            return
        record.version = latest.version_id
        record.baseline = latest.base_model
        record.trigger = list(latest.triggers)
        record.config.download = [
            DownloadRecord(
                file_name=latest.filename,
                file_url=latest.url,
                sha256sum=latest.sha256 or "FIXME",
            ),
        ]

    @override
    def _prepare_download(self, record: HordeLoraModelRecord) -> DownloadTarget | None:
        """Return the latest version's weight file as the download target."""
        version_id = self.find_latest_version(record)
        if version_id is None:
            return None
        version = record.versions[version_id]
        return DownloadTarget(
            filename=version.filename,
            url=version.url,
            size_mb=version.size_mb,
            sha256=version.sha256,
            version_key=version_id,
        )

    @override
    def _commit_download(self, record: HordeLoraModelRecord, target: DownloadTarget, *, downloaded: bool) -> None:
        """Merge *record*'s downloaded version into the reference and refresh indices."""
        with self._mutex:
            self.reload_reference_from_disk()
            new_version_id = next(iter(record.versions.keys()))
            new_version = record.versions[new_version_id]
            if downloaded:
                new_version.last_used = now_timestamp()
            record.last_checked = now_timestamp()

            existing = self.model_reference.get(record.name)
            if existing is None:
                self.model_reference[record.name] = record
                merged = record
            else:
                if new_version_id not in existing.versions:
                    existing.versions[new_version_id] = new_version
                existing.last_checked = now_timestamp()
                merged = existing

            self._sync_top_level(merged)
            self._index_ids[merged.civitai_id] = merged.name
            self._index_version_ids[new_version_id] = merged.name
            self._index_orig_names[merged.orig_name.lower()] = merged.name

    @override
    def _rebuild_indices(self) -> None:
        """Rebuild the id, version-id, and original-name indices from the reference."""
        self._index_ids = {}
        self._index_orig_names = {}
        self._index_version_ids = {}
        for record in self.model_reference.values():
            self._index_ids[record.civitai_id] = record.name
            self._index_orig_names[record.orig_name.lower()] = record.name
            for version_id in record.versions:
                self._index_version_ids[version_id] = record.name

    @override
    def _iter_cache_entries(self):
        """Yield one :class:`CacheEntry` per known LoRA version."""
        for model_key, record in self.model_reference.items():
            for version_id, version in record.versions.items():
                yield CacheEntry(
                    model_key=model_key,
                    version_key=version_id,
                    filename=version.filename,
                    size_mb=version.size_mb,
                    adhoc=version.adhoc,
                    last_used=timestamp_to_datetime(version.last_used),
                )

    @override
    def _drop_entry(self, model_key: str, version_key: str | None) -> None:
        """Remove a single LoRA version (and the LoRA if empty) from memory and indices."""
        record = self.model_reference.get(model_key)
        if record is None or version_key is None or version_key not in record.versions:
            return
        del record.versions[version_key]
        self._index_version_ids.pop(version_key, None)
        if not record.versions:
            self._index_ids.pop(record.civitai_id, None)
            self._index_orig_names.pop(record.orig_name.lower(), None)
            del self.model_reference[model_key]

    @override
    def _delete_model_entry(self, model_key: str, version_key: str | None) -> None:
        """Delete a single LoRA version's file, forget it, and persist.

        Raises:
            ReadOnlyModelManagerError: If the manager is read-only.
        """
        self._ensure_writable()
        record = self.model_reference.get(model_key)
        if record is None or version_key is None or version_key not in record.versions:
            return
        self._delete_weight_files(record.versions[version_key].filename)
        self._drop_entry(model_key, version_key)
        self.save_reference_to_disk()

    def delete_lora(self, lora_version: str) -> None:
        """Delete a specific LoRA version (and the LoRA if it becomes empty)."""
        with self._mutex:
            self.reload_reference_from_disk()
        lora_key = self.find_lora_key_by_version(lora_version)
        if lora_key is None:
            logger.warning("Could not find lora version to delete: {}", lora_version)
            return
        self._delete_model_entry(lora_key, self.ensure_is_version(lora_version))

    def _canonical_lora_key(self, requested: str | int) -> str:
        """Return the sanitised lookup key for *requested* (an id string passes through)."""
        if isinstance(requested, int):
            return str(requested)
        return Sanitizer.sanitise_model_name(requested).lower().strip()

    def ensure_lora_deleted(self, lora_name: str | int) -> None:
        """Delete every version of the LoRA matching *lora_name*, guarding against weak fuzzy matches."""
        lora_key: str | None = None
        if isinstance(lora_name, int) or (isinstance(lora_name, str) and lora_name.isdigit()):
            lora_key = self._index_ids.get(int(lora_name))
        if lora_key is None and isinstance(lora_name, str):
            canonical = self._canonical_lora_key(lora_name)
            if canonical in self.model_reference:
                lora_key = canonical
        if lora_key is None:
            candidate = self.fuzzy_find_lora_key(lora_name)
            if candidate:
                request_key = self._canonical_lora_key(lora_name)
                candidate_key = self._canonical_lora_key(candidate)
                strong_match = (
                    request_key
                    and candidate_key
                    and (request_key == candidate_key or request_key in candidate_key or candidate_key in request_key)
                )
                if strong_match:
                    lora_key = candidate
                else:
                    logger.warning("Rejecting weak fuzzy deletion for {} -> {}", lora_name, candidate)
                    return
        if not lora_key:
            return
        for version_id in list(self.model_reference[lora_key].versions.keys()):
            self.delete_lora(version_id)

    def download_default_models(self, *, nsfw: bool = True, timeout: float | None = None) -> None:
        """Start a background download of the curated default LoRAs and return immediately."""
        if not self.are_downloads_complete():
            logger.warning("Downloads already in progress, skipping download_default_models")
            return
        self.nsfw = nsfw
        self.clear_all_references()
        os.makedirs(self.model_folder_path, exist_ok=True)
        self._thread = threading.Thread(target=self._seed_default_loras, daemon=True)
        self._thread.start()
        if self._download_wait:
            self.wait_for_downloads(timeout)
        if self.is_adhoc_reset_complete():
            self._adhoc_reset_thread = threading.Thread(target=self.reset_adhoc_cache, daemon=True)
            self._adhoc_reset_thread.start()

    def clear_all_references(self) -> None:
        """Empty the in-memory reference and all indices."""
        self.model_reference = {}
        self._index_ids = {}
        self._index_version_ids = {}
        self._index_orig_names = {}

    def _seed_default_loras(self) -> None:
        """Fetch the curated default-LoRA id list and enqueue each for download."""
        default_ids = self._fetch_civitai_json(self.LORA_DEFAULTS)
        if not isinstance(default_ids, list):
            logger.bind(manager="lora").error("lora.defaults_fetch_invalid_type")
            self._default_lora_ids = []
            return
        self._default_lora_ids = default_ids
        self._enqueue_civitai_ids(default_ids)

    def _enqueue_civitai_ids(self, lora_ids: list, *, adhoc: bool = False) -> None:
        """Fetch metadata for a batch of CivitAI model ids and enqueue valid LoRAs for download."""
        if not lora_ids:
            return
        ids_query = "&ids=".join(str(lora_id) for lora_id in lora_ids)
        data = self._fetch_civitai_json(f"https://civitai.com/api/v1/models?limit=100&ids={ids_query}")
        if not data:
            logger.bind(manager="lora").warning("lora.queue_metadata_missing", requested_ids=lora_ids)
            return
        for item in data.get("items", []):
            record = self._parse_civitai_item(item, adhoc=adhoc)
            if record:
                self._enqueue_download(record, {"trigger_source": "adhoc_queue" if adhoc else "default_queue"})

    def reset_adhoc_cache(self) -> None:
        """Wait for in-flight downloads, then evict ad-hoc entries back within budget.

        Raises:
            ReadOnlyModelManagerError: If the manager is read-only.
        """
        self._ensure_writable()
        while not self.are_downloads_complete():
            if self._stop_all_threads:
                return
            time.sleep(self.THREAD_WAIT_TIME)
        self.save_reference_to_disk()
        if self.is_adhoc_cache_full():
            self._evict_adhoc_over_limit()
            self.save_reference_to_disk()

    def get_lora_metadata(self, url: str) -> HordeLoraModelRecord:
        """Return a parsed LoRA record for the first item at *url*.

        Raises:
            he.CivitAIDown: If CivitAI returned no data.
            he.ModelEmpty: If the search returned no items.
            he.ModelInvalid: If the item could not be parsed into a valid record.
        """
        data = self._fetch_civitai_json(url)
        if not data:
            raise he.CivitAIDown("CivitAI Down?")
        if "items" in data:
            if len(data["items"]) == 0:
                raise he.ModelEmpty("Lora appears empty")
            record = self._parse_civitai_item(data["items"][0], adhoc=True)
        else:
            record = self._parse_civitai_item(data, adhoc=True)
        if not record:
            raise he.ModelInvalid("Lora is invalid")
        return record

    @logfire.instrument("lora.fetch_adhoc", extract_args=True)
    def fetch_adhoc_lora(
        self,
        lora_name: str | int,
        timeout: int | None = 45,
        is_version: bool = False,
        job_context: dict | None = None,
    ) -> str | None:
        """Ensure a LoRA is available, downloading it from CivitAI on demand.

        If *timeout* is set, blocks until the download completes and returns the reference key;
        otherwise starts the download and returns ``None`` immediately.

        Raises:
            ReadOnlyModelManagerError: If the manager is read-only.
        """
        self._ensure_writable()
        if is_version and not (isinstance(lora_name, int) or str(lora_name).isdigit()):
            return None
        if isinstance(lora_name, int) or str(lora_name).isdigit():
            # CivitAI returns 500 (not 404) for ids beyond its id space; reject impossible ids early.
            if int(lora_name) >= 2**32:
                return None
            endpoint = "model-versions" if is_version else "models"
            url = f"https://civitai.com/api/v1/{endpoint}/{lora_name}"
        else:
            url = f"{self.LORA_API}&nsfw={str(self.nsfw).lower()}&query={lora_name}"

        try:
            record = self.get_lora_metadata(url)
        except he.CivitAIDown as civitai_down:
            if isinstance(lora_name, str) and lora_name in self.model_reference:
                self._touch_lora(lora_name)
                logger.warning("CivitAI appears down; using cached info: {}", civitai_down)
                return lora_name
            return None
        except (he.ModelEmpty, he.ModelInvalid) as invalid_model:
            logger.info("Adhoc lora '{}' rejected: {}", lora_name, invalid_model)
            return None

        cached_key = self._resolve_metadata_mismatch(lora_name, record, is_version)
        if cached_key is not _NO_CACHE_FALLBACK:
            return cached_key

        existing_key = self.fuzzy_find_lora_key(record.civitai_id)
        if existing_key:
            if is_version and self.ensure_is_version(lora_name) in self.model_reference[existing_key].versions:
                self._touch_lora(lora_name, True)
                return existing_key
            if not is_version and self.find_latest_version(
                self.get_model_reference_info(existing_key),
            ) == self.find_latest_version(record):
                self._touch_lora(lora_name)
                return existing_key

        context = job_context or {}
        context.setdefault("trigger_source", "adhoc_generation")
        self._enqueue_download(record, context)

        if timeout is None:
            return None
        time.sleep(self.THREAD_WAIT_TIME)
        self.wait_for_downloads(timeout)
        version = lora_name if is_version else self.find_latest_version(record)
        if version is None:
            return None
        self._touch_lora(version, True)
        return record.name

    def _resolve_metadata_mismatch(
        self,
        lora_name: str | int,
        record: HordeLoraModelRecord,
        is_version: bool,
    ):
        """Return a cached key (or ``None``) when CivitAI's metadata does not match the request.

        Returns the sentinel :data:`_NO_CACHE_FALLBACK` when the metadata *does* match and the caller
        should proceed to download.
        """
        if self._metadata_matches_request(lora_name, record, is_version):
            return _NO_CACHE_FALLBACK
        logger.warning(
            "CivitAI returned metadata not matching request {} (version={}) -> id={} name={}",
            lora_name,
            is_version,
            record.civitai_id,
            record.name,
        )
        if is_version:
            cached_key = self.find_lora_key_by_version(lora_name)
            if cached_key:
                self._touch_lora(lora_name, True)
                return cached_key
            return None
        cached_key = self.fuzzy_find_lora_key(lora_name)
        if not cached_key:
            canonical = self._canonical_lora_key(lora_name)
            if canonical and canonical != lora_name:
                cached_key = self.fuzzy_find_lora_key(canonical)
        if cached_key:
            self._touch_lora(cached_key)
            return cached_key
        return None

    def _metadata_matches_request(
        self,
        requested: str | int,
        record: HordeLoraModelRecord,
        is_version: bool = False,
    ) -> bool:
        """Return whether CivitAI's returned *record* actually corresponds to *requested*."""
        if is_version:
            version_key = self.ensure_is_version(requested)
            return version_key is not None and version_key in record.versions
        if isinstance(requested, int) or (isinstance(requested, str) and requested.isdigit()):
            return str(record.civitai_id) == str(requested)

        requested_sanitized = self._canonical_lora_key(requested)
        if not requested_sanitized:
            return False
        candidate_names = {record.name.lower()}
        if record.orig_name:
            candidate_names.add(Sanitizer.sanitise_model_name(record.orig_name).lower())
        for version in record.versions.values():
            candidate_names.add(version.lora_key.lower())
        candidate_names = {name for name in candidate_names if name}
        if any(
            requested_sanitized == candidate or requested_sanitized in candidate or candidate in requested_sanitized
            for candidate in candidate_names
        ):
            return True
        best_ratio = max((fuzz.ratio(requested_sanitized, candidate) for candidate in candidate_names), default=0)
        return best_ratio >= self.FUZZ_THRESHOLD


_NO_CACHE_FALLBACK = object()
"""Sentinel: metadata matched the request, so no cache fallback applies and a download should proceed."""
