import os
import threading
from abc import ABC
from collections.abc import Callable, Iterable
from functools import lru_cache
from pathlib import Path
from typing import Any, cast
from urllib import parse

import psutil
from horde_model_reference import (
    MODEL_REFERENCE_CATEGORY,
    category_folder,
    component_relative_path,
    download_engine,
    file_paths_for,
    get_category_descriptor,
    horde_model_reference_paths,
    is_present,
)
from horde_model_reference.model_reference_manager import ModelReferenceManager
from horde_model_reference.model_reference_records import GenericModelRecord
from horde_model_reference.source_consts import SourceSelector
from loguru import logger

from hordelib.beta_models import beta_source_for
from hordelib.config_path import get_hordelib_path
from hordelib.consts import CIVITAI_API_PATH
from hordelib.settings import UserSettings

_ANON_API_KEY = "0000000000"


def _resolve_horde_api_key() -> str | None:
    """Return the worker's horde API key from the environment, or None for anonymous/standalone use.

    The key gates the content-addressed R2 mirror in the download engine. The anonymous key cannot be trusted,
    and a standalone hordelib user without a key simply downloads from each model's origin host. The worker
    exposes its configured key as ``AIHORDE_API_KEY`` (inherited by the download subprocess).
    """
    key = os.environ.get("AIHORDE_API_KEY") or os.environ.get("AI_HORDE_API_KEY")
    if not key or key == _ANON_API_KEY:
        return None
    return key


@lru_cache(maxsize=1)
def _resolve_r2_gateway_url() -> str | None:
    """Return the gated-R2 gateway base URL from horde_model_reference settings (process-static, cached).

    None when no gateway is configured, in which case the engine downloads only from origin hosts. An older
    horde_model_reference without the R2 settings block degrades to the same origin-only behaviour rather than
    breaking the download path, so the worker keeps working ahead of a coordinated hmr release.

    The gateway URL is owned by the official deployment (the horde_model_reference default); it is deliberately
    not a per-worker config knob. A sophisticated operator can still override it via the
    ``HORDE_MODEL_REFERENCE_R2__GATEWAY_URL`` environment variable, in which case the safety guard below applies:
    because the worker's API key is sent to the gateway, a plaintext (non-``https``) gateway is refused outright
    (the mirror is disabled and origins are used) so the key cannot travel in the clear.
    """
    try:
        from horde_model_reference import HordeModelReferenceSettings
        from horde_model_reference.download_engine import gateway_accepts_key

        gateway_url = HordeModelReferenceSettings().r2.gateway_url
    except (AttributeError, ImportError):
        # Older horde_model_reference without the R2 settings block or the key-safety guard: origin-only.
        return None

    if not gateway_url:
        return None
    if not gateway_accepts_key(gateway_url):
        logger.warning(
            "Configured R2 gateway is not https; refusing to send the API key over plaintext and disabling the "
            "mirror (downloads use origin hosts): gateway={}",
            gateway_url,
        )
        return None
    # Normal path logs at DEBUG only: the gateway is a deployment-owned technical detail, not something a typical
    # operator configures or needs to reason about, so it stays out of the default console.
    logger.debug("Gated R2 mirror enabled; the API key will be sent to gateway={}", parse.urlparse(gateway_url).netloc)
    return gateway_url


class BaseModelManager[RecordT: GenericModelRecord | dict[str, Any]](ABC):
    """Abstract base for the per-category model managers.

    ``RecordT`` is the model reference record type a manager operates on. Managers backed by
    ``horde_model_reference`` use pydantic ``GenericModelRecord`` (or a subtype); the LoRA/TI
    managers still operate on raw dicts until upstream ships records for those categories.
    """

    model_folder_path: Path
    """The path to the directory to store this model type."""
    _weights_root: Path
    """The model-weights root under which the category folders live (the parent of model_folder_path)."""
    model_reference: dict[str, RecordT]
    """Model reference data, keyed by horde model name."""

    available_models: list[str]
    """The models available for immediate use."""

    tainted_models: list
    """Models which seem to be corrupted and should be deleted when the correct replacement is downloaded."""

    _disk_write_mutex = threading.Lock()

    def __init__(
        self,
        *,
        model_category: MODEL_REFERENCE_CATEGORY,
        civitai_api_token: str | None = None,
        download_reference: bool = False,
        models_db_path: str | Path | None = None,
        **kwargs,
    ):
        """Create a new instance of this model manager.

        Args:
            model_category (MODEL_REFERENCE_CATEGORY): The canonical category this manager handles. Determines
                the on-disk folder and reference data via horde_model_reference.
            civitai_api_token (str | None): Optional API token for Civitai. Required to access certain models
            and improves rate limits. Defaults to None.
            download_reference (bool): Whether to download the model reference on init. Defaults to False.
            models_db_path (str | Path | None): Path to a specific model database JSON file. If None,
                resolved via horde_model_reference paths.
            **kwargs: May include ``multiprocessing_lock`` and other backend-specific arguments.

        Raises:
            ValueError: If *model_category* has no on-disk weights folder, or no database path resolves.
        """
        if len(kwargs) > 0:
            logger.debug("Unused kwargs: type={}, kwargs={}", type(self), kwargs)

        self._model_category = model_category
        self.model_reference = {}
        self.available_models: list[str] = []
        self.tainted_models: list[str] = []

        self.models_db_name = str(model_category)
        self.download_reference = download_reference
        self._civitai_api_token = parse.quote_plus(civitai_api_token) if civitai_api_token else None

        # Set the on-disk folder where model weight files are stored
        folder_name = category_folder(model_category)
        if folder_name is None:
            raise ValueError(f"Model category {model_category} has no on-disk weights folder")
        self._weights_root = UserSettings.get_model_directory()
        self.model_folder_path = self._weights_root / folder_name

        if models_db_path is None:
            if not get_category_descriptor(model_category).managed_download_elsewhere:
                # Canonical categories are served by horde_model_reference.
                models_db_path = horde_model_reference_paths.get_model_reference_filename(
                    model_category,
                    base_path=horde_model_reference_paths.legacy_path,
                )
            else:
                # LoRA/TI are managed by the CivitAI ad-hoc engine and persist their own cache.
                models_db_path = get_hordelib_path() / "model_database" / f"{self.models_db_name}.json"

        if models_db_path is None:
            raise ValueError(f"Model database path not found for {model_category}")

        logger.debug("Model database path: path={}", models_db_path)
        self.models_db_path = Path(models_db_path)

        self.load_model_database()

    def progress(self, desc="done", current=0, total=0):
        # TODO
        return

    def load_model_database(self) -> None:
        if self.model_reference:
            logger.info("Model reference was already loaded.")
            logger.info("Reloading model reference...")

        if self.download_reference:
            raise NotImplementedError("Downloading model databases is no longer supported within hordelib.")

        if get_category_descriptor(self._model_category).managed_download_elsewhere:
            # Categories managed outside horde_model_reference (LoRA, TI) must override this method.
            raise NotImplementedError(
                f"Model category {self._model_category} is managed outside horde_model_reference and "
                f"{type(self).__name__} does not override load_model_database().",
            )

        if not ModelReferenceManager.has_instance():
            raise RuntimeError(
                "ModelReferenceManager has not been initialised. Call SharedModelManager.load_model_managers() "
                "(or otherwise construct ModelReferenceManager) before creating model managers.",
            )

        ref_manager = ModelReferenceManager.get_instance()
        pydantic_records = ref_manager.get_model_reference_or_none(
            self._model_category,
            source=self._reference_source(ref_manager),
        )
        if pydantic_records is None:
            raise RuntimeError(
                f"horde_model_reference returned no data for category {self._model_category}. "
                "The model reference may have failed to download; cannot continue with an empty reference.",
            )

        # horde_model_reference guarantees the record subtype per category (e.g. ImageGenerationModelRecord
        # for image_generation), which is what each manager declares as its RecordT.
        self.model_reference = cast(dict[str, RecordT], dict(pydantic_records))
        self.available_models = [
            model_name for model_name in self.model_reference if self.is_model_available(model_name)
        ]
        logger.info(
            "Loaded {} available models for {} via horde_model_reference.",
            len(self.available_models),
            self.models_db_name,
        )

    def _reference_source(self, ref_manager: ModelReferenceManager) -> SourceSelector:
        """Return the source selector this manager loads its reference from.

        Defaults to the beta-aware selector (pending/beta records override canonical for opted-in
        categories, canonical only otherwise). A manager whose records are served exclusively by a
        registered provider rather than canonical data overrides this to name that provider's source.
        """
        return beta_source_for(self._model_category, ref_manager)

    def download_model_reference(self) -> dict:
        raise NotImplementedError("Downloading model databases is no longer supported within hordelib.")

    def get_free_ram_mb(self) -> int:
        """Returns the amount of free RAM in MB rounded down to the nearest integer.

        Returns:
            int: The amount of free RAM in MB
        """
        return int(psutil.virtual_memory().available / (1024 * 1024))

    def get_model_reference_info(self, model_name: str) -> RecordT | None:
        """Return the model reference entry for a given model name, or ``None`` if not found."""
        return self.model_reference.get(model_name, None)

    def _get_generic_record(self, model_name: str) -> GenericModelRecord | None:
        """Return the pydantic record for a model, or ``None`` if absent or not a pydantic record."""
        record = self.model_reference.get(model_name)
        return record if isinstance(record, GenericModelRecord) else None

    def get_model_filenames(self, model_name: str) -> list[dict]:  # TODO: Convert dict into class
        """Return the filenames of the model for a given model name.

        Args:
            model_name (str): The name of the model to get the filename for.

        Returns:
            list[dict]: Each has at least one value "file_path" with the Path to the filename
            Optionally it also has a key "file_type" with the type of file this is for the model
        Raises:
            ValueError: If the model name is not in the model reference.
        """
        record = self._get_generic_record(model_name)
        if record is None:
            raise ValueError(f"Model {model_name} not found in model reference")

        model_files: list[dict] = []
        for download_entry in record.config.download:
            file_name = download_entry.file_name
            if file_name.endswith((".ckpt", ".safetensors", ".pt", ".pth", ".bin")):
                # Multi-file components (vae/text_encoders) are redirected to their sibling
                # ComfyUI folder; everything else stays in this manager's folder.
                path_entry: dict[str, Any] = {
                    "file_path": component_relative_path(file_name, download_entry.file_purpose),
                }
                if download_entry.file_purpose:
                    path_entry["file_type"] = download_entry.file_purpose
                # horde_model_reference uses the literal "FIXME" as its placeholder for unknown checksums
                if download_entry.sha256sum and download_entry.sha256sum != "FIXME":
                    path_entry["sha256sum"] = download_entry.sha256sum
                model_files.append(path_entry)
        if len(model_files) == 0:
            # If no weight files in download entries, use the primary file_name as file_path
            if record.primary_download_url:
                primary_file = record.config.download[0].file_name
                model_files.append({"file_path": Path(primary_file)})
        if len(model_files) == 0:
            raise ValueError(f"Model {model_name} does not have a valid file entry")
        return model_files

    def get_model_config_files(self, model_name: str) -> list[dict]:
        """Return the config files for a given model name.

        Args:
            model_name (str): The name of the model to get the config files for.

        Returns:
            list[dict]: The config files for the model.
        """
        record = self._get_generic_record(model_name)
        if record is None:
            return []

        # Convert DownloadRecord entries to dict form for backward compatibility
        return [
            {
                "file_name": dl.file_name,
                "file_url": dl.file_url,
                "sha256sum": dl.sha256sum,
                "file_purpose": dl.file_purpose,
            }
            for dl in record.config.download
        ]

    def get_model_download(self, model_name: str) -> list[dict]:
        """Return the download config for a given model name.

        Args:
            model_name (str): The name of the model to get the download config for.

        Returns:
            list[dict]: The download config for the model.
        """
        # The download entries and the config-file entries are one and the same in
        # horde_model_reference records.
        return self.get_model_config_files(model_name)

    def get_available_models(self) -> list[str]:
        """Return the available (downloaded and verified) models."""
        return self.available_models

    def get_available_models_by_types(self, model_types: Iterable[str] | None = None) -> list[str]:
        """Return the available (downloaded and verified) models.

        Args:
            model_types (Iterable[str] | None, optional): Legacy parameter, now ignored:
            horde_model_reference records have no "type" field and every record in a
            manager's reference is of that manager's category. Defaults to None.

        Returns:
            list[str]: The available models.
        """
        if model_types:
            logger.debug("get_available_models_by_types: model_types is deprecated and ignored: {}", model_types)
        return [model for model in self.model_reference if self.is_model_available(model)]

    def count_available_models_by_types(self, model_types: Iterable[str] | None = None) -> int:
        """Return the number of available (downloaded and verified) models of a given type.

        Args:
            model_types (Iterable[str] | None, optional): The type of model to return. See the model
            reference for valid values. Defaults to None.

        Returns:
            int: The number of available models of the given type.
        """
        return len(self.get_available_models_by_types(model_types))

    def taint_model(self, model_name: str) -> None:
        """Mark a model as not valid by removing it from available_models"""
        if model_name in self.available_models:
            self.available_models.remove(model_name)
            self.tainted_models.append(model_name)

    def taint_models(self, models: list[str]) -> None:
        """Mark a list of models as not valid by removing them from available_models"""
        for model in models:
            self.taint_model(model)

    def validate_model(self, model_name: str, skip_checksum: bool = False) -> bool | None:
        """Check the if the model file is on disk and, optionally, also if the checksum is correct.

        Args:
            model_name (str): The name of the model to check.
            skip_checksum (bool, optional): Defaults to False.

        Returns:
            bool | None: `True` if the model is valid, `False` if not, `None` if the model is not on disk.
        """
        model_files = self.get_model_filenames(model_name)
        logger.debug("Validating model: name={}, files={}", model_name, model_files)
        for file_entry in model_files:
            if not self.is_file_available(file_entry["file_path"]):
                return None
            if not skip_checksum and not self.validate_file(file_entry):
                logger.warning("File has different contents than expected: file_path={}", file_entry["file_path"])
                try:
                    # The file must have been considered valid once, or we wouldn't have renamed
                    # it from the ".part" download. Likely there is an update, or a model database hash problem
                    logger.warning(
                        "Likely updated, will attempt to re-download: file_path={}",
                        file_entry["file_path"],
                    )
                    self.taint_model(model_name)
                except OSError as e:
                    logger.error("Unable to delete file: file_path={}, error={}", file_entry["file_path"], e)
                    logger.error("Please delete file if this error persists: file_path={}", file_entry["file_path"])
                return False

        # FIXME: The below commented lines are already done in L267. Is this still needed?
        # file_details = self.get_model_config_files(model_name)
        # for file_detail in file_details:
        #     if ".yaml" in file_detail["path"] or ".json" in file_detail["path"]:
        #         continue
        #     if not self.is_file_available(file_detail["path"]):
        #         logger.debug(f"File {file_detail['path']} not found")
        #         return None

        if model_name not in self.available_models:
            self.available_models.append(model_name)
        return True

    @staticmethod
    def get_file_md5sum_hash(file_name: str | Path) -> str | None:
        """Return the md5 of *file_name* (or None if missing), via the hmr engine's sidecar cache."""
        return download_engine.md5_of(Path(file_name))

    @staticmethod
    def get_file_sha256_hash(file_name: str | Path) -> str:
        """Return the sha256 of *file_name*, via the hmr engine's mtime-keyed sidecar cache.

        Raises:
            FileNotFoundError: If *file_name* is not an existing file.
        """
        return download_engine.sha256_of(Path(file_name))

    def validate_file(self, file_details: dict) -> bool:
        # FIXME This isn't enough or isn't being called at the right times
        """
        :param file_details: A single file from the model's files list
        Checks if the file exists and if the checksum is correct
        Returns True if the file is valid, False otherwise
        """
        # TODO: It's all a bit ugly now, trying to handle both get_model_filenames()
        # as well as direct image reference dicts
        # But I couldn't figure out how to handle multiple files per model,
        # where I want to place them in other locations than in compvis.
        if "file_path" in file_details:  # This means it's an dict that was processed through get_model_filenames()
            full_path = file_details["file_path"]
            if isinstance(full_path, Path) and not full_path.is_absolute():
                full_path = f"{self.model_folder_path}/{file_details['file_path']}"
        else:
            full_path = f"{self.model_folder_path}/{file_details['path']}"
        # Default to sha256 hashes
        if "sha256sum" in file_details:
            logger.debug("Getting sha256sum: full_path={}", full_path)
            sha256_file_hash = self.get_file_sha256_hash(full_path).lower()
            expected_hash = file_details["sha256sum"].lower()
            logger.debug("sha256sum: hash={}", sha256_file_hash)
            logger.debug("Expected: hash={}", expected_hash)
            return expected_hash == sha256_file_hash

        # If sha256 is not available, fall back to md5
        if "md5sum" in file_details:
            logger.debug("Getting md5sum: full_path={}", full_path)
            md5_file_hash = self.get_file_md5sum_hash(full_path)
            logger.debug("md5sum: hash={}", md5_file_hash)
            logger.debug("Expected: hash={}", file_details["md5sum"])
            return file_details["md5sum"] == md5_file_hash

        # If no hashes available, return True for now
        # THIS IS A SECURITY RISK, EVENTUALLY WE SHOULD RETURN FALSE
        # But currently not all models specify hashes
        # XXX this warning preexists me (@tazlin), probably should look into it

        logger.debug(
            "Model doesn't have a checksum, skipping validation: path={}",
            file_details.get("file_path", file_details.get("path")),
        )

        return True

    def is_file_available(self, file_path: str | Path) -> bool:
        """
        :param file_path: Path of the model's file. File is from the model's files list.
        Checks if the file exists
        Returns True if the file exists, False otherwise
        """
        parsed_full_path = Path(f"{self.model_folder_path}/{file_path}")
        is_custom_model = False
        if isinstance(file_path, str):
            check_path = Path(file_path)
            if check_path.is_absolute():
                parsed_full_path = Path(file_path)
                is_custom_model = True
        if isinstance(file_path, Path) and file_path.is_absolute():
            parsed_full_path = Path(file_path)
            is_custom_model = True
        if parsed_full_path.suffix == ".part":
            logger.debug("File is a partial download, skipping: file_path={}", file_path)
            return False
        sha_file_path = Path(f"{parsed_full_path.parent}/{parsed_full_path.stem}.sha256")
        if parsed_full_path.exists() and not sha_file_path.exists() and not is_custom_model:
            self.get_file_sha256_hash(parsed_full_path)
        return parsed_full_path.exists() and (sha_file_path.exists() or is_custom_model)

    def download_file(
        self,
        url: str,
        filename: str,
        callback: Callable[[int, int], None] | None = None,
    ) -> bool:
        """Download *url* to *filename* (relative to this manager's folder) via the hmr engine.

        A thin wrapper over :func:`horde_model_reference.download_engine.download_file`, retained for
        callers that fetch a single explicit file. The model-level entry point is :meth:`download_model`.
        """
        destination = self.model_folder_path / filename
        outcome = download_engine.download_file(url, destination, progress_callback=callback)
        return outcome.success

    def _civitai_token_for(self, record: GenericModelRecord) -> str | None:
        """Return the CivitAI token only when *record*'s URLs are CivitAI-hosted, else None.

        The hmr engine appends the token to every URL it fetches; restricting it to CivitAI hosts
        preserves the prior behaviour of never leaking the token to other download hosts.
        """
        if not self._civitai_api_token:
            return None
        if any(self.is_model_url_from_civitai(download.file_url) for download in record.config.download):
            return self._civitai_api_token
        return None

    def _delete_on_disk_files(self, record: GenericModelRecord) -> None:
        """Remove *record*'s on-disk weight files and their checksum/partial sidecars.

        Used when a model is tainted: the hmr engine skips files already present, so a corrupt or stale
        copy must be cleared before it will re-fetch.
        """
        for file_path in file_paths_for(record, self._weights_root):
            companions = (
                file_path,
                file_path.with_suffix(".sha256"),
                file_path.with_suffix(".md5"),
                Path(f"{file_path}.part"),
            )
            for companion in companions:
                try:
                    companion.unlink(missing_ok=True)
                except OSError as delete_error:
                    logger.warning("Could not delete tainted file: path={}, error={}", companion, delete_error)

    def download_model(
        self,
        model_name: str,
        *,
        callback: Callable[[int, int], None] | None = None,
        connections: int = 1,
    ) -> bool | None:
        """Ensure *model_name*'s declared files are present, fetching any that are missing.

        Downloading and checksum verification are delegated to the horde_model_reference engine; this
        method handles discovery concerns (already-available short-circuit, tainted re-download) and
        post-download validation.

        Args:
            model_name: The reference key of the model to download.
            callback: Optional ``(downloaded_bytes, total_bytes)`` progress callback, per file.
            connections: Maximum concurrent connections per large file, forwarded to the engine. 1 keeps the
                single-stream path; a higher value lets the engine segment a large file across that many
                ranged connections to raise single-file throughput (the caller's configured value).

        Returns:
            True if the model is present and validated, False on download/validation failure, and None
            when the category is managed elsewhere (LoRA/TI) or the model has no reference record.
        """
        if get_category_descriptor(self._model_category).managed_download_elsewhere:
            # LoRA/TI are fetched on demand by the CivitAI ad-hoc engine, not the hmr download engine.
            logger.debug("download_model is a no-op for externally-managed category: {}", self._model_category)
            return None

        is_model_tainted = model_name in self.tainted_models
        if not is_model_tainted and model_name in self.available_models:
            logger.debug("Model is already available: model={}", model_name)
            return True

        record = self._get_generic_record(model_name)
        if record is None:
            logger.error("Cannot download model without a reference record: model={}", model_name)
            return False

        if not is_model_tainted and self.is_model_available(model_name):
            logger.debug("Model is already downloaded: model={}", model_name)
            return True

        if is_model_tainted:
            logger.debug("Model is tainted; clearing on-disk files before re-download: model={}", model_name)
            self._delete_on_disk_files(record)

        download_succeeded = download_engine.download_record_files(
            record,
            self._weights_root,
            progress_callback=callback,
            auth_query_token=self._civitai_token_for(record),
            gateway_base_url=_resolve_r2_gateway_url(),
            apikey=_resolve_horde_api_key(),
            connections=connections,
        )
        if not download_succeeded:
            return False
        return self.validate_model(model_name)

    def download_all_models(
        self,
        callback: Callable[[int, int], None] | None = None,
    ) -> bool:
        """Download every not-yet-available model in this manager's reference (best effort)."""
        if get_category_descriptor(self._model_category).managed_download_elsewhere:
            # The CivitAI ad-hoc engine fetches LoRA/TI on demand; there is no "download everything".
            return True
        for model in self.model_reference:
            if not self.is_model_available(model):
                logger.info("Downloading model: model={}", model)
                self.download_model(model, callback=callback)
            else:
                self.validate_model(model)
        return True

    def _extra_weights_roots(self) -> list[Path]:
        """Additional weights roots to search for already-present files, from the environment.

        Mirrors the worker's ``AIWORKER_EXTRA_MODEL_DIRECTORIES`` (an ``os.pathsep``-separated list of
        weights-root directories) so a model spread across disks is judged present here exactly as the
        worker's download planner judges it.
        """
        raw = os.environ.get("AIWORKER_EXTRA_MODEL_DIRECTORIES", "")
        return [Path(entry) for entry in raw.split(os.pathsep) if entry.strip()]

    def is_model_available(self, model_name: str) -> bool:
        """Return whether *model_name*'s declared files all exist on disk (existence-only).

        Presence is delegated to horde_model_reference's canonical on-disk layout, so every consumer
        (the worker's download plan, the TUI picker, this manager) answers "is it on disk?" the same way
        and a model placed on disk is never reported as needing download. Integrity is a separate concern:
        a present-but-corrupt file counts as available here and is caught by ``validate_model`` (or at load).
        """
        if model_name not in self.model_reference:
            return False

        if model_name in self.tainted_models:
            return False

        record = self._get_generic_record(model_name)
        if record is None:
            return False

        return is_present(record, self._weights_root, extra_roots=self._extra_weights_roots())

    def is_model_url_from_civitai(self, url: str) -> bool:
        return CIVITAI_API_PATH in url
