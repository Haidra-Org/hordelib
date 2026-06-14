import hashlib
import os
import shutil
import threading
import time
import zipfile
from abc import ABC
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, cast
from urllib import parse
from uuid import uuid4

import git
import psutil
import requests
from horde_model_reference import (
    MODEL_REFERENCE_CATEGORY,
    category_folder,
    component_relative_path,
    get_category_descriptor,
    horde_model_reference_paths,
)
from horde_model_reference.model_reference_manager import ModelReferenceManager
from horde_model_reference.model_reference_records import GenericModelRecord
from loguru import logger
from tqdm import tqdm

from hordelib.beta_models import beta_source_for
from hordelib.config_path import get_hordelib_path
from hordelib.consts import CIVITAI_API_PATH
from hordelib.settings import UserSettings


class BaseModelManager[RecordT: GenericModelRecord | dict[str, Any]](ABC):
    """Abstract base for the per-category model managers.

    ``RecordT`` is the model reference record type a manager operates on. Managers backed by
    ``horde_model_reference`` use pydantic ``GenericModelRecord`` (or a subtype); the LoRA/TI
    managers still operate on raw dicts until upstream ships records for those categories.
    """

    model_folder_path: Path
    """The path to the directory to store this model type."""
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
        self.model_folder_path = UserSettings.get_model_directory() / folder_name

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
            source=beta_source_for(self._model_category, ref_manager),
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
        # XXX There *must* be a library for this.
        # Bail out if the source file doesn't exist

        file_name = Path(file_name)
        if not file_name.exists() or not file_name.is_file():
            return None

        # Check if we have a cached md5 hash for the source file
        # and use that unless our source file is newer than our hash
        md5_file = Path(f"{file_name.parent / file_name.stem}.md5")
        source_timestamp = file_name.stat().st_mtime
        hash_timestamp = md5_file.stat().st_mtime if os.path.isfile(md5_file) else 0
        if hash_timestamp > source_timestamp:
            # Use our cached hash
            with open(md5_file) as handle:
                return handle.read().split()[0]

        # Calculate the hash of the source file
        with open(file_name, "rb") as file_to_check:
            file_hash = hashlib.md5()
            while True:
                chunk = file_to_check.read(2**20)  # Changed just because it broke pylint
                if not chunk:
                    break
                file_hash.update(chunk)
        md5_hash = file_hash.hexdigest()

        # Cache this md5 hash we just calculated. Use md5sum format files
        # so we can also use OS tools to manipulate these md5 files
        try:
            with open(md5_file, "w") as handle:
                handle.write(f"{md5_hash} *{file_name.name}")
        except (OSError, PermissionError):
            logger.debug("Could not write to md5sum file, ignoring")

        return md5_hash

    @staticmethod
    def get_file_sha256_hash(file_name):
        # XXX There *must* be a library for this.
        if not os.path.isfile(file_name):
            raise FileNotFoundError(f"No file {file_name}")

        # Check if we have a cached sha256 hash for the source file
        # and use that unless our source file is newer than our hash
        sha256_file = f"{os.path.splitext(file_name)[0]}.sha256"
        source_timestamp = os.path.getmtime(file_name)
        hash_timestamp = os.path.getmtime(sha256_file) if os.path.isfile(sha256_file) else 0
        if hash_timestamp > source_timestamp:
            # Use our cached hash
            with open(sha256_file) as handle:
                return handle.read().split()[0]
        logger.info("Calculating sha256sum. This may take a while: file_name={}", file_name)
        # Calculate the hash of the source file
        with open(file_name, "rb") as file_to_check:
            file_hash = hashlib.sha256()
            while True:
                chunk = file_to_check.read(2**20)
                if not chunk:
                    break
                file_hash.update(chunk)
        sha256_hash = file_hash.hexdigest()

        # Cache this sha256 hash we just calculated. Use sha256sum format files
        # so we can also use OS tools to manipulate these md5 files
        try:
            with open(sha256_file, "w") as handle:
                handle.write(f"{sha256_hash} *{os.path.basename(sha256_file)}")
        except (OSError, PermissionError):
            logger.debug("Could not write to sha256sum file, ignoring")

        return sha256_hash

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
        """
        :param url: URL of the file to download. URL is from the model's download list.
        :param file_path: Path of the model's file. File is from the model's files list.
        Downloads a file
        """
        final_pathname = f"{self.model_folder_path}/{filename}"
        partial_pathname = f"{self.model_folder_path}/{filename}.part"
        filename = os.path.basename(filename)
        os.makedirs(os.path.dirname(final_pathname), exist_ok=True)

        # Number of download attempts this time through
        retries = 5

        # Determine the remote file size
        headreq = requests.head(url, allow_redirects=True)
        if headreq.ok:
            remote_file_size = int(headreq.headers.get("Content-length", 0))
        else:
            remote_file_size = 0

        while retries:
            if os.path.exists(partial_pathname):
                # If file exists, find the size and append to it
                partial_size = os.path.getsize(partial_pathname)
                logger.info("Resuming download of file: filename={}", filename)
            else:
                # If file doesn't exist, start from beginning
                logger.info("Starting download of file: filename={}", filename)
                partial_size = 0

            # Add the 'Range' header to start downloading from where we left off
            headers = {}
            if partial_size:
                headers = {"Range": f"bytes={partial_size}-"}

            try:
                response = requests.get(url, stream=True, headers=headers, allow_redirects=True, timeout=20)

                # If the response was 416 (invalid range) check if we already downloaded all the data?
                if response.status_code == 416:
                    response = requests.get(url, stream=True, allow_redirects=True)
                    remote_file_size = int(response.headers.get("Content-length", 0))
                    if partial_size == remote_file_size:
                        # Successful download, swap the files
                        self.progress()
                        logger.info("Successfully downloaded the file: filename={}", filename)
                        if os.path.exists(final_pathname):
                            os.remove(final_pathname)
                        # Move the downloaded data into it's final location
                        os.rename(partial_pathname, final_pathname)
                        return True

                if response.ok:
                    remote_file_size = int(response.headers.get("Content-length", 0))

                # If we requested a resumed download but the server didnt respond 206, that's a problem,
                # the server didn't support resuming downloads
                if partial_size and response.status_code != 206:
                    if partial_size == remote_file_size:
                        pass  # already downloaded
                    else:
                        logger.warning(
                            f"Server did not support resuming download, "
                            f"restarting download {response.status_code}: {partial_size} != {remote_file_size}",
                        )
                        # try again without resuming, i.e. delete the partial download
                        if os.path.exists(final_pathname):
                            os.remove(final_pathname)
                        continue

                # Handle non-2XX status codes
                if not response.ok:
                    response.raise_for_status()

                # Write the content to file in chunks
                with (
                    open(partial_pathname, "ab") as f,
                    tqdm(
                        # all optional kwargs
                        unit="B",
                        initial=partial_size,
                        unit_scale=True,
                        unit_divisor=1024,
                        miniters=1,
                        desc=filename,
                        total=remote_file_size + partial_size,
                        # disable=UserSettings.download_progress_callback is not None,
                    ) as pbar,
                ):
                    downloaded = partial_size
                    for chunk in response.iter_content(chunk_size=1024 * 1024 * 16):
                        response.raise_for_status()
                        if chunk:
                            downloaded += len(chunk)
                            f.write(chunk)
                            pbar.update(len(chunk))
                            self.progress(filename, downloaded, remote_file_size + partial_size)
                            if callback:
                                callback(downloaded, remote_file_size + partial_size)
                # Successful download, swap the files
                logger.info("Successfully downloaded the file: filename={}", filename)
                self.progress()
                if os.path.exists(final_pathname):
                    os.remove(final_pathname)
                # Move the downloaded data into it's final location
                os.rename(partial_pathname, final_pathname)
                return True

            except requests.RequestException:
                logger.info("Download of file failed: filename={}", filename)
                retries -= 1
                if retries:
                    logger.info("Attempting download of file again")
                    time.sleep(2)
                else:
                    self.progress()
                    return False
        return False

    def download_model(
        self,
        model_name: str,
        *,
        callback: Callable[[int, int], None] | None = None,
    ) -> bool | None:
        """
        :param model_name: Name of the model
        Checks if the model is available, downloads the model if it is not available.
        After download, validates the model.
        Returns True if the model is available, False otherwise.

        Supported download types:
        - http(s) (url)
        - git (repo url)

        Other:
        - write content to file
        - symlink file
        - delete file
        - unzip file
        """
        # XXX this function is wacky in its premise and needs to be reworked
        is_model_tainted = model_name in self.tainted_models
        if not is_model_tainted and model_name in self.available_models:
            logger.debug("Model is already available: model={}", model_name)
            return True
        download = self.get_model_download(model_name)
        files = self.get_model_config_files(model_name)
        if self.is_model_available(model_name):
            logger.debug("Model is already downloaded: model={}", model_name)
            return True
        for i in range(len(download)):
            # Resolve the on-disk file path from the download/config entry.
            # With horde_model_reference pydantic records the dict has "file_name" (not "file_path"/"path").
            if "file_path" in download[i] and download[i]["file_path"]:
                file_path = f"{download[i]['file_path']}/{download[i]['file_name']}"
            elif "file_name" in download[i]:
                # Route vae/text_encoders components to their sibling ComfyUI folder so they land
                # where the loaders look (and where older hordelib versions already placed them).
                file_path = str(component_relative_path(download[i]["file_name"], download[i].get("file_purpose")))
            elif i < len(files) and "path" in files[i]:
                file_path = files[i]["path"]
            else:
                logger.error("Cannot determine file path for model download: model={}", model_name)
                return False
            download_url = None
            download_name = None
            download_path = None

            if "file_url" in download[i]:
                download_url = download[i]["file_url"]
                if self._civitai_api_token and self.is_model_url_from_civitai(download_url):
                    if "?" not in download_url:
                        download_url += f"?token={self._civitai_api_token}"
                    else:
                        download_url += f"&token={self._civitai_api_token}"
            if "file_name" in download[i]:
                download_name = download[i]["file_name"]
            if "file_path" in download[i]:
                download_path = download[i]["file_path"]

            if "manual" in download[i]:
                logger.warning(
                    f"The model {model_name} requires manual download from {download_url}. "
                    f"Please place it in {download_path}/{download_name} then press ENTER to continue...",
                )
                input("")
                continue
            # TODO: simplify
            if "file_content" in download[i]:
                file_content = download[i]["file_content"]
                logger.info("Writing file content: content={}, path={}", file_content, file_path)
                if not download_path or not download_name:
                    raise RuntimeError(
                        f"download_path and download_name are required for file_content download type for "
                        f"{model_name}",
                    )
                os.makedirs(
                    os.path.join(self.model_folder_path, download_path),
                    exist_ok=True,
                )
                with open(
                    os.path.join(self.model_folder_path, os.path.join(download_path, download_name)),
                    "w",
                ) as f:
                    f.write(file_content)
            elif "symlink" in download[i]:
                logger.info("Creating symlink: file_path={}, target={}", file_path, download[i]["symlink"])
                symlink = download[i]["symlink"]
                if not download_path or not download_name:
                    raise RuntimeError(
                        f"download_path and download_name are required for symlink download type for {model_name}",
                    )
                os.makedirs(
                    os.path.join(self.model_folder_path, download_path),
                    exist_ok=True,
                )
                os.symlink(
                    symlink,
                    os.path.join(
                        self.model_folder_path,
                        os.path.join(download_path, download_name),
                    ),
                )
            elif "git" in download[i]:
                logger.info("Git clone: url={}, path={}", download_url, file_path)
                os.makedirs(
                    os.path.join(self.model_folder_path, file_path),
                    exist_ok=True,
                )
                git.Git(os.path.join(self.model_folder_path, file_path)).clone(
                    download_url,
                )
            elif "unzip" in download[i]:
                zip_path = f"{self.model_folder_path}/{download_name}.zip"
                temp_path = f"{self.model_folder_path}/{str(uuid4())}/"
                os.makedirs(temp_path, exist_ok=True)
                if not download_url or not download_path:
                    raise RuntimeError(
                        f"download_url and download_path are required for unzip download type for {model_name}",
                    )
                download_succeeded = self.download_file(download_url, zip_path)
                if not download_succeeded:
                    return False

                logger.info("Unzipping: zip_path={}", zip_path)
                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(temp_path)

                logger.info("Moving unzipped content: from={}, to={}", temp_path, download_path)
                shutil.move(
                    temp_path,
                    os.path.join(self.model_folder_path, download_path),
                )

                logger.info("Deleting zip file: zip_path={}", zip_path)
                os.remove(zip_path)

                logger.info("Deleting temp directory: temp_path={}", temp_path)
                shutil.rmtree(temp_path)
            else:
                if not self.is_file_available(file_path) or is_model_tainted:
                    logger.debug("Downloading: url={}, path={}", download_url, self.model_folder_path / file_path)
                    if is_model_tainted:
                        logger.debug("Model is tainted: model={}", model_name)

                    if not download_url:
                        logger.error(
                            f"download_url is required for download type for {model_name}",
                        )
                        return False

                    download_succeeded = self.download_file(download_url, file_path, callback=callback)
                    if not download_succeeded:
                        return False
        return self.validate_model(model_name)

    def download_all_models(
        self,
        callback: Callable[[int, int], None] | None = None,
    ) -> bool:
        """
        Downloads all models
        """
        # FIXME this has no fall back and always returns true
        for model in self.model_reference:
            if not self.is_model_available(model):
                logger.info("Downloading model: model={}", model)
                self.download_model(model, callback=callback)
            # else:
            #   logger.debug(f"{model} is already downloaded.")
            else:
                self.validate_model(model)
        return True

    def is_model_available(self, model_name: str) -> bool:
        """
        :param model_name: Name of the model
        Checks if the model is available.
        Returns True if the model is available, False otherwise.
        """
        if model_name not in self.model_reference:
            return False

        if model_name in self.tainted_models:
            return False

        model_files = self.get_model_filenames(model_name)
        for file_entry in model_files:
            if not self.is_file_available(file_entry["file_path"]):
                logger.debug([file_entry["file_path"], self.is_file_available(file_entry["file_path"])])
                return False
        return True

    def is_model_url_from_civitai(self, url: str) -> bool:
        return CIVITAI_API_PATH in url
