import hashlib
import importlib.resources as importlib_resources
import json
import os
import shutil
import threading
import time
import zipfile
from abc import ABC
from collections.abc import Callable, Iterable
from pathlib import Path
from urllib import parse
from uuid import uuid4

import git
import psutil
import requests
from horde_model_reference import LEGACY_REFERENCE_FOLDER, MODEL_REFERENCE_CATEGORY, get_model_reference_filename
from loguru import logger
from tqdm import tqdm

from hordelib.config_path import get_hordelib_path
from hordelib.consts import CIVITAI_API_PATH, MODEL_CATEGORY_NAMES, MODEL_DB_NAMES, MODEL_FOLDER_NAMES, REMOTE_MODEL_DB
from hordelib.settings import UserSettings

_temp_reference_lookup = {
    MODEL_CATEGORY_NAMES.codeformer: MODEL_REFERENCE_CATEGORY.codeformer,
    MODEL_CATEGORY_NAMES.compvis: MODEL_REFERENCE_CATEGORY.stable_diffusion,
    MODEL_CATEGORY_NAMES.controlnet: MODEL_REFERENCE_CATEGORY.controlnet,
    MODEL_CATEGORY_NAMES.esrgan: MODEL_REFERENCE_CATEGORY.esrgan,
    MODEL_CATEGORY_NAMES.gfpgan: MODEL_REFERENCE_CATEGORY.gfpgan,
    MODEL_CATEGORY_NAMES.safety_checker: MODEL_REFERENCE_CATEGORY.safety_checker,
}


class BaseModelManager(ABC):
    model_folder_path: Path
    """The path to the directory to store this model type."""
    model_reference: dict  # XXX is this even wanted/used/useful?
    available_models: list[str]  # XXX rework as a property?
    """The models available for immediate use."""
    tainted_models: list
    """Models which seem to be corrupted and should be deleted when the correct replacement is downloaded."""
    models_db_name: str
    models_db_path: Path
    recommended_gpu: list
    download_reference: bool
    remote_db: str

    _disk_write_mutex = threading.Lock()

    def __init__(
        self,
        *,
        download_reference: bool = False,
        model_category_name: MODEL_CATEGORY_NAMES = MODEL_CATEGORY_NAMES.default_models,
        models_db_path: Path | None = None,
        civitai_api_token: str | None = None,
        **kwargs,
    ):
        """Create a new instance of this model manager.

        Args:
            model_category_name (MODEL_CATEGORY_NAMES): The category of model to manage.
            download_reference (bool, optional): Get the model DB from github. Defaults to False.
        """

        if len(kwargs) > 0:
            logger.debug(f"Unused kwargs in {type(self)}: {kwargs}")

        self.model_folder_path = UserSettings.get_model_directory() / f"{MODEL_FOLDER_NAMES[model_category_name]}"

        os.makedirs(self.model_folder_path, exist_ok=True)

        self.model_reference = {}
        self.available_models = []
        self.tainted_models = []
        self.pkg = importlib_resources.files("hordelib")  # XXX Remove
        self.models_db_name = MODEL_DB_NAMES[model_category_name]
        self._civitai_api_token = parse.quote_plus(civitai_api_token) if civitai_api_token else None

        if not models_db_path:
            models_db_path = get_hordelib_path() / "model_database" / f"{self.models_db_name}.json"

            if model_category_name in _temp_reference_lookup:
                models_db_path = Path(
                    get_model_reference_filename(
                        _temp_reference_lookup[model_category_name],
                        base_path=LEGACY_REFERENCE_FOLDER,
                    ),
                )

        if models_db_path is None:
            raise ValueError(f"Model database path not found for {model_category_name}")

        logger.debug(f"Model database path: {models_db_path}")
        self.models_db_path = models_db_path

        self.remote_db = f"{REMOTE_MODEL_DB}{self.models_db_name}.json"
        self.download_reference = download_reference
        self.load_model_database()

    def progress(self, desc="done", current=0, total=0):
        # TODO
        return

    def load_model_database(self) -> None:
        if self.model_reference:
            logger.info("Model reference was already loaded.")
            logger.info("Reloading model reference...")

        is_model_db_present = self.models_db_path.exists()

        if self.download_reference or not is_model_db_present:
            raise NotImplementedError("Downloading model databases is no longer supported within hordelib.")

        for attempt in range(3):
            try:
                self.model_reference = json.loads((self.models_db_path).read_text())
            except json.decoder.JSONDecodeError as e:
                if attempt <= 2:
                    logger.warning(
                        f"Model database {self.models_db_path} is not valid JSON: {e}. Will retry: {attempt+1}/3",
                    )
                    time.sleep(1)
                    continue
                logger.error(f"Model database {self.models_db_path} is not valid JSON: {e}")
                raise

        models_available = []
        for model in self.model_reference:
            if self.is_model_available(model):
                models_available.append(model)
        self.available_models = models_available
        logger.info(
            f"Got {len(self.available_models)} available models for {self.models_db_name}.",
        )

    def download_model_reference(self) -> dict:
        try:
            logger.debug(f"Downloading Model Reference for {self.models_db_name}")
            response = requests.get(self.remote_db)
            logger.debug("Downloaded Model Reference successfully")
            models = response.json()
            logger.info("Updated Model Reference from remote.")
            return models
        except Exception as e:  # XXX Double check and/or rework this
            logger.error(
                f"Download failed: {e}",
            )
            logger.warning("Model Reference not downloaded, using local copy")
            if self.models_db_path.exists():
                return json.loads(self.models_db_path.read_text())
            logger.error("No local copy of Model Reference found!")
            return {}

    def get_free_ram_mb(self) -> int:
        """Returns the amount of free RAM in MB rounded down to the nearest integer.

        Returns:
            int: The amount of free RAM in MB
        """
        return int(psutil.virtual_memory().available / (1024 * 1024))

    def get_model_reference_info(self, model_name: str) -> dict | None:
        return self.model_reference.get(model_name, None)

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
        if model_name not in self.model_reference:
            raise ValueError(f"Model {model_name} not found in model reference")

        model_file_entries = self.model_reference.get(model_name, {}).get("config", {}).get("files", [])
        model_files = []
        for model_file_entry in model_file_entries:
            path_config_item = model_file_entry.get("path")
            path_config_type = model_file_entry.get("file_type")
            if path_config_item:
                if path_config_item.endswith((".ckpt", ".safetensors", ".pt", ".pth", ".bin")):
                    path_entry = {"file_path": Path(path_config_item)}
                    if path_config_type:
                        path_entry["file_type"] = path_config_type
                    model_files.append(path_entry)
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
        return self.model_reference.get(model_name, {}).get("config", {}).get("files", [])

    def get_model_download(self, model_name: str) -> dict:
        """Return the download config for a given model name.

        Args:
            model_name (str): The name of the model to get the download config for.

        Returns:
            dict: The download config for the model.
        """

        return self.model_reference.get(model_name, {}).get("config", {}).get("download", [])

    def get_available_models(self) -> list[str]:
        """Return the available (downloaded and verified) models."""
        return self.available_models

    def get_available_models_by_types(self, model_types: Iterable[str] | None = None) -> list[str]:
        """Return the available (downloaded and verified) models of a given type.

        Args:
            model_types (Iterable[str] | None, optional): The type of model to return. See the model
            reference for valid values. Defaults to None.

        Returns:
            list[str]: The available models of the given type.
        """
        if not model_types:
            model_types = ["ckpt"]
        models_available = []
        for model in self.model_reference:
            if self.model_reference[model]["type"] in model_types and self.is_model_available(model):
                models_available.append(model)
        return models_available

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
        logger.debug(f"Validating {model_name}. Files: {model_files}")
        for file_entry in model_files:
            if not self.is_file_available(file_entry["file_path"]):
                return None

        file_details = self.get_model_config_files(model_name)

        for file_detail in file_details:
            if ".yaml" in file_detail["path"] or ".json" in file_detail["path"]:
                continue
            if not self.is_file_available(file_detail["path"]):
                logger.debug(f"File {file_detail['path']} not found")
                return None
            if not skip_checksum and not self.validate_file(file_detail):
                logger.warning(f"File {file_detail['path']} has different contents to what we expected.")
                try:
                    # The file must have been considered valid once, or we wouldn't have renamed
                    # it from the ".part" download. Likely there is an update, or a model database hash problem
                    logger.warning(f"Likely updated, will attempt to re-download {file_detail['path']}.")
                    self.taint_model(model_name)
                except OSError as e:
                    logger.error(f"Unable to delete {file_detail['path']}: {e}.")
                    logger.error(f"Please delete {file_detail['path']} if this error persists.")
                return False

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
        logger.info(f"Calculating sha256sum of {file_name}. This may take a while.")
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
        full_path = f"{self.model_folder_path}/{file_details['path']}"

        # Default to sha256 hashes
        if "sha256sum" in file_details:
            logger.debug(f"Getting sha256sum of {full_path}")
            sha256_file_hash = self.get_file_sha256_hash(full_path).lower()
            expected_hash = file_details["sha256sum"].lower()
            logger.debug(f"sha256sum: {sha256_file_hash}")
            logger.debug(f"Expected: {expected_hash}")
            return expected_hash == sha256_file_hash

        # If sha256 is not available, fall back to md5
        if "md5sum" in file_details:
            logger.debug(f"Getting md5sum of {full_path}")
            md5_file_hash = self.get_file_md5sum_hash(full_path)
            logger.debug(f"md5sum: {md5_file_hash}")
            logger.debug(f"Expected: {file_details['md5sum']}")
            return file_details["md5sum"] == md5_file_hash

        # If no hashes available, return True for now
        # THIS IS A SECURITY RISK, EVENTUALLY WE SHOULD RETURN FALSE
        # But currently not all models specify hashes
        # XXX this warning preexists me (@tazlin), probably should look into it

        logger.debug(f"Model {file_details['path']} doesn't have a checksum, skipping validation!")

        return True

    def is_file_available(self, file_path: str | Path) -> bool:
        """
        :param file_path: Path of the model's file. File is from the model's files list.
        Checks if the file exists
        Returns True if the file exists, False otherwise
        """
        parsed_full_path = Path(f"{self.model_folder_path}/{file_path}")
        if parsed_full_path.suffix == ".part":
            logger.debug(f"File {file_path} is a partial download, skipping")
            return False
        sha_file_path = Path(f"{self.model_folder_path}/{parsed_full_path.stem}.sha256")

        if parsed_full_path.exists() and not sha_file_path.exists():
            self.get_file_sha256_hash(parsed_full_path)

        return parsed_full_path.exists() and sha_file_path.exists()

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
                logger.info(f"Resuming download of file {filename}")
            else:
                # If file doesn't exist, start from beginning
                logger.info(f"Starting download of file {filename}")
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
                        logger.info(f"Successfully downloaded the file {filename}")
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
                logger.info(f"Successfully downloaded the file {filename}")
                self.progress()
                if os.path.exists(final_pathname):
                    os.remove(final_pathname)
                # Move the downloaded data into it's final location
                os.rename(partial_pathname, final_pathname)
                return True

            except requests.RequestException:
                logger.info(f"Download of file {filename} failed.")
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
            logger.debug(f"{model_name} is already available.")
            return True
        download = self.get_model_download(model_name)
        files = self.get_model_config_files(model_name)
        if self.is_model_available(model_name):
            logger.debug(f"{model_name} is already downloaded.")
            return True
        for i in range(len(download)):
            file_path = (
                f"{download[i]['file_path']}/{download[i]['file_name']}"
                if "file_path" in download[i]
                else files[i]["path"]
            )
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
                logger.info(f"writing {file_content} to {file_path}")
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
                logger.info(f"symlink {file_path} to {download[i]['symlink']}")
                symlink = download[i]["symlink"]
                if not download_path or not download_name:
                    raise RuntimeError(
                        f"download_path and download_name are required for symlink download type for " f"{model_name}",
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
                logger.info(f"git clone {download_url} to {file_path}")
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

                logger.info(f"unzip {zip_path}")
                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(temp_path)

                logger.info(f"moving {temp_path} to {download_path}")
                shutil.move(
                    temp_path,
                    os.path.join(self.model_folder_path, download_path),
                )

                logger.info(f"delete {zip_path}")
                os.remove(zip_path)

                logger.info(f"delete {temp_path}")
                shutil.rmtree(temp_path)
            else:
                if not self.is_file_available(file_path) or is_model_tainted:
                    logger.debug(f"Downloading {download_url} to {self.model_folder_path / file_path}")
                    if is_model_tainted:
                        logger.debug(f"Model {model_name} is tainted.")

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
                logger.info(f"Downloading {model}")
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
                return False
        return True

    def is_model_url_from_civitai(self, url: str) -> bool:
        return CIVITAI_API_PATH in url
