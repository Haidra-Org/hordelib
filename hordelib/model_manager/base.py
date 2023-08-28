import hashlib
import importlib.resources as importlib_resources
import json
import os
import shutil
import threading
import time
import typing
import zipfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

import git
import psutil
import requests
import torch
from horde_model_reference import LEGACY_REFERENCE_FOLDER, MODEL_REFERENCE_CATEGORY, get_model_reference_filename
from loguru import logger
from tqdm import tqdm

from hordelib.comfy_horde import (
    cleanup,
    get_torch_free_vram_mb,
)
from hordelib.comfy_horde import get_torch_device as _get_torch_device
from hordelib.config_path import get_hordelib_path
from hordelib.consts import MODEL_CATEGORY_NAMES, MODEL_DB_NAMES, MODEL_FOLDER_NAMES, REMOTE_MODEL_DB
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
    modelFolderPath: str  # XXX # TODO Convert to `Path`
    """The path to the directory to store this model type."""
    model_reference: dict  # XXX is this even wanted/used/useful?
    available_models: list[str]  # XXX rework as a property?
    """The models available for immediate use."""
    tainted_models: list
    """Models which seem to be corrupted and should be deleted when the correct replacement is downloaded."""
    models_db_name: str
    models_db_path: Path
    cuda_available: bool
    cuda_devices: list
    recommended_gpu: list
    download_reference: bool
    remote_db: str

    _disk_write_mutex = threading.Lock()

    def get_torch_device(self):
        return _get_torch_device()

    def __init__(
        self,
        *,
        download_reference: bool = False,
        model_category_name: MODEL_CATEGORY_NAMES = MODEL_CATEGORY_NAMES.default_models,
        models_db_path: Path | None = None,
    ):
        """Create a new instance of this model manager.

        Args:
            model_category_name (MODEL_CATEGORY_NAMES): The category of model to manage.
            download_reference (bool, optional): Get the model DB from github. Defaults to False.
        """

        self.modelFolderPath = os.path.join(
            UserSettings.get_model_directory(),
            f"{MODEL_FOLDER_NAMES[model_category_name]}",
        )

        self.model_reference = {}
        self.available_models = []
        self.tainted_models = []
        self.pkg = importlib_resources.files("hordelib")  # XXX Remove
        self.models_db_name = MODEL_DB_NAMES[model_category_name]

        if not models_db_path:
            models_db_path = Path(get_hordelib_path()).joinpath(
                "model_database/",
                f"{self.models_db_name}.json",
            )

            if model_category_name in _temp_reference_lookup:
                models_db_path = Path(
                    get_model_reference_filename(
                        _temp_reference_lookup[model_category_name],
                        base_path=LEGACY_REFERENCE_FOLDER,
                    ),
                )

        logger.debug(f"Model database path: {models_db_path}")
        self.models_db_path = models_db_path

        self.cuda_available = torch.cuda.is_available()  # XXX Remove?
        self.cuda_devices, self.recommended_gpu = self.get_cuda_devices()  # XXX Remove?
        self.remote_db = f"{REMOTE_MODEL_DB}{self.models_db_name}.json"
        self.download_reference = download_reference
        self.load_model_database()

    def progress(self, desc="done", current=0, total=0):
        if UserSettings.download_progress_callback:
            UserSettings.download_progress_callback(desc, current, total)

    @classmethod
    def set_download_callback(cls, callback):
        UserSettings.download_progress_callback = callback

    def load_model_database(self, list_models=False):
        if self.model_reference:
            logger.info(
                (
                    "Model reference was already loaded."
                    f" Got {len(self.model_reference)} models for {self.models_db_name}."
                ),
            )
            logger.info("Reloading model reference...")
        is_model_db_present = self.models_db_path.exists()
        if self.download_reference or not is_model_db_present:
            logger.debug(f"Model reference already on disk: {is_model_db_present}")
            self.model_reference = self.download_model_reference()
            logger.info(
                (
                    "Downloaded model reference.",
                    f"Got {len(self.model_reference)} models for {self.models_db_name}.",
                ),
            )
        else:
            self.model_reference = json.loads((self.models_db_path).read_text())
            logger.info(
                (
                    "Loaded model reference from disk.",
                    f"Got {len(self.model_reference)} models for {self.models_db_name}.",
                ),
            )
        if list_models:
            for model in self.model_reference:
                logger.info(model)
        models_available = []
        for model in self.model_reference:
            if self.is_model_available(model):
                models_available.append(model)
        self.available_models = models_available
        logger.info(
            f"Got {len(self.available_models)} available models for {self.models_db_name}.",
        )
        if list_models:
            for model in self.available_models:
                logger.info(model)

    def download_model_reference(self):
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

    def _modelref_to_name(self, modelref):
        for name, data in self.get_loaded_models().items():
            if modelref and data["model"] is modelref:
                return name
        return None

    def get_free_ram_mb(self):
        return int(psutil.virtual_memory().available / (1024 * 1024))

    def get_full_model_path(self, model_name: str):
        """Returns the fully qualified filename for the specified model."""
        ckpt_path = model_name

        return f"{self.modelFolderPath}/{ckpt_path}"

    def get_model(self, model_name: str):
        return self.model_reference.get(model_name)

    def get_model_files(self, model_name: str):
        """
        :param model_name: Name of the model
        Returns the files for a model
        """
        return self.model_reference.get(model_name, {}).get("config", {}).get("files", [])

    def get_model_download(self, model_name: str):
        """
        :param model_name: Name of the model
        Returns the download details for a model
        """
        return self.model_reference[model_name]["config"]["download"]

    def get_available_models(self):
        """
        Returns the available models
        """
        return self.available_models

    def get_available_models_by_types(self, model_types: Iterable[str] | None = None):
        if not model_types:
            model_types = ["ckpt"]
        models_available = []
        for model in self.model_reference:
            if self.model_reference[model]["type"] in model_types and self._is_available(
                self.get_model_files(model),
            ):
                models_available.append(model)
        return models_available

    def count_available_models_by_types(self, model_types: Iterable[str] | None = None):
        return len(self.get_available_models_by_types(model_types))

    def taint_model(self, model_name: str):
        """Marks a model as not valid by removing it from available_models"""
        if model_name in self.available_models:
            self.available_models.remove(model_name)
            self.tainted_models.append(model_name)

    def taint_models(self, models: list[str]) -> None:
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
        files = self.get_model_files(model_name)
        logger.debug(f"Validating {model_name} with {len(files)} files")
        logger.debug(files)
        for file_details in files:
            if ".yaml" in file_details["path"]:
                continue
            if not self.is_file_available(file_details["path"]):
                logger.debug(f"File {file_details['path']} not found")
                return None
            if not skip_checksum and not self.validate_file(file_details):
                logger.warning(f"File {file_details['path']} has different contents to what we expected.")
                try:
                    # The file must have been considered valid once, or we wouldn't have renamed
                    # it from the ".part" download. Likely there is an update, or a model database hash problem
                    logger.warning(f"Likely updated, will attempt to re-download {file_details['path']}.")
                    self.taint_model(model_name)
                except OSError as e:
                    logger.error(f"Unable to delete {file_details['path']}: {e}.")
                    logger.error(f"Please delete {file_details['path']} if this error persists.")
                return False
        return True

    @staticmethod
    def get_file_md5sum_hash(file_name):  # XXX Convert to path?
        # XXX There *must* be a library for this.
        # Bail out if the source file doesn't exist
        if not os.path.isfile(file_name):
            return None

        # Check if we have a cached md5 hash for the source file
        # and use that unless our source file is newer than our hash
        md5_file = f"{os.path.splitext(file_name)[0]}.md5"
        source_timestamp = os.path.getmtime(file_name)
        hash_timestamp = os.path.getmtime(md5_file) if os.path.isfile(md5_file) else 0
        if hash_timestamp > source_timestamp:
            # Use our cached hash
            with open(md5_file) as handle:
                md5_hash = handle.read().split()[0]
            return md5_hash

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
                handle.write(f"{md5_hash} *{os.path.basename(md5_file)}")
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
                sha256_hash = handle.read().split()[0]
            return sha256_hash

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

    def validate_file(self, file_details):
        # FIXME This isn't enough or isn't being called at the right times
        """
        :param file_details: A single file from the model's files list
        Checks if the file exists and if the checksum is correct
        Returns True if the file is valid, False otherwise
        """
        full_path = f"{self.modelFolderPath}/{file_details['path']}"

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

    def is_file_available(self, file_path: str) -> bool:
        """
        :param file_path: Path of the model's file. File is from the model's files list.
        Checks if the file exists
        Returns True if the file exists, False otherwise
        """
        full_path = f"{self.modelFolderPath}/{file_path}"
        return os.path.exists(full_path)

    def _is_available(self, files) -> bool:  # FIXME what is `files` type?
        """
        :param files: List of files from the model's files list
        Checks if all files exist
        Returns True if all files exist, False otherwise
        """
        available = True
        for file in files:
            if ".yaml" in file["path"]:
                continue
            if not self.is_file_available(file["path"]):
                available = False
        return available

    def download_file(self, url: str, filename: str) -> bool:
        """
        :param url: URL of the file to download. URL is from the model's download list.
        :param file_path: Path of the model's file. File is from the model's files list.
        Downloads a file
        """
        final_pathname = f"{self.modelFolderPath}/{filename}"
        partial_pathname = f"{self.modelFolderPath}/{filename}.part"
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
                with open(partial_pathname, "ab") as f, tqdm(
                    # all optional kwargs
                    unit="B",
                    initial=partial_size,
                    unit_scale=True,
                    unit_divisor=1024,
                    miniters=1,
                    desc=filename,
                    total=remote_file_size + partial_size,
                    disable=UserSettings.download_progress_callback is not None,
                ) as pbar:
                    downloaded = partial_size
                    for chunk in response.iter_content(chunk_size=1024 * 1024 * 16):
                        response.raise_for_status()
                        if chunk:
                            downloaded += len(chunk)
                            f.write(chunk)
                            pbar.update(len(chunk))
                            self.progress(filename, downloaded, remote_file_size + partial_size)
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

    def download_model(self, model_name: str):
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
        files = self.get_model_files(model_name)
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
                    os.path.join(self.modelFolderPath, download_path),
                    exist_ok=True,
                )
                with open(
                    os.path.join(self.modelFolderPath, os.path.join(download_path, download_name)),
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
                    os.path.join(self.modelFolderPath, download_path),
                    exist_ok=True,
                )
                os.symlink(
                    symlink,
                    os.path.join(
                        self.modelFolderPath,
                        os.path.join(download_path, download_name),
                    ),
                )
            elif "git" in download[i]:
                logger.info(f"git clone {download_url} to {file_path}")
                os.makedirs(
                    os.path.join(self.modelFolderPath, file_path),
                    exist_ok=True,
                )
                git.Git(os.path.join(self.modelFolderPath, file_path)).clone(
                    download_url,
                )
            elif "unzip" in download[i]:
                zip_path = f"{self.modelFolderPath}/{download_name}.zip"
                temp_path = f"{self.modelFolderPath}/{str(uuid4())}/"
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
                    os.path.join(self.modelFolderPath, download_path),
                )

                logger.info(f"delete {zip_path}")
                os.remove(zip_path)

                logger.info(f"delete {temp_path}")
                shutil.rmtree(temp_path)
            else:
                if not self.is_file_available(file_path) or is_model_tainted:
                    logger.debug(f"Downloading {download_url} to {file_path}")
                    if is_model_tainted:
                        logger.debug(f"Model {model_name} is tainted.")

                    if not download_url:
                        logger.error(
                            f"download_url is required for download type for {model_name}",
                        )
                        return False

                    download_succeeded = self.download_file(download_url, file_path)
                    if not download_succeeded:
                        return False

        return self.validate_model(model_name)

    def download_all_models(self) -> bool:
        """
        Downloads all models
        """
        # FIXME this has no fall back and always returns true
        for model in self.get_filtered_model_names(download_all=True):
            if not self.is_model_available(model):
                logger.info(f"Downloading {model}")
                self.download_model(model)
            else:
                logger.info(f"{model} is already downloaded.")
        return True

    def is_model_available(self, model_name: str) -> bool:
        """
        :param model_name: Name of the model
        Checks if the model is available.
        Returns True if the model is available, False otherwise.
        """
        if model_name not in self.model_reference:
            return False
        return self._is_available(self.get_model_files(model_name))

    def get_cuda_devices(self):
        """
        Checks if CUDA is available.
        If CUDA is available, it returns a list of all available CUDA devices.
        If CUDA is not available, it returns an empty list.
        CUDA Device info: id, name, sm
        List is sorted by sm (compute capability) in descending order.
        Also returns the recommended GPU (highest sm).
        """
        # XXX Remove? Just make the torch calls or make a util class...s

        if torch.cuda.is_available():
            number_of_cuda_devices = torch.cuda.device_count()
            cuda_arch = []
            for i in range(number_of_cuda_devices):
                cuda_device = {
                    "id": i,
                    "name": torch.cuda.get_device_name(i),
                    "sm": torch.cuda.get_device_capability(i)[0] * 10 + torch.cuda.get_device_capability(i)[1],
                }
                cuda_arch.append(cuda_device)
            cuda_arch = sorted(cuda_arch, key=lambda k: k["sm"], reverse=True)
            recommended_gpu = [x for x in cuda_arch if x["sm"] == cuda_arch[0]["sm"]]
            return cuda_arch, recommended_gpu

        return None, None

    def get_filtered_models(self, **kwargs):
        """
        Get all models
        :param kwargs: filter based on metadata of the model reference db
        :return: list of models
        """
        filtered_models = self.model_reference
        for keyword in kwargs:
            iterating_models = filtered_models.copy()
            filtered_models = {}
            for model in iterating_models:
                if iterating_models[model].get(keyword) == kwargs[keyword]:
                    filtered_models[model] = iterating_models[model]
        return filtered_models

    def get_filtered_model_names(self, **kwargs):
        """
        Get all model names
        :param kwargs: filter based on metadata of the model reference db
        :return: list of model names
        """
        filtered_models = self.get_filtered_models(**kwargs)
        return list(filtered_models.keys())
