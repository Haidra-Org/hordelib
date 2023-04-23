import gc
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
from typing import Any
from uuid import uuid4

import git
import psutil
import requests
import torch
from loguru import logger
from tqdm import tqdm
from transformers import logging

from hordelib.cache import get_cache_directory
from hordelib.comfy_horde import get_models_on_gpu, remove_model_from_memory
from hordelib.comfy_horde import get_torch_device as _get_torch_device
from hordelib.config_path import get_hordelib_path
from hordelib.consts import REMOTE_MODEL_DB
from hordelib.settings import UserSettings

logging.set_verbosity_error()


class BaseModelManager(ABC):
    modelFolderPath: str  # XXX # TODO Convert to `Path`
    """The path to the directory to store this model type."""
    model_reference: dict  # XXX is this even wanted/used/useful?
    available_models: list[str]  # XXX rework as a property?
    _loaded_models: dict[str, dict]
    """The models available for immediate use."""
    tainted_models: list
    models_db_name: str
    models_db_path: Path
    cuda_available: bool
    cuda_devices: list
    recommended_gpu: list
    download_reference: bool
    remote_db: str
    _mutex = threading.RLock()

    def get_torch_device(self):
        return _get_torch_device()

    def get_loaded_models(self):
        return self._loaded_models.copy()

    def add_loaded_model(self, model_name, model_data):
        with self._mutex:
            self._loaded_models[model_name] = model_data

    def remove_loaded_model(self, model_name):
        with self._mutex:
            if model_name in self._loaded_models:
                del self._loaded_models[model_name]

    def __init__(
        self,
        *,
        models_db_name: str,  # XXX Remove default
        modelFolder: str | None = None,
        download_reference: bool = False,
    ):
        """Create a new instance of this model manager.

        Args:
            models_db_name (str): The standard name for this model category. (e.g., 'clip')
            download_reference (bool, optional): Get the model DB from github. Defaults to False.
        """
        if not modelFolder:
            self.modelFolderPath = os.path.join(
                get_cache_directory(),
                f"{models_db_name}",
            )
        else:  # XXX # FIXME The only exception to this right now is compvis
            self.modelFolderPath = os.path.join(get_cache_directory(), f"{modelFolder}")

        self.model_reference = {}
        self.available_models = []
        self._loaded_models = {}
        self.tainted_models = []
        self.pkg = importlib_resources.files("hordelib")  # XXX Remove
        self.models_db_name = models_db_name
        self.models_db_path = Path(get_hordelib_path()).joinpath(
            "model_database/",
            f"{self.models_db_name}.json",
        )

        self.cuda_available = torch.cuda.is_available()  # XXX Remove?
        self.cuda_devices, self.recommended_gpu = self.get_cuda_devices()  # XXX Remove?
        self.remote_db = f"{REMOTE_MODEL_DB}{self.models_db_name}.json"
        self.download_reference = download_reference
        self.loadModelDatabase()

    def loadModelDatabase(self, list_models=False):
        if self.model_reference:
            logger.info(
                (
                    "Model reference was already loaded."
                    f" Got {len(self.model_reference)} models for {self.models_db_name}."
                ),
            )
            logger.info("Reloading model reference...")

        if self.download_reference:
            self.model_reference = self.download_model_reference()
            logger.info(
                " ".join(
                    [
                        "Downloaded model reference.",
                        f"Got {len(self.model_reference)} models for {self.models_db_name}.",
                    ],
                ),
            )
        else:
            self.model_reference = json.loads((self.models_db_path).read_text())
            logger.info(
                " ".join(
                    [
                        "Loaded model reference from disk.",
                        f"Got {len(self.model_reference)} models for {self.models_db_name}.",
                    ],
                ),
            )
        if list_models:
            for model in self.model_reference:
                logger.info(model)
        models_available = []
        for model in self.model_reference:
            if self.check_model_available(model):
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
            logger.init("Model Reference", status="Downloading")
            response = requests.get(self.remote_db)
            logger.init_ok("Model Reference", status="OK")
            models = response.json()
            return models
        except Exception as e:  # XXX Double check and/or rework this
            logger.init_err(
                "Model Reference",
                status=f"Download failed: {e}",
            )
            logger.init_warn("Model Reference", status="Local")
            return json.loads((self.models_db_path).read_text())

    def _modelref_to_name(self, modelref):
        with self._mutex:
            for name, data in self.get_loaded_models().items():
                if modelref and data["model"] is modelref:
                    return name
            return None

    def ensure_memory_available(self):
        # Can this type of model be cached?
        if not self.can_cache_on_disk():
            return
        with self._mutex:
            # If we have less than the minimum RAM free, free some up
            freemem = round(psutil.virtual_memory().available / (1024 * 1024))
            logger.debug(f"Free RAM is: {freemem} MB, ({len(self.get_loaded_models())} models loaded in RAM)")
            if freemem > UserSettings.ram_to_leave_free_mb + 4096:
                return
            logger.debug("Not enough free RAM attempting to free some")
            # Grab a list of models (ModelPatcher) that are loaded on the gpu
            # These are actually returned with the least important at the bottom of the list
            busy_models = get_models_on_gpu()
            # Try to find one we have in ram that isn't on the gpu
            idle_model = None
            for ram_model_name, ram_model_data in self.get_loaded_models().items():
                if type(ram_model_data["model"]) is not str and ram_model_data["model"] not in busy_models:
                    idle_model = ram_model_name
                    break
            # If we didn't have one hanging around in ram not on the gpu
            # pick the least used gpu model
            if not idle_model and busy_models:
                idle_model = self._modelref_to_name(busy_models[-1])

            if idle_model:
                logger.debug(f"Moving model {idle_model} to disk to free up RAM")
                self.move_to_disk_cache(idle_model)
            else:
                # Nothing else to release
                logger.debug("Could not find a model to free RAM")

    # @abstractmethod # TODO
    def move_to_disk_cache(self, model_name):
        pass

    def can_cache_on_disk(self):
        """Can this of type model be cached on disk?"""
        return False

    def load(
        self,
        model_name: str,
        *,
        half_precision: bool = True,
        gpu_id: int | None = 0,
        cpu_only: bool = False,
        local: bool = False,
        **kwargs,
    ):  # XXX # FIXME
        with self._mutex:
            self.ensure_memory_available()
            if not local and model_name not in self.model_reference:
                logger.error(f"{model_name} not found")
                return False
            if not local and model_name not in self.available_models:
                logger.error(f"{model_name} not available")
                download_succeeded = self.download_model(model_name)
                if not download_succeeded:
                    logger.init_err(f"{model_name} failed to download", status="Error")
                    return False
                logger.init_ok(f"{model_name}", status="Downloaded")
            if model_name not in self.get_loaded_models():
                if not local:
                    model_validated = self.validate_model(model_name)
                    if not model_validated:
                        return False
                logger.init(f"{model_name}", status="Loading")

                tic = time.time()
                logger.init(f"{model_name}", status="Loading")

                try:
                    self.add_loaded_model(
                        model_name,
                        self.modelToRam(
                            model_name=model_name,
                            half_precision=half_precision,
                            gpu_id=gpu_id,
                            cpu_only=cpu_only,
                            local=local,
                            **kwargs,
                        ),
                    )

                except RuntimeError:
                    # It failed, it happens.
                    logger.error(f"Failed to load model {model_name}")
                    return None

                toc = time.time()

                logger.init_ok(f"{model_name}: {round(toc-tic,2)} seconds", status="Loaded")
                return True
            return None

    def getFullModelPath(self, model_name: str):
        """Returns the fully qualified filename for the specified model."""
        ckpt_path = self.get_model_files(model_name)[0]["path"]  # XXX Rework?
        return f"{self.modelFolderPath}/{ckpt_path}"

    @abstractmethod
    def modelToRam(
        self,
        *,
        model_name: str,
        half_precision: bool = True,
        gpu_id: int | None = None,
        cpu_only: bool = False,
        **kwargs,  # XXX I'd like to refactor the need for this away
    ) -> dict[str, typing.Any]:  # XXX Flesh out signature
        """Load a model into RAM. Returns a dict with at least key 'model'.

        Args:
            model_name (str): The name of the model to load.
            half_precision (bool, optional): Whether to use half precision. Defaults to True.
            gpu_id (int | None, optional): The GPU to load into. Defaults to None.
            cpu_only (bool, optional): Whether to only use CPU + System RAM. Defaults to False.
            kwargs (dict[str, typing.Any]): Additional arguments.

        Returns:
            dict[str, typing.Any]: A dict with at least key 'model'.
        """

    def get_model(self, model_name: str):
        return self.model_reference.get(model_name)

    def get_model_files(self, model_name: str):
        """
        :param model_name: Name of the model
        Returns the files for a model
        """
        return self.model_reference[model_name]["config"]["files"]

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

    def get_available_models_by_types(self, model_types: list[str] | None = None):
        if not model_types:
            model_types = ["ckpt", "diffusers"]
        models_available = []
        for model in self.model_reference:
            if self.model_reference[model]["type"] in model_types and self.check_available(
                self.get_model_files(model),
            ):
                models_available.append(model)
        return models_available

    def count_available_models_by_types(self, model_types: list[str] | None = None):
        return len(self.get_available_models_by_types(model_types))

    def get_loaded_model(self, model_name: str):
        """
        :param model_name: Name of the model
        Returns the loaded model
        """
        return self.get_loaded_models()[model_name]

    def get_loaded_models_names(self, string=False) -> list[str] | str:  # XXX Rework 'string' param
        """Return a list of loaded model names.

        Args:
            string (bool, optional): Return as a comma separated string. Defaults to False.

        Returns:
            list[str] | str: The list of models, as a `list` or a comma separated string.
        """
        # return ["Deliberate"]
        with self._mutex:
            if string:
                return ", ".join(self.get_loaded_models().keys())
            return list(self.get_loaded_models().keys())

    def is_model_loaded(self, model_name: str) -> bool:
        """Returns True if the model is loaded, False otherwise.

        Args:
            model_name (str): The name of the model to check.

        Returns:
            _type_: _description_
        """
        with self._mutex:
            return model_name in self.get_loaded_models()

    def unload_model(self, model_name: str):
        """
        :param model_name: Name of the model
        Issue a request to unload a model, completely remove it, free all resources, forget it exists.
        This may not be possible right now, as it may be being used, so we actually issue a request for it to
        be unloaded at the earliest opportunity.
        """
        with self._mutex:
            if model_name in self._loaded_models:
                self.free_model_resources(model_name)
                self.remove_loaded_model(model_name)
                return True

    def free_model_resources(self, model_name: str):
        with self._mutex:
            remove_model_from_memory(model_name, self.get_loaded_model(model_name))

    def unload_all_models(self):
        """
        Unloads all models
        """
        with self._mutex:
            for model in self.get_loaded_models():
                self.unload_model(model)
            return True

    def taint_model(self, model_name: str):
        """Marks a model as not valid by removing it from available_models"""
        if model_name in self.available_models:
            self.available_models.remove(model_name)
            self.tainted_models.append(model_name)

    def taint_models(self, models: list[str]) -> None:
        for model in models:
            self.taint_model(model)

    def validate_model(self, model_name: str, skip_checksum: bool = False):
        # XXX This isn't enough or isn't called at the right times.
        """
        :param model_name: Name of the model
        :param skip_checksum: If True, skips checksum validation
        For each file in the model, checks if the file exists and if the checksum is correct
        Returns True if all files are valid, False otherwise
        """
        files = self.get_model_files(model_name)
        logger.debug(f"Validating {model_name} with {len(files)} files")
        logger.debug(files)
        for file_details in files:
            if ".yaml" in file_details["path"]:
                continue
            if not self.check_file_available(file_details["path"]):
                logger.debug(f"File {file_details['path']} not found")
                return False
            if not skip_checksum and not self.validate_file(file_details):
                logger.error(f"File {file_details['path']} has invalid checksum")
                try:
                    modelPath = Path(self.modelFolderPath).joinpath(file_details["path"])
                    logger.error(f"Deleting {file_details['path']}.")
                    modelPath.unlink(True)
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
                chunk = file_to_check.read(8192)  # Changed just because it broke pylint
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
                chunk = file_to_check.read(8192)
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
        # XXX TODO This isn't enough or isn't being called at the right times
        """
        :param file_details: A single file from the model's files list
        Checks if the file exists and if the checksum is correct
        Returns True if the file is valid, False otherwise
        """
        full_path = f"{self.modelFolderPath}/{file_details['path']}"

        # Default to sha256 hashes
        if "sha256sum" in file_details:
            logger.debug(f"Getting sha256sum of {full_path}")
            sha256_file_hash = self.get_file_sha256_hash(full_path)
            logger.debug(f"sha256sum: {sha256_file_hash}")
            logger.debug(f"Expected: {file_details['sha256sum']}")
            return file_details["sha256sum"] == sha256_file_hash

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

    def check_file_available(self, file_path):
        """
        :param file_path: Path of the model's file. File is from the model's files list.
        Checks if the file exists
        Returns True if the file exists, False otherwise
        """
        full_path = f"{self.modelFolderPath}/{file_path}"
        return os.path.exists(full_path)

    def check_available(self, files):  # XXX what is `files` type?
        """
        :param files: List of files from the model's files list
        Checks if all files exist
        Returns True if all files exist, False otherwise
        """
        available = True
        for file in files:
            if ".yaml" in file["path"]:
                continue
            if not self.check_file_available(file["path"]):
                available = False
        return available

    def download_file(self, url: str, filename: str) -> bool:
        """
        :param url: URL of the file to download. URL is from the model's download list.
        :param file_path: Path of the model's file. File is from the model's files list.
        Downloads a file
        """
        # XXX convert the path nonsense to pathlib
        full_path = f"{self.modelFolderPath}/{filename}"
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        pbar_desc = full_path.split("/")[-1]
        try:
            response = requests.get(url, stream=True, allow_redirects=True)
            with open(full_path, "wb") as f, tqdm(
                # all optional kwargs
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                miniters=1,
                desc=pbar_desc,
                total=int(response.headers.get("content-length", 0)),
                disable=UserSettings.disable_download_progress.active,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=16 * 1024):
                    response.raise_for_status()
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            return True
        except requests.exceptions.HTTPError as e:
            logger.error(f"Error downloading {url}: {e}")
            if os.path.exists(full_path):
                os.remove(full_path)
            return False
        except requests.exceptions.RequestException as e:
            logger.error(f"Error downloading {url}: {e}")
            logger.error("Are you connected to the internet?")
            if os.path.exists(full_path):
                os.remove(full_path)
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
        if model_name in self.available_models and model_name not in self.tainted_models:
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
                if not self.check_file_available(file_path) or model_name in self.tainted_models:
                    logger.debug(f"Downloading {download_url} to {file_path}")
                    download_succeeded = self.download_file(download_url, file_path)
                    if not download_succeeded:
                        return False

        return self.validate_model(model_name)

    def download_all_models(self) -> bool:
        """
        Downloads all models
        """
        # XXX this has no fall back and always returns true
        for model in self.get_filtered_model_names(download_all=True):
            if not self.check_model_available(model):
                logger.init(f"{model}", status="Downloading")  # logger.init
                self.download_model(model)
            else:
                logger.init(f"{model} is already downloaded.", status="Skipped")
        return True

    def check_model_available(self, model_name: str) -> bool:
        """
        :param model_name: Name of the model
        Checks if the model is available.
        Returns True if the model is available, False otherwise.
        """
        if model_name not in self.model_reference:
            return False
        return self.check_available(self.get_model_files(model_name))

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
