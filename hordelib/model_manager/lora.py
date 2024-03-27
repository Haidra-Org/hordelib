import copy
import glob
import hashlib
import json
import os
import re
import threading
import time
from collections import deque
from contextlib import nullcontext
from datetime import datetime, timedelta
from enum import auto
from multiprocessing.synchronize import Lock as multiprocessing_lock
from pathlib import Path

import requests
from fuzzywuzzy import fuzz
from horde_model_reference import LEGACY_REFERENCE_FOLDER
from loguru import logger
from strenum import StrEnum
from typing_extensions import override

import hordelib.exceptions as he
from hordelib.consts import MODEL_CATEGORY_NAMES
from hordelib.model_manager.base import BaseModelManager
from hordelib.utils.sanitizer import Sanitizer


class DOWNLOAD_SIZE_CHECK(StrEnum):
    everything = auto()
    top = auto()
    adhoc = auto()


TESTS_ONGOING = os.getenv("TESTS_ONGOING", "0") == "1"

AIWORKER_LORA_CACHE_SIZE_DEFAULT = 10 * 1024  # 10GB


class LoraModelManager(BaseModelManager):
    LORA_DEFAULTS = "https://raw.githubusercontent.com/Haidra-Org/AI-Horde-image-model-reference/main/lora.json"
    LORA_VERSIONS_DEFAULTS = (
        "https://raw.githubusercontent.com/Haidra-Org/AI-Horde-image-model-reference/main/lora_versions.json"
    )
    LORA_API = "https://civitai.com/api/v1/models?types=LORA&sort=Highest%20Rated&primaryFileOnly=true"
    MAX_RETRIES = 10 if not TESTS_ONGOING else 3
    MAX_DOWNLOAD_THREADS = 5 if not TESTS_ONGOING else 15
    RETRY_DELAY = 3 if not TESTS_ONGOING else 0.2
    """The time to wait between retries in seconds"""
    REQUEST_METADATA_TIMEOUT = 20  # Longer because civitai performs poorly on metadata requests for more than 5 models
    """The maximum time for no data to be received before we give up on a metadata fetch, in seconds"""
    REQUEST_DOWNLOAD_TIMEOUT = 10 if not TESTS_ONGOING else 1
    """The maximum time for no data to be received before we give up on a download, in seconds

    This is not the time to download the file, but the time to wait in between data packets. \
    If we're actively downloading and the connection to the server is alive, this doesn't come into play
    """

    THREAD_WAIT_TIME = 0.1
    """The time to wait between checking the download queue in seconds"""

    _file_lock: multiprocessing_lock | nullcontext

    def __init__(
        self,
        download_reference=False,
        allowed_top_lora_storage=10240 if not TESTS_ONGOING else 1024,
        allowed_adhoc_lora_storage=AIWORKER_LORA_CACHE_SIZE_DEFAULT,
        download_wait=False,
        multiprocessing_lock: multiprocessing_lock | None = None,
        civitai_api_token: str | None = None,
    ):
        self.max_adhoc_disk = allowed_adhoc_lora_storage
        try:
            AIWORKER_LORA_CACHE_SIZE = os.getenv("AIWORKER_LORA_CACHE_SIZE")
            if AIWORKER_LORA_CACHE_SIZE is not None:
                self.max_adhoc_disk = int(AIWORKER_LORA_CACHE_SIZE)
                logger.debug(f"AIWORKER_LORA_CACHE_SIZE is {self.max_adhoc_disk}")
        except (ValueError, TypeError):
            logger.error(f"AIWORKER_LORA_CACHE_SIZE is not a valid integer: {AIWORKER_LORA_CACHE_SIZE}")
            logger.warning(f"Using default value of {AIWORKER_LORA_CACHE_SIZE_DEFAULT}")
            self.max_adhoc_disk = AIWORKER_LORA_CACHE_SIZE_DEFAULT

        self._max_top_disk = allowed_top_lora_storage

        self._data = None
        self._next_page_url = None
        self._mutex = threading.Lock()
        self._file_mutex = threading.Lock()
        self._download_mutex = threading.Lock()
        self._file_lock = multiprocessing_lock or nullcontext()
        self._adhoc_loras_savefile: Path

        self._file_count = 0
        self._download_threads = {}  # type: ignore # FIXME: add type
        self._download_queue = deque()  # type: ignore # FIXME: add type
        self._thread = None
        self.stop_downloading = True
        # Not yet handled, as we need a global reference to search through.
        self._previous_model_reference = {}  # type: ignore # FIXME: add type
        self._adhoc_loras = set()  # type: ignore # FIXME: add type
        self._download_wait = download_wait
        # If false, this MM will only download SFW loras
        self.nsfw = True
        self._adhoc_reset_thread = None
        self._stop_all_threads = False
        self._index_ids = {}  # type: ignore # FIXME: add type
        self._index_version_ids = {}  # type: ignore # FIXME: add type
        self._index_orig_names = {}  # type: ignore # FIXME: add type
        self.total_retries_attempted = 0
        self._default_lora_ids: list = []

        models_db_path = LEGACY_REFERENCE_FOLDER.joinpath("lora.json").resolve()

        super().__init__(
            model_category_name=MODEL_CATEGORY_NAMES.lora,
            download_reference=download_reference,
            models_db_path=models_db_path,
            civitai_api_token=civitai_api_token,
        )
        # FIXME (shift lora.json handling into horde_model_reference?)

    @override
    def load_model_database(self) -> None:
        if self.model_reference:
            logger.info(
                (
                    "Model reference was already loaded."
                    f" Got {len(self.model_reference)} models for {self.models_db_name}."
                ),
            )
            logger.info("Reloading model reference...")

        self._adhoc_loras_savefile = Path(f"{os.path.splitext(self.models_db_path)[0]}_adhoc.json")
        if self.download_reference:
            os.makedirs(self.model_folder_path, exist_ok=True)
            self.download_model_reference()
            logger.info("Lora reference download begun asynchronously.")
        else:
            if self.models_db_path.exists():
                try:
                    with self._file_mutex, self._file_lock:
                        self.model_reference = json.loads((self.models_db_path).read_text())
                    new_model_reference = {}
                    for old_lora_key in self.model_reference.keys():
                        lora = self.model_reference[old_lora_key]
                        # If for some reasons the file for that lora doesn't exist, we ignore it.
                        if "versions" not in lora:
                            filepath = os.path.join(self.model_folder_path, lora["filename"])
                            if not Path(f"{filepath}").exists():
                                logger.debug(filepath)
                                continue
                            lora_key = Sanitizer.sanitise_model_name(lora["orig_name"]).lower().strip()
                            lora_filename = f'{Sanitizer.sanitise_filename(lora["orig_name"])}_{lora["version_id"]}'
                            version = {
                                "filename": f"{lora_filename}.safetensors",
                                "sha256": lora["sha256"],
                                "adhoc": lora.get("adhoc"),
                                "size_mb": lora["size_mb"],
                                "url": lora["url"],
                                "triggers": lora["triggers"],
                                "baseModel": lora["baseModel"],
                                "version_id": self.ensure_is_version(lora["version_id"]),
                                "lora_key": lora_key,
                            }
                            old_filename = os.path.join(self.model_folder_path, lora["filename"])
                            new_filename = os.path.join(self.model_folder_path, version["filename"])
                            logger.warning(
                                f"Old LoRa format detected for {lora_key}. Converting format and renaming files: "
                                f"{old_filename} -> {new_filename}",
                            )
                            if os.path.exists(new_filename):
                                logger.warning(
                                    f"New filename {new_filename} already exists. "
                                    "This shouldn't happen. Skipping...",
                                )
                                continue

                            os.rename(old_filename, new_filename)
                            old_hashfile = f"{os.path.splitext(old_filename)[0]}.sha256"
                            new_hashfile = f"{os.path.splitext(new_filename)[0]}.sha256"
                            os.rename(old_hashfile, new_hashfile)
                            new_lora_entry = {
                                "name": lora_key,
                                "orig_name": lora["orig_name"],
                                "id": lora["id"],
                                "nsfw": lora["nsfw"],
                                "versions": {self.ensure_is_version(lora["version_id"]): version},
                            }
                            new_model_reference[lora_key] = new_lora_entry
                        else:
                            existing_versions = {}
                            for version in lora["versions"]:
                                filepath = os.path.join(self.model_folder_path, lora["versions"][version]["filename"])
                                if not Path(f"{filepath}").exists():
                                    logger.warning(f"{filepath} doesn't exist. Removing lora version from reference.")
                                    continue
                                existing_versions[version] = lora["versions"][version]
                            lora["versions"] = existing_versions
                            new_model_reference[old_lora_key] = lora
                    for lora in new_model_reference.values():
                        self._index_ids[lora["id"]] = lora["name"]
                        orig_name = lora.get("orig_name", lora["name"]).lower()
                        self._index_orig_names[orig_name] = lora["name"]
                        for version_id in lora["versions"]:
                            self._index_version_ids[version_id] = lora["name"]
                    self.model_reference = new_model_reference
                    logger.info("Loaded model reference from disk.")
                except json.JSONDecodeError:
                    logger.error(f"Could not load {self.models_db_name} model reference from disk! Bad JSON?")
                    self.model_reference = {}
            else:
                logger.warning(
                    f"Could not load {self.models_db_name} model reference from disk! File not found. "
                    "This is normal if you're running the this for the first time.",
                )
                self.model_reference = {}
            self.save_cached_reference_to_disk()
            logger.info(
                (f"Got {len(self.model_reference)} models for {self.models_db_name}.",),
            )

    def download_model_reference(self):
        # We have to wipe it, as we are going to be adding it it instead of replacing it
        # We're not downloading now, as we need to be able to init without it
        self.model_reference = {}
        self.save_cached_reference_to_disk()

    def _get_lora_defaults(self):
        try:
            self._default_lora_ids = self._get_json(self.LORA_DEFAULTS)
            if not isinstance(self._default_lora_ids, list):
                logger.error("Could not download default LoRas reference!")
                self._default_lora_ids = []
            self._add_lora_ids_to_download_queue(self._default_lora_ids)

        except Exception as err:
            logger.error(f"_get_lora_defaults() raised {err}")
            raise err

    def _add_lora_ids_to_download_queue(self, lora_ids, adhoc=False, version_compare=None, id_type="lora"):
        if id_type == "version":
            # CivitAI doesn't support fetching multiple versions from the same API call, so we have to loop
            for version_id in lora_ids:
                model_items = []
                if isinstance(version_id, str) and not version_id.isdigit:
                    logger.warning(f"Non-integer lora model version sent: {version_id}. Ignoring...")
                url = f"https://civitai.com/api/v1/model-versions/{version_id}"
                data = self._get_json(url)
                if not data:
                    logger.warning(f"metadata for LoRa {lora_ids} could not be downloaded!")
                    return
                model_items.append(data)
        else:
            idsq = "&ids=".join([str(id) for id in lora_ids])
            url = f"https://civitai.com/api/v1/models?limit=100&ids={idsq}"
            data = self._get_json(url)
            if not data:
                logger.warning(f"metadata for LoRa {lora_ids} could not be downloaded!")
                return
            model_items = data.get("items", [])
        for lora_data in model_items:
            lora = self._parse_civitai_lora_data(lora_data, adhoc=adhoc)
            # If we're comparing versions, then we don't download if the existing lora metadata matches
            # Instead we just refresh metadata information
            if not lora:
                continue
            if version_compare and self.find_latest_version(lora) == version_compare:
                logger.debug(
                    f"Downloaded metadata for LoRa {lora['name']} "
                    f"('{lora['name']}') and found version match! Refreshing metadata.",
                )
                lora["last_checked"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self._add_lora_to_reference(lora)
                continue
            # logger.debug(f"Downloaded metadata for LoRas {lora['id']} ('{lora['name']}') and added to download queue"
            self._download_lora(lora)

    def _get_json(self, url):
        retries = 0
        while retries <= self.MAX_RETRIES:
            response = None
            try:
                response = requests.get(
                    url,
                    timeout=self.REQUEST_METADATA_TIMEOUT if len(url) < 200 else self.REQUEST_METADATA_TIMEOUT * 1.5,
                )
                response.raise_for_status()
                # Attempt to decode the response to JSON
                return response.json()

            except (requests.HTTPError, requests.ConnectionError, requests.Timeout, json.JSONDecodeError) as e:
                logger.debug(f"url '{url}' download failed {type(e)} {e}")

                # If this is a 401, 404, or 500, we're not going to get anywhere, just give up
                # The following are the CivitAI errors encountered so far
                # [401: requires a token, 404: model ID too long, 500: internal server error]
                if response is not None:
                    if response.status_code in [401, 404]:
                        logger.debug(f"url '{url}' download failed with status code {response.status_code}")
                        return None
                    if response.status_code == 500:
                        logger.debug(f"url '{url}' download failed with status code {response.status_code}")
                        retries += 3

                # The json being invalid is a CivitAI issue, possibly it showing an HTML page and
                # this isn't likely to change in the next 30 seconds, so we'll try twice more
                # and give up if it doesn't work
                if isinstance(e, json.JSONDecodeError):
                    logger.debug(f"url '{url}' download failed with {type(e)} {e}")
                    retries += 3

                # If the network connection timed out, then self.REQUEST_METADATA_TIMEOUT seconds passed
                # and the clock is ticking, so we'll try once more
                if response is None:
                    retries += 5

                retries += 1
                self.total_retries_attempted += 1

                if retries <= self.MAX_RETRIES:
                    time.sleep(self.RETRY_DELAY)
                else:
                    # Max retries exceeded, give up
                    return None

            except Exception as e:
                # Failed badly
                logger.error(f"url '{url}' download failed {e}")
                return None
        return None

    def _get_more_items(self):
        if not self._data:
            # We need to lowercase the boolean, or CivitAI doesn't understand it >.>
            url = f"{self.LORA_API}&nsfw={str(self.nsfw).lower()}"
        else:
            url = self._next_page_url

        # This may be the end of the road, unlikely but...
        if not url:
            logger.warning("End of LORA data reached")
            self.stop_downloading = True
        else:
            # Get the actual item data
            items = self._get_json(url)
            if items:
                self._data = items
                self._next_page_url = self._data.get("metadata", {}).get("nextPage", "")
            else:
                # We failed to get more items
                logger.error("Failed to download all LORA data even after retries.")
                self._data = None
                self._next_page_url = None  # give up

    def _parse_civitai_lora_data(self, item, adhoc=False):
        """Return a simplified dictionary with the information we actually need about a lora"""
        lora = {
            "name": "",
            "orig_name": "",
            "id": "",
            "nsfw": False,
            "versions": {},
        }
        # This means it's a model version, not a full model
        if "modelId" in item:
            version = item
            lora_id = item["modelId"]
            # If we've seen this lora before, we avoid redownloading its info
            if lora_id in self._index_ids:
                lora_data = self.model_reference[self._index_ids[lora_id]]
                lora_name = lora_data.get("orig_name", "")
            else:
                lora_data = self._get_json(f"https://civitai.com/api/v1/models/{lora_id}")
                if lora_data is None:
                    logger.debug(f"Rejecting LoRa version {lora_id} because we can't retrieve it's model data")
                    return None
                lora_name = lora_data.get("name", "")
            lora_nsfw = lora_data.get("nsfw", True)
        # If it's a full model, we only grab the first version in the list
        else:
            try:
                version = item.get("modelVersions", {})[0]
            except IndexError:
                version = {}
            lora_name = item.get("name", "")
            lora_id = item.get("id", 0)
            lora_nsfw = item.get("nsfw", True)
        lora_version = self.ensure_is_version(version.get("id", 0))
        # Get model triggers
        triggers = version.get("trainedWords", [])
        # get first file that is a primary file and a safetensor
        for file in version.get("files", {}):
            if file.get("primary", False) and file.get("name", "").endswith(".safetensors"):
                sanitized_name = Sanitizer.sanitise_model_name(lora_name)
                lora_filename = f'{Sanitizer.sanitise_filename(lora_name)[0:128]}_{version.get("id", 0)}'
                lora_key = sanitized_name.lower().strip()
                lora["name"] = lora_key
                lora["orig_name"] = lora_name
                lora["id"] = lora_id
                lora["nsfw"] = lora_nsfw
                lora["versions"] = {}
                lora["versions"][lora_version] = {}
                lora["versions"][lora_version]["filename"] = f"{lora_filename}.safetensors"
                lora["versions"][lora_version]["sha256"] = file.get("hashes", {}).get("SHA256")
                lora["versions"][lora_version]["adhoc"] = adhoc
                try:
                    lora["versions"][lora_version]["size_mb"] = round(file.get("sizeKB", 0) / 1024)
                except TypeError:
                    lora["versions"][lora_version][
                        "size_mb"
                    ] = 144  # guess common case of 144MB, it's not critical here
                    logger.debug(f"Could not get sizeKB for {lora['name']} version {lora_version}. Using 144MB")
                lora["versions"][lora_version]["url"] = file.get("downloadUrl", "")
                lora["versions"][lora_version]["triggers"] = triggers
                lora["versions"][lora_version]["baseModel"] = version.get("baseModel", "SD 1.5")
                lora["versions"][lora_version]["version_id"] = self.ensure_is_version(version.get("id", 0))
                # To be able to refer back to the parent if needed
                lora["versions"][lora_version]["lora_key"] = lora_key
                break
        # If we don't have everything required, fail
        if lora["versions"][lora_version]["adhoc"] and not lora["versions"][lora_version].get("sha256"):
            logger.debug(f"Rejecting LoRa {lora.get('name')} because it doesn't have a sha256")
            return None
        if not lora["versions"][lora_version].get("filename") or not lora["versions"][lora_version].get("url"):
            logger.debug(f"Rejecting LoRa {lora.get('name')} because it doesn't have a url")
            return None
        # We don't want to start downloading GBs of a single LoRa.
        # We just ignore anything over 150Mb. Them's the breaks...
        if (
            lora["versions"][lora_version]["adhoc"]
            and lora["versions"][lora_version]["size_mb"] > 220
            and lora["id"] not in self._default_lora_ids
        ):
            logger.debug(f"Rejecting LoRa {lora.get('name')} version {lora_version} because its size is over 220Mb.")
            return None
        if lora["versions"][lora_version]["adhoc"] and lora["nsfw"] and not self.nsfw:
            logger.debug(f"Rejecting LoRa {lora.get('name')} because worker is SFW.")
            return None
        # Fixup A1111 centric triggers
        for i, trigger in enumerate(lora["versions"][lora_version]["triggers"]):
            if re.match("<lora:(.*):.*>", trigger):
                lora["versions"][lora_version]["triggers"][i] = re.sub("<lora:(.*):.*>", "\\1", trigger)
        return lora

    def _download_thread(self, thread_number):
        # We try to download the LORA. There are tens of thousands of these things, we aren't
        # picky if downloads fail, as they often will if civitai is the host, we just move on to
        # the next one
        logger.debug(f"Started Download Thread {thread_number}")
        while True:
            # Endlessly look for files to download and download them
            if self._stop_all_threads:
                logger.debug(f"Stopped Download Thread {thread_number}")
                return
            try:
                lora = self._download_queue.popleft()
                self._download_threads[thread_number]["lora"] = lora
            except IndexError:
                # Nothing in the queue
                # logger.debug(f"Lora download thread {thread_number} idle.")
                self._download_threads[thread_number]["lora"] = None
                time.sleep(self.THREAD_WAIT_TIME)
                continue
            # Download the lora
            retries = 0
            # Normally a lora in this method should only have one version specified
            # So we can use this method to extract it.
            version = self.find_latest_version(lora)
            logger.debug(f"Starting download thread  {thread_number} for lora {lora['orig_name']} version {version}.")
            if version is None:
                logger.warning("Lora without version sent to the download queue. This should never happen. Ignoring")
                continue
            while retries <= self.MAX_RETRIES:
                try:
                    # Just before we download this file, check if we already have it
                    filename = os.path.join(self.model_folder_path, lora["versions"][version]["filename"])
                    hashfile = f"{os.path.splitext(filename)[0]}.sha256"
                    if os.path.exists(filename) and os.path.exists(hashfile):
                        # Check the hash
                        with open(hashfile) as infile:
                            try:
                                hashdata = infile.read().split()[0]
                            except (IndexError, OSError, PermissionError) as e:
                                logger.error(
                                    f"Error reading hash file {hashfile} for {filename} {type(e)} {e}",
                                )
                                hashdata = ""
                        if (
                            not lora["versions"][version].get("sha256")
                            or hashdata.lower() == lora["versions"][version]["sha256"].lower()
                        ):
                            # we already have this lora, consider it downloaded
                            # the SHA256 might not exist when the lora has been selected in the curation list
                            # Where we allow them to skip it
                            if not lora["versions"][version].get("sha256"):
                                logger.debug(
                                    f"Already have LORA {lora['versions'][version]['filename']}. "
                                    "Bypassing SHA256 check as there's none stored",
                                )
                            # else:
                            #     logger.debug(f"Already have LORA {lora['filename']}")
                            # We store as lower to allow case-insensitive search
                            lora["last_checked"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            self._add_lora_to_reference(lora)
                            if self.is_default_cache_full():
                                self.stop_downloading = True
                            self.save_cached_reference_to_disk()
                            break

                    logger.info(f"Starting download of LORA {lora['versions'][version]['filename']}")

                    lora_url = lora["versions"][version]["url"]
                    if self._civitai_api_token and self.is_model_url_from_civitai(lora_url):
                        lora_url += f"?token={self._civitai_api_token}"

                    response = requests.get(lora_url, timeout=self.REQUEST_DOWNLOAD_TIMEOUT)
                    response.raise_for_status()
                    if "reason=download-auth" in response.url:
                        logger.error(
                            f"Error downloading {lora['versions'][version]['filename']}. "
                            "CivitAI appears to be redirecting us to a login page. Aborting",
                        )
                        break
                    # Check the data hash
                    hash_object = hashlib.sha256()
                    hash_object.update(response.content)
                    sha256 = hash_object.hexdigest()
                    if not lora.get("sha256") or sha256.lower() == lora["versions"][version]["sha256"].lower():
                        # wow, we actually got a valid file, save it
                        with open(filename, "wb") as outfile:
                            outfile.write(response.content)
                        # Save the hash file
                        with open(hashfile, "w") as outfile:
                            outfile.write(f"{sha256} *{lora['versions'][version]['filename']}")
                        # Shout about it
                        logger.info(
                            f"Downloaded LORA {lora['versions'][version]['filename']} "
                            f"({lora['versions'][version]['size_mb']} MB)",
                        )
                        # Maybe we're done
                        # We store as lower to allow case-insensitive search
                        lora["versions"][version]["last_used"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        lora["last_checked"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        self._add_lora_to_reference(lora)
                        if self.is_adhoc_cache_full():
                            self.delete_oldest_lora()
                        self.save_cached_reference_to_disk()
                        break

                    # We will retry
                    logger.debug(
                        f"Downloaded LORA file for {lora['versions'][version]['filename']} didn't match hash. "
                        f"Retry {retries}/{self.MAX_RETRIES}",
                    )

                except (requests.HTTPError, requests.ConnectionError, requests.Timeout, json.JSONDecodeError) as e:
                    # We will retry
                    logger.debug(
                        f"Error downloading {lora['versions'][version]['filename']} {e} ({type(e)}). "
                        f"Retry {retries}/{self.MAX_RETRIES}",
                    )

                    # If this is a 401, we're not going to get anywhere, just just give up
                    if isinstance(e, requests.HTTPError) and e.response.status_code == 401:
                        logger.error(
                            f"Error downloading {lora['versions'][version]['filename']}. "
                            "CivitAI appears to be redirecting us to a login page. Aborting",
                        )
                        break

                except Exception as e:
                    # Failed badly, ignore and retry
                    logger.debug(
                        f"Fatal error downloading {lora['versions'][version]['filename']} {e} ({type(e)}). "
                        f"Retry {retries}/{self.MAX_RETRIES}",
                    )

                retries += 1
                self.total_retries_attempted += 1
                if retries > self.MAX_RETRIES:
                    break  # fail

                time.sleep(self.RETRY_DELAY)

    def _download_lora(self, lora):
        with self._download_mutex:
            # Start download threads if they aren't already started
            while len(self._download_threads) < self.MAX_DOWNLOAD_THREADS:
                thread_iter = len(self._download_threads)
                logger.debug(f"Starting download thread {thread_iter}")
                thread = threading.Thread(target=self._download_thread, daemon=True, args=(thread_iter,))
                self._download_threads[thread_iter] = {"thread": thread, "lora": None}
                logger.debug(f"Started download thread {thread_iter}")
                logger.debug(f"Download threads: {self._download_threads}")
                thread.start()

            # Add this lora to the download queue
            self._download_queue.append(lora)

    def _process_items(self):
        # i.e. given a bunch of LORA item metadata, download them
        if not self._data:
            logger.debug("No LORA data to process")
            return
        for item in self._data.get("items", []):
            lora = self._parse_civitai_lora_data(item)
            if lora:
                self._file_count += 1
                # Allow a queue of 20% larger than the max disk space as we'll lose some
                if self.calculate_download_queue() + self.calculate_downloaded_loras() > self._max_top_disk:
                    return
                # We have valid lora data, download it
                self._download_lora(lora)

    def _start_processing(self):
        self.stop_downloading = False

        while not self.stop_downloading:
            if self._stop_all_threads:
                logger.debug("Stopped processing thread")
                return
            # Get some items to download
            self._get_more_items()

            # If we have some items to process, process them
            if self._data:
                self._process_items()

    def _add_lora_to_reference(self, lora):
        # We don't want other threads trying to write on the reference while we're reloading it
        with self._mutex:
            self.reload_reference_from_disk()
            lora_key = lora["name"]
            if lora.get("adhoc", False):
                self._adhoc_loras.add(lora_key)
                # Once added to our set, we don't need to specify it was adhoc anymore
                del lora["adhoc"]
            # A lora coming to add to our reference should only have 1 version
            # we do the version merge in this function
            # We convert the lora version to string when using it as key
            # To avoid converting the lora_name to int every time later
            new_lora_version = self.ensure_is_version(list(lora["versions"].keys())[0])
            if lora_key not in self.model_reference:
                self.model_reference[lora_key] = lora
            else:
                # If we already know of one version of this lora, we simply add the extra version to our versions
                if new_lora_version not in self.model_reference[lora_key]["versions"]:
                    self.model_reference[lora_key]["versions"][new_lora_version] = lora["versions"][new_lora_version]
            self._index_ids[lora["id"]] = lora_key
            self._index_version_ids[new_lora_version] = lora_key
            orig_name = lora.get("orig_name", lora["name"]).lower()
            self._index_orig_names[orig_name] = lora_key

    def reload_reference_from_disk(self):
        with self._file_mutex, self._file_lock:
            reloaded_model_reference: dict = json.loads((self.models_db_path).read_text())
            if self.models_db_path.exists():
                reloaded_adhoc_loras = set(json.loads((self._adhoc_loras_savefile).read_text()))
            else:
                reloaded_adhoc_loras = set()
            reloaded_index_ids = {}
            reloaded_index_orig_names = {}
            reloaded_index_version_ids = {}
            for lora in self.model_reference.values():
                reloaded_index_ids[lora["id"]] = lora["name"]
                orig_name = lora.get("orig_name", lora["name"]).lower()
                reloaded_index_orig_names[orig_name] = lora["name"]
                for version in lora["versions"]:
                    reloaded_index_version_ids[version] = lora["name"]
            self.model_reference = reloaded_model_reference
            self._adhoc_loras = reloaded_adhoc_loras
            self._index_ids = reloaded_index_ids
            self._index_orig_names = reloaded_index_orig_names
            self._index_version_ids = reloaded_index_version_ids

    def download_default_loras(self, nsfw=True, timeout=None):
        """Start up a background thread downloading and return immediately"""
        # Don't start if we're already busy doing something
        if not self.are_downloads_complete():
            return
        self.nsfw = nsfw
        self._previous_model_reference = copy.deepcopy(self.model_reference)
        # TODO: Avoid clearing this out, until we know CivitAI is not dead.
        self.clear_all_references()
        os.makedirs(self.model_folder_path, exist_ok=True)
        # Start processing in a background thread
        self._thread = threading.Thread(target=self._get_lora_defaults, daemon=True)
        self._thread.start()
        # Wait for completion of our threads if requested
        if self._download_wait:
            self.wait_for_downloads(timeout)
        if self.is_adhoc_reset_complete():
            self._adhoc_reset_thread = threading.Thread(target=self.reset_adhoc_loras, daemon=True)
            self._adhoc_reset_thread.start()

    def clear_all_references(self):
        self.model_reference = {}
        self._index_ids = {}
        self._index_version_ids = {}
        self._index_orig_names = {}

    def wait_for_downloads(self, timeout=None):
        rtr = 0
        while not self.are_downloads_complete():
            time.sleep(self.THREAD_WAIT_TIME)
            rtr += self.THREAD_WAIT_TIME
            if timeout and rtr > timeout:
                raise Exception(f"Lora downloads exceeded specified timeout ({timeout})")
        logger.debug("Downloads complete")

    def are_downloads_complete(self):
        # If we don't have any models in our reference, then we haven't downloaded anything
        # perhaps faulty civitai?
        if self._thread and self._thread.is_alive():
            return False
        if not self.are_download_threads_idle():
            return False
        if len(self._download_queue) > 0:
            return False
        return self.stop_downloading

    def are_download_threads_idle(self):
        # logger.debug([dthread["lora"] for dthread in self._download_threads.values()])
        for dthread in self._download_threads.values():
            if dthread["lora"] is not None:
                return False
        return True

    def find_lora_key_by_version(self, lora_version: int | str) -> str | None:
        version = self.ensure_is_version(lora_version)
        if version in self._index_version_ids:
            return self._index_version_ids[version]
        return None

    def ensure_is_version(self, lora_version: int | str) -> str | None:
        """The version has to be a string, as json keys can only be strings"""
        if not isinstance(lora_version, str):
            if not isinstance(lora_version, int):
                return None
        return str(lora_version)

    def fuzzy_find_lora_key(self, lora_name: int | str):
        # sname = Sanitizer.remove_version(lora_name).lower()
        # logger.debug(f"Looking for lora {lora_name}")
        if isinstance(lora_name, int) or lora_name.isdigit():
            if int(lora_name) in self._index_ids:
                return self._index_ids[int(lora_name)]
            return None
        if lora_name in self.model_reference:
            return lora_name
        if lora_name.lower() in self._index_orig_names:
            return self._index_orig_names[lora_name.lower()]
        if Sanitizer.has_unicode(lora_name):
            for lora in self._index_orig_names:
                if lora_name in lora:
                    return self._index_orig_names[lora]
            # If a unicode name is not found in the orig_names index
            # it won't be found anywhere else, as unicode chars are converted to ascii in the keys
            # This saves us time doing unnecessary fuzzy searches
            return None
        for lora in self.model_reference:
            if lora_name in lora:
                return lora
        for lora in self.model_reference:
            if fuzz.ratio(lora_name, lora) > 80:
                return lora
            if fuzz.ratio(lora_name, self.model_reference[lora]["orig_name"]) > 80:
                return lora
        return None

    # Using `get_model` instead of `get_lora` as it exists in the base class
    def get_model_reference_info(self, model_name: str | int, is_version: bool = False) -> dict | None:
        """Returns the actual lora details dict for the specified model_name search string
        Returns None if lora name not found"""
        with self._mutex:
            if is_version:
                lora_key = self.find_lora_key_by_version(model_name)
                if lora_key is None:
                    return None
                if lora_key not in self.model_reference:
                    logger.warning(
                        "LoRa key still in versions but missing from our reference. Attempting to refresh indexes...",
                    )
                    self.reload_reference_from_disk()
                    self.save_cached_reference_to_disk()
                    lora_key = self.find_lora_key_by_version(model_name)
                    if lora_key not in self.model_reference:
                        logger.error(
                            "Missing lora still in the versions index. Something has gone wrong. "
                            "Please contact the developers.",
                        )
                    else:
                        logger.info(
                            "Reloaded LoRa reference and now lora properly cleared. "
                            "Will not attempt to download it again at this time.",
                        )
                    return None
                return self.model_reference[self.find_lora_key_by_version(model_name)]
            lora_name = self.fuzzy_find_lora_key(model_name)
            if not lora_name:
                return None
            return self.model_reference.get(lora_name)

    def get_lora_filename(self, model_name: str, is_version: bool = False):
        """Returns the actual lora filename for the specified model_name search string
        Returns None if lora name not found"""
        lora = self.get_model_reference_info(model_name, is_version)
        if not lora:
            return None
        if is_version:
            return lora["versions"][model_name]["filename"]
        latest_version = self.get_latest_version(lora)
        if latest_version is None:
            return None
        return latest_version["filename"]

    def get_lora_name(self, model_name: str, is_version: bool = False):
        """Returns the actual lora name for the specified model_name search string
        Returns None if lora name not found
        The lora name is effectively the key in the dictionary"""
        lora = self.get_model_reference_info(model_name, is_version)
        if not lora:
            return None
        return lora["name"]

    def get_lora_triggers(self, model_name: str, is_version: bool = False):
        """Returns a list of triggers for a specified lora name
        Returns an empty list if no triggers are found
        Returns None if lora name not found"""
        lora = self.get_model_reference_info(model_name, is_version)
        if not lora:
            return None
        if is_version:
            triggers = lora["versions"][model_name].get("triggers")
        else:
            logger.debug(lora)
            lora_version = self.get_latest_version(lora)
            if lora_version is None:
                return None
            triggers = lora_version.get("triggers")
        if triggers:
            return triggers
        # We don't `return lora.get("triggers", [])`` to avoid the returned list object being modified
        # and then we keep returning previous items
        return []

    def find_lora_trigger(self, model_name: str, trigger_search: str, is_version: bool = False):
        """Searches for a specific trigger for a specified lora name
        Returns None if string not found even with fuzzy search"""
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

    def save_cached_reference_to_disk(self):
        with self._file_mutex, self._file_lock:
            with open(self.models_db_path, "w", encoding="utf-8", errors="ignore") as outfile:
                outfile.write(json.dumps(self.model_reference.copy(), indent=4))
            with open(self._adhoc_loras_savefile, "w", encoding="utf-8", errors="ignore") as outfile:
                outfile.write(json.dumps(list(self._adhoc_loras), indent=4))

    def calculate_downloaded_loras(self, mode=DOWNLOAD_SIZE_CHECK.everything):
        total_size = 0
        for lora in self.model_reference:
            if mode == DOWNLOAD_SIZE_CHECK.top and lora in self._adhoc_loras:
                continue
            if mode == DOWNLOAD_SIZE_CHECK.adhoc and lora not in self._adhoc_loras:
                continue
            for version in self.model_reference[lora]["versions"]:
                total_size += self.model_reference[lora]["versions"][version]["size_mb"]
        return total_size

    def calculate_default_loras_cache(self):
        return self.calculate_downloaded_loras(DOWNLOAD_SIZE_CHECK.top)

    def calculate_adhoc_loras_cache(self):
        return self.calculate_downloaded_loras(DOWNLOAD_SIZE_CHECK.adhoc)

    def is_default_cache_full(self):
        return self.calculate_default_loras_cache() >= self._max_top_disk

    def is_adhoc_cache_full(self):
        return self.calculate_adhoc_loras_cache() >= self.max_adhoc_disk

    def calculate_download_queue(self):
        total_queue = 0
        for lora in self._download_queue:
            total_queue += lora["size_mb"]
        return total_queue

    def find_oldest_adhoc_lora(self) -> str | None:
        oldest_lora: str | None = None
        oldest_datetime: datetime | None = None
        for lora in self._adhoc_loras:
            for version in self.model_reference[lora]["versions"]:
                lora_datetime = datetime.strptime(
                    self.model_reference[lora]["versions"][version]["last_used"],
                    "%Y-%m-%d %H:%M:%S",
                )
                if not oldest_lora:
                    oldest_lora = version
                    oldest_datetime = lora_datetime
                    continue
                if oldest_datetime and oldest_datetime > lora_datetime:
                    oldest_lora = version
                    oldest_datetime = lora_datetime
        return oldest_lora

    def delete_oldest_lora(self):
        oldest_version = self.find_oldest_adhoc_lora()
        if not oldest_version:
            return
        self.delete_lora(oldest_version)

    def find_lora_from_filename(self, filename: str):
        for lora in self.model_reference:
            for version in self.model_reference[lora]["versions"]:
                if self.model_reference[lora]["versions"][version]["filename"] == filename:
                    return lora
        return None

    def find_unused_loras(self):
        files = glob.glob(f"{self.model_folder_path}/*.safetensors")
        filenames = set()
        for stfile in files:
            filename = os.path.basename(stfile)
            if not self.find_lora_from_filename(filename):
                filenames.add(filename)
        return filenames

    def delete_unused_loras(self, timeout=0):
        """Deletes downloaded LoRas which do not appear in the model_reference
        By default protects the user by not running if are_downloads_complete() is not done
        """
        waited = 0
        while not self.are_downloads_complete():
            if waited >= timeout:
                raise Exception(f"Waiting for current LoRa downloads exceeded specified timeout ({timeout})")
            waited += 0.2
            time.sleep(0.2)
        loras_to_delete = self.find_unused_loras()
        for lora_filename in loras_to_delete:
            self.delete_lora_files(lora_filename)
        return loras_to_delete

    def delete_lora_files(self, lora_filename: str):
        filename = os.path.join(self.model_folder_path, lora_filename)
        if not os.path.exists(filename):
            logger.warning(f"Could not find LoRa file on disk to delete: {filename}")
            return
        os.remove(filename)
        hashfile = f"{os.path.splitext(filename)[0]}.sha256"
        if not os.path.exists(hashfile):
            os.remove(hashfile)
        logger.info(f"Deleted LoRa file: {filename}")

    def delete_lora(self, lora_version: str):
        """Deletes the version passed. If not versions remains, forgets the lora"""
        # We don't want other threads trying to write on the reference while we're reloading it
        with self._mutex:
            self.reload_reference_from_disk()
        lora_info = self.get_model_reference_info(lora_version, is_version=True)
        if not lora_info:
            logger.warning(f"Could not find lora version {lora_version} to delete")
            return
        lora_key = lora_info["name"]
        self.delete_lora_files(lora_info["versions"][lora_version]["filename"])
        del self._index_version_ids[lora_version]
        del lora_info["versions"][lora_version]
        if len(lora_info["versions"]) == 0:
            if lora_key in self._adhoc_loras:
                self._adhoc_loras.remove(lora_key)
            del self._index_ids[lora_info["id"]]
            del self._index_orig_names[lora_info["orig_name"].lower()]
            del self.model_reference[lora_key]
        self.save_cached_reference_to_disk()

    def ensure_lora_deleted(self, lora_name: str):
        logger.debug(f"Ensuring {lora_name} is deleted")
        lora_key = self.fuzzy_find_lora_key(lora_name)
        if not lora_key:
            return
        logger.debug(lora_key)
        logger.debug(self.model_reference[lora_key])
        logger.debug(self.model_reference[lora_key]["versions"])
        for version in list(self.model_reference[lora_key]["versions"].keys()):
            self.delete_lora(version)

    def reset_adhoc_loras(self):
        """Compared the known loras from the previous run to the current one
        Adds any definitions as adhoc loras, until we have as many Mb as self.max_adhoc_disk"""
        while not self.are_downloads_complete():
            if self._stop_all_threads:
                logger.debug("Stopped processing thread")
                return
            time.sleep(self.THREAD_WAIT_TIME)
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._adhoc_loras = set()
        unsorted_items = []
        sorted_items = []
        for plora_key, plora in self._previous_model_reference.items():
            for version in plora.get("versions", {}).values():
                unsorted_items.append((plora_key, version))
        try:
            sorted_items = sorted(
                unsorted_items,
                key=lambda x: x[1].get("last_used", now),
                reverse=True,
            )
        except Exception as err:
            logger.error(err)
        while not self.is_adhoc_cache_full() and len(sorted_items) > 0:
            prevlora_key, prevversion = sorted_items.pop()
            if prevlora_key in self.model_reference:
                continue
            # If True, it will initiates a redownload and call _add_lora_to_reference() later
            if not self._check_for_refresh(prevlora_key):
                if "last_used" not in prevversion:
                    prevversion["last_used"] = now
                # We create a temp lora dict holding the just one version (the one we want to keep)
                # The _add_lora_to_reference() will anyway merge versions if we keep more than 1
                temp_lora = self._previous_model_reference[prevlora_key].copy()
                temp_lora["versions"] = {}
                temp_lora["versions"][prevversion["version_id"]] = prevversion
                self._add_lora_to_reference(temp_lora)
            self._adhoc_loras.add(prevlora_key)
        self._previous_model_reference = {}
        self.save_cached_reference_to_disk()
        logger.debug("Finished lora reset")

    def get_lora_metadata(self, url: str) -> dict:
        """Returns parsed Lora details from civitAI
        or returns None on errors or when it cannot be found"""
        data = self._get_json(url)
        # CivitAI down
        if not data:
            raise he.CivitAIDown("CivitAI Down?")
        if "items" in data:
            if len(data["items"]) == 0:
                raise he.ModelEmpty("Lora appears empty")
            lora = self._parse_civitai_lora_data(data["items"][0], adhoc=True)
        else:
            lora = self._parse_civitai_lora_data(data, adhoc=True)
        # This can happen if the lora for example does not provide a sha256
        if not lora:
            raise he.ModelInvalid("Lora is invalid")
        return lora

    def _check_for_refresh(self, lora_name: str):
        """Returns True if a refresh has been initiated
        Else returns False
        """
        lora_details = self.get_model_reference_info(lora_name)
        if not lora_details:
            return False
        refresh = False
        if "last_checked" not in lora_details:
            refresh = True
        elif "versions" not in lora_details:
            refresh = True
        else:
            lora_datetime = datetime.strptime(lora_details["last_checked"], "%Y-%m-%d %H:%M:%S")
            if lora_datetime < datetime.now() - timedelta(days=1):
                refresh = True
        if refresh:
            logger.debug(f"Lora {lora_name} found needing refresh. Initiating metadata download...")
            url = f"https://civitai.com/api/v1/models/{lora_details['id']}"
            try:
                lora = self.get_lora_metadata(url)
            except Exception as e:
                logger.warning(f"Could not refresh lora {lora_name} metadata: ({type(e)}) {e}")
                return False
            latest_version = self.find_latest_version(lora)
            if latest_version != self.find_latest_version(lora_details):
                self._add_lora_ids_to_download_queue([lora["id"]], latest_version)
                return True
            # We need to record to disk the last time this lora was checked
            with self._mutex:
                self.reload_reference_from_disk()
            lora_details = self.get_model_reference_info(lora_name)
            if lora_details is not None:
                lora_details["last_checked"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self.save_cached_reference_to_disk()
        return False

    # def check_for_valid

    def is_adhoc_reset_complete(self):
        if self._adhoc_reset_thread and self._adhoc_reset_thread.is_alive():
            return False
        return True

    def wait_for_adhoc_reset(self, timeout=15):
        rtr = 0
        while not self.is_adhoc_reset_complete():
            time.sleep(self.THREAD_WAIT_TIME)
            rtr += self.THREAD_WAIT_TIME
            if timeout and rtr > timeout:
                raise Exception(f"Lora adhoc reset exceeded specified timeout ({timeout})")

    def stop_all(self):
        self._stop_all_threads = True

    def _touch_lora(self, lora_name: str | int, is_version: bool = False):
        """Updates the "last_used" key in a lora entry to current UTC time"""
        lora = self.get_model_reference_info(lora_name, is_version)
        if not lora:
            logger.warning(f"Could not find lora {lora_name} to touch")
            return
        if not is_version:
            version = self.find_latest_version(lora)
        else:
            version = self.ensure_is_version(lora_name)
        if version is None:
            return
        lora["versions"][version]["last_used"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.save_cached_reference_to_disk()
        # logger.debug(f"Touched lora {lora_name}")

    def find_latest_version(self, lora) -> str | None:
        all_versions = [int(v) for v in lora.get("versions", {}).keys()]
        if len(all_versions) > 0:
            all_versions.sort(reverse=True)
            return self.ensure_is_version(all_versions[0])
        return None

    def get_latest_version(self, lora) -> dict | None:
        version_id = self.find_latest_version(lora)
        if version_id is None:
            return None
        return lora["versions"][version_id]

    def get_lora_last_use(self, lora_name, is_version: bool = False):
        """Returns a dateimte object based on the "last_used" key in a lora entry"""
        lora = self.get_model_reference_info(lora_name, is_version)
        if not lora:
            logger.warning(f"Could not find lora {lora_name} to get last use")
            return None
        if is_version:
            version = self.ensure_is_version(lora_name)
        else:
            version = self.find_latest_version(lora)
        if version is None:
            return None
        return datetime.strptime(lora["versions"][version]["last_used"], "%Y-%m-%d %H:%M:%S")

    def fetch_adhoc_lora(self, lora_name, timeout: int | None = 45, is_version: bool = False) -> str | None:
        """Checks if a LoRa is available. If not, waits for it to be downloaded to complete *if timeout is set*.

        - If timeout is set, it will wait for the download to complete and return the lora name.
        - If timeout is *not* set, it will begin a download thread and return `None` immediately.
        """
        if is_version and not isinstance(lora_name, int) and not lora_name.isdigit():
            logger.debug("Lora version requested, but lora name is not an integer")
            return None
        if isinstance(lora_name, int) or lora_name.isdigit():
            if is_version:
                url = f"https://civitai.com/api/v1/model-versions/{lora_name}"
            else:
                url = f"https://civitai.com/api/v1/models/{lora_name}"
        else:
            url = f"{self.LORA_API}&nsfw={str(self.nsfw).lower()}&query={lora_name}"
        # We're not using self.get_lora_metadata(url)
        # in order to be able to know if CivitAI is down
        try:
            lora = self.get_lora_metadata(url)
        except he.CivitAIDown as err:
            if isinstance(lora_name, str):
                if lora_name in self.model_reference:
                    self._touch_lora(lora_name)
                    logger.warning(f"CivitAI Appears down. Using cached info: {err}")
                    return lora_name
            logger.warning(f"Exception when getting lora metadata: {err}")
            return None
        except Exception as err:
            logger.warning(f"Exception when getting lora metadata: {err}")
            return None
        # We double-check that somehow our search missed it but CivitAI searches differently and found it
        fuzzy_find = self.fuzzy_find_lora_key(lora["id"])
        if fuzzy_find:
            # For versions of loras, we check in the lora versions
            # to see if we've downloaded this specific version before
            if is_version and lora_name in self.model_reference[fuzzy_find]["versions"]:
                logger.debug(f"Found lora version with ID: {fuzzy_find}")
                self._touch_lora(lora_name, True)
                return fuzzy_find
            # We only return the known version, if it's also the latest version when no version was requested.
            if not is_version and self.get_latest_version(
                self.get_model_reference_info(fuzzy_find),
            ) == self.get_latest_version(lora):
                logger.debug(f"Found lora with ID: {fuzzy_find}")
                self._touch_lora(lora_name, False)
                return fuzzy_find
        self._download_lora(lora)
        if timeout is not None:
            # We need to wait a bit to make sure the threads pick up the download
            time.sleep(self.THREAD_WAIT_TIME)
            self.wait_for_downloads(timeout)
            version = self.find_latest_version(lora)
            if is_version:
                version = lora_name
            if version is None:
                return None
            self._touch_lora(version, True)
            return lora["name"]
        return None

    def do_baselines_match(self, lora_name, model_details, is_version: bool = False):
        self._check_for_refresh(lora_name)
        lora_details = self.get_model_reference_info(lora_name, is_version)
        return True  # FIXME: Disabled for now
        if not lora_details:
            logger.warning(f"Could not find lora {lora_name} to check baselines")
            return False
        if "SD 1.5" in lora_details["baseModel"] and model_details["baseline"] == "stable diffusion 1":
            return True
        if "SD 2.1" in lora_details["baseModel"] and model_details["baseline"] == "stable diffusion 2":
            return True
        return False

    @override
    def is_model_available(self, model_name):
        found_model_name = self.fuzzy_find_lora_key(model_name)
        if found_model_name is None:
            return False
        self._touch_lora(found_model_name)
        return True

    def is_lora_available(self, lora_name: str, timeout=45, is_version: bool = False):
        if is_version:
            return self.ensure_is_version(lora_name) in self._index_version_ids
        found_model_name = self.fuzzy_find_lora_key(lora_name)
        if found_model_name and self._check_for_refresh(lora_name):
            self.wait_for_downloads(timeout)
        return self.is_model_available(lora_name)
