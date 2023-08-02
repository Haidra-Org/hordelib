import copy
import glob
import hashlib
import json
import os
import re
import threading
import time
import typing
from collections import deque
from datetime import datetime, timedelta
from enum import auto

import requests
from fuzzywuzzy import fuzz
from horde_model_reference import LEGACY_REFERENCE_FOLDER
from horde_model_reference.path_consts import get_model_reference_filename
from loguru import logger
from strenum import StrEnum
from typing_extensions import override

from hordelib.consts import MODEL_CATEGORY_NAMES
from hordelib.model_manager.base import BaseModelManager
from hordelib.utils.sanitizer import Sanitizer


class DOWNLOAD_SIZE_CHECK(StrEnum):
    everything = auto()
    top = auto()
    adhoc = auto()


TESTS_ONGOING = os.getenv("TESTS_ONGOING", "0") == "1"


class LoraModelManager(BaseModelManager):
    LORA_DEFAULTS = "https://raw.githubusercontent.com/Haidra-Org/AI-Horde-image-model-reference/main/lora.json"
    LORA_API = "https://civitai.com/api/v1/models?types=LORA&sort=Highest%20Rated&primaryFileOnly=true"
    MAX_RETRIES = 10 if not TESTS_ONGOING else 1
    MAX_DOWNLOAD_THREADS = 3  # max concurrent downloads
    RETRY_DELAY = 5  # seconds
    REQUEST_METADATA_TIMEOUT = 30  # seconds
    REQUEST_DOWNLOAD_TIMEOUT = 300  # seconds
    THREAD_WAIT_TIME = 2

    def __init__(
        self,
        download_reference=False,
        allowed_top_lora_storage=10240 if not TESTS_ONGOING else 1024,
        allowed_adhoc_lora_storage=1024,
        download_wait=False,
    ):
        self._max_top_disk = allowed_top_lora_storage

        self.max_adhoc_disk = allowed_adhoc_lora_storage
        self._data = None
        self._next_page_url = None
        self._mutex = threading.Lock()
        self._file_count = 0
        self._download_threads = {}
        self._download_queue = deque()
        self._thread = None
        self.stop_downloading = True
        # Not yet handled, as we need a global reference to search through.
        self._previous_model_reference = {}
        self._adhoc_loras = set()
        self._download_wait = download_wait
        # If false, this MM will only download SFW loras
        self.nsfw = True
        self._adhoc_reset_thread = None
        self._stop_all_threads = False
        self._index_ids = {}
        self._index_orig_names = {}

        super().__init__(
            model_category_name=MODEL_CATEGORY_NAMES.lora,
            download_reference=download_reference,
        )
        # FIXME (shift lora.json handling into horde_model_reference?)
        self.models_db_path = LEGACY_REFERENCE_FOLDER.joinpath("lora.json").resolve()

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
            os.makedirs(self.modelFolderPath, exist_ok=True)
            self.download_model_reference()
            logger.info("Lora reference download begun asynchronously.")
        else:
            if self.models_db_path.exists():
                try:
                    self.model_reference = json.loads((self.models_db_path).read_text())
                    logger.info("Loaded model reference from disk.")
                except json.JSONDecodeError:
                    logger.error(f"Could not load {self.models_db_name} model reference from disk! Bad JSON?")
                    self.model_reference = {}
            else:
                logger.error(f"Could not load {self.models_db_name} model reference from disk! File not found.")
                self.model_reference = {}

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
            json_ret = self._get_json(self.LORA_DEFAULTS)
            if not json_ret:
                logger.error("Could not download default LoRas reference!")
            self._add_lora_ids_to_download_queue(json_ret)

        except Exception as err:
            logger.error(f"_get_lora_defaults() raised {err}")
            raise err

    def _add_lora_ids_to_download_queue(self, lora_ids, adhoc=False, version_compare=None):
        idsq = "&ids=".join([str(id) for id in lora_ids])
        url = f"https://civitai.com/api/v1/models?limit=100&ids={idsq}"
        data = self._get_json(url)
        if not data:
            logger.warning(f"metadata for LoRa {lora_ids} could not be downloaded!")
            return
        for lora_data in data.get("items", []):
            lora = self._parse_civitai_lora_data(lora_data, adhoc=adhoc)
            # If we're comparing versions, then we don't download if the existing lora metadata matches
            # Instead we just refresh metadata information
            if not lora:
                continue
            if version_compare and lora["version_id"] == version_compare:
                logger.debug(
                    f"Downloaded metadata for LoRa {lora['name']} "
                    f"('{lora['name']}') and found version match! Refreshing metadata.",
                )
                lora["last_checked"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self._add_lora_to_reference(lora)
                continue
            logger.debug(f"Downloaded metadata for LoRas {lora['id']} ('{lora['name']}') and added to download queue")
            self._download_lora(lora)

    def _get_json(self, url):
        retries = 0
        while retries <= self.MAX_RETRIES:
            try:
                response = requests.get(url, timeout=self.REQUEST_METADATA_TIMEOUT)
                response.raise_for_status()
                # Attempt to decode the response to JSON
                return response.json()

            except (requests.HTTPError, requests.ConnectionError, requests.Timeout, json.JSONDecodeError):
                retries += 1
                if retries <= self.MAX_RETRIES:
                    time.sleep(self.RETRY_DELAY)
                else:
                    # Max retries exceeded, give up
                    return None

            except Exception as e:
                # Failed badly
                logger.error(f"url '{url}' download failed {e}")
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
            "sha256": "",
            "filename": "",
            "id": "",
            "url": "",
            "triggers": [],
            "size_mb": 0,
            "adhoc": adhoc,
            "nsfw": False,
        }
        # get top version
        try:
            version = item.get("modelVersions", {})[0]
        except IndexError:
            version = {}
        # Get model triggers
        triggers = version.get("trainedWords", [])
        # get first file that is a primary file and a safetensor
        for file in version.get("files", {}):
            if file.get("primary", False) and file.get("name", "").endswith(".safetensors"):
                lora["name"] = Sanitizer.sanitise_model_name(item.get("name", ""))
                lora["orig_name"] = item.get("name", "")
                lora["id"] = item.get("id", 0)
                lora["filename"] = f'{Sanitizer.sanitise_filename(lora["name"])}.safetensors'
                lora["sha256"] = file.get("hashes", {}).get("SHA256")
                try:
                    lora["size_mb"] = round(file.get("sizeKB", 0) / 1024)
                except TypeError:
                    lora["size_mb"] = 144  # guess common case of 144MB, it's not critical here
                lora["url"] = file.get("downloadUrl", "")
                lora["triggers"] = triggers
                lora["nsfw"] = item.get("nsfw", True)
                lora["baseModel"] = version.get("baseModel", "SD 1.5")
                lora["version_id"] = version.get("id", 0)
                break
        # If we don't have everything required, fail
        if lora["adhoc"] and not lora.get("sha256"):
            logger.debug(f"Rejecting LoRa {lora.get('name')} because it doesn't have a sha256")
            return
        if not lora.get("filename") or not lora.get("url"):
            logger.debug(f"Rejecting LoRa {lora.get('name')} because it doesn't have a url")
            return
        # We don't want to start downloading GBs of a single LoRa.
        # We just ignore anything over 150Mb. Them's the breaks...
        if lora["adhoc"] and lora["size_mb"] > 220:
            logger.debug(f"Rejecting LoRa {lora.get('name')} because its size is over 220Mb.")
            return
        if lora["adhoc"] and lora["nsfw"] and not self.nsfw:
            logger.debug(f"Rejecting LoRa {lora.get('name')} because worker is SFW.")
            return
        # Fixup A1111 centric triggers
        for i, trigger in enumerate(lora["triggers"]):
            if re.match("<lora:(.*):.*>", trigger):
                lora["triggers"][i] = re.sub("<lora:(.*):.*>", "\\1", trigger)
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
                self._download_threads[thread_number]["lora"] = None
                time.sleep(self.THREAD_WAIT_TIME)
                continue
            # Download the lora
            retries = 0
            while retries <= self.MAX_RETRIES:
                try:
                    # Just before we download this file, check if we already have it
                    filename = os.path.join(self.modelFolderPath, lora["filename"])
                    hashfile = f"{os.path.splitext(filename)[0]}.sha256"
                    if os.path.exists(filename) and os.path.exists(hashfile):
                        # Check the hash
                        with open(hashfile) as infile:
                            try:
                                hashdata = infile.read().split()[0]
                            except (IndexError, OSError, IOError, PermissionError):
                                hashdata = ""
                        if not lora.get("sha256") or hashdata.lower() == lora["sha256"].lower():
                            # we already have this lora, consider it downloaded
                            # the SHA256 might not exist when the lora has been selected in the curation list
                            # Where we allow them to skip it
                            if not lora.get("sha256"):
                                logger.debug(
                                    f"Already have LORA {lora['filename']}. "
                                    "Bypassing SHA256 check as there's none stored",
                                )
                            else:
                                logger.debug(f"Already have LORA {lora['filename']}")
                            with self._mutex:
                                # We store as lower to allow case-insensitive search
                                lora["last_checked"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                self._add_lora_to_reference(lora)
                                if self.is_default_cache_full():
                                    self.stop_downloading = True
                                self.save_cached_reference_to_disk()
                            break

                    logger.info(f"Starting download of LORA {lora['filename']}")
                    response = requests.get(lora["url"], timeout=self.REQUEST_DOWNLOAD_TIMEOUT)
                    response.raise_for_status()
                    # Check the data hash
                    hash_object = hashlib.sha256()
                    hash_object.update(response.content)
                    sha256 = hash_object.hexdigest()
                    if not lora.get("sha256") or sha256.lower() == lora["sha256"].lower():
                        # wow, we actually got a valid file, save it
                        with open(filename, "wb") as outfile:
                            outfile.write(response.content)
                        # Save the hash file
                        with open(hashfile, "wt") as outfile:
                            outfile.write(f"{sha256} *{lora['filename']}")

                        # Shout about it
                        logger.info(f"Downloaded LORA {lora['filename']} ({lora['size_mb']} MB)")
                        # Maybe we're done
                        with self._mutex:
                            # We store as lower to allow case-insensitive search
                            lora["last_used"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            lora["last_checked"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            self._add_lora_to_reference(lora)
                            if self.is_adhoc_cache_full():
                                self.delete_oldest_lora()
                            self.save_cached_reference_to_disk()
                        break
                    else:
                        # We will retry
                        logger.debug(
                            f"Downloaded LORA file for {lora['filename']} didn't match hash. "
                            f"Retry {retries}/{self.MAX_RETRIES}",
                        )

                except (requests.HTTPError, requests.ConnectionError, requests.Timeout, json.JSONDecodeError) as e:
                    # We will retry
                    logger.debug(f"Error downloading {lora['filename']} {e}. Retry {retries}/{self.MAX_RETRIES}")

                except Exception as e:
                    # Failed badly, ignore and retry
                    logger.debug(f"Fatal error downloading {lora['filename']} {e}. Retry {retries}/{self.MAX_RETRIES}")

                retries += 1
                if retries > self.MAX_RETRIES:
                    break  # fail

                time.sleep(self.RETRY_DELAY)

    def _download_lora(self, lora):
        with self._mutex:
            # Start download threads if they aren't already started
            while len(self._download_threads) < self.MAX_DOWNLOAD_THREADS:
                thread_iter = len(self._download_threads)
                thread = threading.Thread(target=self._download_thread, daemon=True, args=(thread_iter,))
                self._download_threads[thread_iter] = {"thread": thread, "lora": None}
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
        lora_key = lora["name"].lower().strip()
        if lora.get("adhoc", False):
            self._adhoc_loras.add(lora_key)
            # Once added to our set, we don't need to specify it was adhoc anymore
            del lora["adhoc"]
        self.model_reference[lora_key] = lora
        self._index_ids[lora["id"]] = lora_key
        orig_name = lora.get("orig_name", lora["name"]).lower()
        self._index_orig_names[orig_name] = lora_key

    def download_default_loras(self, nsfw=True, timeout=None):
        """Start up a background thread downloading and return immediately"""
        # Don't start if we're already busy doing something
        if not self.are_downloads_complete():
            return
        self.nsfw = nsfw
        self._previous_model_reference = copy.deepcopy(self.model_reference)
        # TODO: Avoid clearing this out, until we know CivitAI is not dead.
        self.model_reference = {}
        os.makedirs(self.modelFolderPath, exist_ok=True)
        # Start processing in a background thread
        self._thread = threading.Thread(target=self._get_lora_defaults, daemon=True)
        self._thread.start()
        # Wait for completion of our threads if requested
        if self._download_wait:
            self.wait_for_downloads(timeout)
        if self.is_adhoc_reset_complete():
            self._adhoc_reset_thread = threading.Thread(target=self.reset_adhoc_loras, daemon=True)
            self._adhoc_reset_thread.start()

    def wait_for_downloads(self, timeout=None):
        rtr = 0
        while not self.are_downloads_complete():
            time.sleep(0.5)
            rtr += 0.5
            if timeout and rtr > timeout:
                raise Exception(f"Lora downloads exceeded specified timeout ({timeout})")

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

    def fuzzy_find_lora_key(self, lora_name):
        # sname = Sanitizer.remove_version(lora_name).lower()
        if type(lora_name) is int or lora_name.isdigit():
            if int(lora_name) in self._index_ids:
                return self._index_ids[int(lora_name)]
            return None
        sname = lora_name.lower().strip()
        if sname in self.model_reference:
            return sname
        if sname in self._index_orig_names:
            return self._index_orig_names[sname]
        if Sanitizer.has_unicode(sname):
            for lora in self._index_orig_names:
                if sname in lora:
                    return self._index_orig_names[lora]
            # If a unicode name is not found in the orig_names index
            # it won't be found anywhere else, as unicode chars are converted to ascii in the keys
            # This saves us time doing unnecessary fuzzy searches
            return None
        for lora in self.model_reference:
            if sname in lora:
                return lora
        for lora in self.model_reference:
            if fuzz.ratio(sname, lora) > 80:
                return lora
        return None

    # Using `get_model` instead of `get_lora` as it exists in the base class
    def get_model(self, model_name: str) -> dict | None:
        """Returns the actual lora details dict for the specified model_name search string
        Returns None if lora name not found"""
        lora_name = self.fuzzy_find_lora_key(model_name)
        if not lora_name:
            return None
        return self.model_reference.get(lora_name)

    def get_lora_filename(self, model_name: str):
        """Returns the actual lora filename for the specified model_name search string
        Returns None if lora name not found"""
        lora = self.get_model(model_name)
        if not lora:
            return None
        return lora["filename"]

    def get_lora_name(self, model_name: str):
        """Returns the actual lora name for the specified model_name search string
        Returns None if lora name not found"""
        lora = self.get_model(model_name)
        if not lora:
            return None
        return lora["name"]

    def get_lora_triggers(self, model_name: str):
        """Returns a list of triggers for a specified lora name
        Returns an empty list if no triggers are found
        Returns None if lora name not found"""
        lora = self.get_model(model_name)
        if not lora:
            return None
        triggers = lora.get("triggers")
        if triggers:
            return triggers
        # We don't `return lora.get("triggers", [])`` to avoid the returned list object being modified
        # and then we keep returning previous items
        return []

    def find_lora_trigger(self, model_name: str, trigger_search: str):
        """Searches for a specific trigger for a specified lora name
        Returns None if string not found even with fuzzy search"""
        triggers = self.get_lora_triggers(model_name)
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
        with open(self.models_db_path, "wt", encoding="utf-8", errors="ignore") as outfile:
            outfile.write(json.dumps(self.model_reference, indent=4))

    def calculate_downloaded_loras(self, mode=DOWNLOAD_SIZE_CHECK.everything):
        total_size = 0
        for lora in self.model_reference:
            if mode == DOWNLOAD_SIZE_CHECK.top and lora in self._adhoc_loras:
                continue
            if mode == DOWNLOAD_SIZE_CHECK.adhoc and lora not in self._adhoc_loras:
                continue
            total_size += self.model_reference[lora]["size_mb"]
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
            lora_datetime = datetime.strptime(self.model_reference[lora]["last_used"], "%Y-%m-%d %H:%M:%S")
            if not oldest_lora:
                oldest_lora = lora
                oldest_datetime = lora_datetime
                continue
            if oldest_datetime and oldest_datetime > lora_datetime:
                oldest_lora = lora
                oldest_datetime = lora_datetime
        return oldest_lora

    def delete_oldest_lora(self):
        oldest_lora = self.find_oldest_adhoc_lora()
        if not oldest_lora:
            return
        self.delete_lora(oldest_lora)

    def find_lora_from_filename(self, filename: str):
        for lora in self.model_reference:
            if self.model_reference[lora]["filename"] == filename:
                return lora
        return None

    def find_unused_loras(self):
        files = glob.glob(f"{self.modelFolderPath}/*.safetensors")
        filesnames = set()
        for stfile in files:
            filename = os.path.basename(stfile)
            if not self.find_lora_from_filename(filename):
                filesnames.add(filename)
        return filesnames

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
        filename = os.path.join(self.modelFolderPath, lora_filename)
        if not os.path.exists(filename):
            logger.warning(f"Could not find LoRa file on disk to delete: {filename}")
            return
        os.remove(filename)
        logger.info(f"Deleted LoRa file: {filename}")

    def delete_lora(self, lora_name: str):
        lora_info = self.get_model(lora_name)
        if not lora_info:
            logger.warning(f"Could not find lora {lora_name} to delete")
            return
        self.delete_lora_files(lora_info["filename"])
        self._adhoc_loras.remove(lora_name)
        del self._index_ids[lora_info["id"]]
        del self._index_orig_names[lora_info["orig_name"].lower()]
        del self.model_reference[lora_name]
        self.save_cached_reference_to_disk()

    def ensure_lora_deleted(self, lora_name: str):
        lora_key = self.fuzzy_find_lora_key(lora_name)
        if not lora_key:
            return
        self.delete_lora(lora_key)

    def reset_adhoc_loras(self):
        """Compared the known loras from the previous run to the current one
        Adds any definitions as adhoc loras, until we have as many Mb as self.max_adhoc_disk"""
        while not self.are_downloads_complete():
            if self._stop_all_threads:
                logger.debug("Stopped processing thread")
                return
            time.sleep(0.2)
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._adhoc_loras = set()
        sorted_items = []
        try:
            sorted_items = sorted(
                self._previous_model_reference.items(),
                key=lambda x: x[1].get("last_used", now),
                reverse=True,
            )
        except Exception as err:
            logger.error(err)
        while not self.is_adhoc_cache_full() and len(sorted_items) > 0:
            prevlora_key, prevlora_value = sorted_items.pop()
            if prevlora_key in self.model_reference:
                continue
            # If False, it will initiates a redownload and call _add_lora_to_reference() later
            if self._check_for_refresh(prevlora_key):
                self._add_lora_to_reference(prevlora_value)
            self._adhoc_loras.add(prevlora_key)
        for lora_key in self.model_reference:
            if lora_key in self._previous_model_reference:
                self.model_reference[lora_key]["last_used"] = self._previous_model_reference[lora_key].get(
                    "last_used",
                    now,
                )
        # Final assurance that all our loras have a last_used timestamp
        for lora in self.model_reference.values():
            if "last_used" not in lora:
                lora["last_used"] = now
        self._previous_model_reference = {}
        self.save_cached_reference_to_disk()

    def _check_for_refresh(self, lora_name: str):
        """Returns True if a refresh is needed
        and also initiates a refresh
        Else returns False
        """
        lora_details = self.get_model(lora_name)
        if not lora_details:
            return True
        refresh = False
        if "last_checked" not in lora_details:
            refresh = True
        elif "baseModel" not in lora_details:
            refresh = True
        else:
            lora_datetime = datetime.strptime(lora_details["last_checked"], "%Y-%m-%d %H:%M:%S")
            if lora_datetime < datetime.now() - timedelta(days=1):
                refresh = True
        if refresh:
            logger.debug(f"Lora {lora_name} found needing refresh. Initiating metadata download...")
            self._add_lora_ids_to_download_queue([lora_details["id"]], lora_details.get("version_id", -1))
            return False
        return True

    # def check_for_valid

    def is_adhoc_reset_complete(self):
        if self._adhoc_reset_thread and self._adhoc_reset_thread.is_alive():
            return False
        return True

    def wait_for_adhoc_reset(self, timeout=15):
        rtr = 0
        while not self.is_adhoc_reset_complete():
            time.sleep(0.2)
            rtr += 0.2
            if timeout and rtr > timeout:
                raise Exception(f"Lora adhoc reset exceeded specified timeout ({timeout})")

    def stop_all(self):
        self._stop_all_threads = True

    def touch_lora(self, lora_name):
        """Updates the "last_used" key in a lora entry to current UTC time"""
        lora = self.get_model(lora_name)
        if not lora:
            logger.warning(f"Could not find lora {lora_name} to touch")
            return
        lora["last_used"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def get_lora_last_use(self, lora_name):
        """Returns a dateimte object based on the "last_used" key in a lora entry"""
        lora = self.get_model(lora_name)
        if not lora:
            logger.warning(f"Could not find lora {lora_name} to get last use")
            return None
        return datetime.strptime(lora["last_used"], "%Y-%m-%d %H:%M:%S")

    def fetch_adhoc_lora(self, lora_name, timeout=45):
        if type(lora_name) is int or lora_name.isdigit():
            url = f"https://civitai.com/api/v1/models/{lora_name}"
        else:
            url = f"{self.LORA_API}&nsfw={str(self.nsfw).lower()}&query={lora_name}"
        data = self._get_json(url)
        # CivitAI down
        if not data:
            return None
        if "items" in data:
            if len(data["items"]) == 0:
                return None
            lora = self._parse_civitai_lora_data(data["items"][0], adhoc=True)
        else:
            lora = self._parse_civitai_lora_data(data, adhoc=True)
        # For example epi_noiseoffset doesn't have sha256 so we ignore it
        # This avoid us faulting
        if not lora:
            return None
        # We double-check that somehow our search missed it but CivitAI searches differently and found it
        fuzzy_find = self.fuzzy_find_lora_key(lora["id"])
        if fuzzy_find:
            logger.error(fuzzy_find)
            return fuzzy_find
        self._download_queue.append(lora)
        # We need to wait a bit to make sure the threads pick up the download
        time.sleep(self.THREAD_WAIT_TIME)
        self.wait_for_downloads(timeout)
        return lora["name"].lower()

    def do_baselines_match(self, lora_name, model_details):
        self._check_for_refresh(lora_name)
        lota_details = self.get_model(lora_name)
        if not lota_details:
            logger.warning(f"Could not find lora {lora_name} to check baselines")
            return False
        if "SD 1.5" in lota_details["baseModel"] and model_details["baseline"] == "stable diffusion 1":
            return True
        if "SD 2.1" in lota_details["baseModel"] and model_details["baseline"] == "stable diffusion 2":
            return True
        return False

    @override
    def is_local_model(self, model_name):
        return self.fuzzy_find_lora_key(model_name) is not None

    @override
    def load(
        self,
        model_name: str,
        *,
        half_precision: bool = True,
        gpu_id: int | None = 0,
        cpu_only: bool = False,
        **kwargs,
    ) -> bool | None:
        error = "load is not supported for LoRas"
        logger.error(error)
        raise NotImplementedError(error)

    @override
    def modelToRam(
        self,
        model_name: str,
        **kwargs,
    ) -> dict[str, typing.Any]:
        error = "modelToRam is not supported for LoRas"
        logger.error(error)
        raise NotImplementedError(error)

    def get_available_models(self):
        """
        Returns the available model names
        """
        return list(self.model_reference.keys())
