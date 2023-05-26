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
from datetime import datetime
from enum import Enum

import requests
from fuzzywuzzy import fuzz
from loguru import logger
from typing_extensions import override

from hordelib.consts import MODEL_CATEGORY_NAMES, MODEL_DB_NAMES, MODEL_FOLDER_NAMES
from hordelib.model_manager.base import BaseModelManager
from hordelib.utils.sanitizer import Sanitizer


class DOWNLOAD_SIZE_CHECK(str, Enum):
    everything = "everything"
    top = "top"
    adhoc = "adhoc"


class LoraModelManager(BaseModelManager):

    LORA_API = "https://civitai.com/api/v1/models?types=LORA&sort=Highest%20Rated&primaryFileOnly=true"
    MAX_RETRIES = 10
    MAX_DOWNLOAD_THREADS = 3  # max concurrent downloads
    RETRY_DELAY = 5  # seconds
    REQUEST_METADATA_TIMEOUT = 30  # seconds
    REQUEST_DOWNLOAD_TIMEOUT = 300  # seconds

    def __init__(
        self,
        download_reference=False,
        allowed_top_lora_storage=10240,
        allowed_adhoc_lora_storage=1024,
        download_wait=False,
    ):

        self._max_top_disk = allowed_top_lora_storage
        self._max_adhoc_disk = allowed_adhoc_lora_storage
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
        self._adhoc_mutex = {}
        self._download_wait = download_wait
        # If false, this MM will only download SFW loras
        self.nsfw = True
        self._adhoc_reset_thread = None
        self._stop_all_threads = False

        # Example of how to inject mandatory LORAs, we use these two for our tests
        # We need to ensure their format is the same as after they are returned _parse_civitai_lora_data
        # i.e. without versions in their names.
        self._download_queue.append(
            {
                "name": "GlowingRunesAI",
                "sha256": "1E7DF5F25B76B3D1B5FCEBB7AEB229FF33D805DC10B2F7CBB56F3A7BA0ED4686",
                "filename": "GlowingRunesAI.safetensors",
                "url": "https://civitai.com/api/download/models/75193?type=Model&format=SafeTensor",
                "triggers": ["GlowingRunesAIV3_green", "GlowingRunesAI_red", "GlowingRunesAI_paleblue"],
                "size_mb": 144,
            },
        )
        self._download_queue.append(
            {
                "name": "Dra9onScaleAI",
                "sha256": "E562FC8EE097774E2C6A48AA9F279DB78AE4D1BFE14EF52F6AA76450C188B92B",
                "filename": "Dra9onScaleAI.safetensors",
                "url": "https://civitai.com/api/download/models/70189?type=Model&format=SafeTensor",
                "triggers": ["Dr490nSc4leAI"],
                "size_mb": 144,
            },
        )

        super().__init__(
            modelFolder=MODEL_FOLDER_NAMES[MODEL_CATEGORY_NAMES.lora],
            models_db_name=MODEL_DB_NAMES[MODEL_CATEGORY_NAMES.lora],
            download_reference=download_reference,
        )

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
            try:
                self.model_reference = json.loads((self.models_db_path).read_text())
            except FileNotFoundError:
                self.model_reference = {}
            logger.info(
                " ".join(
                    [
                        "Loaded model reference from disk.",
                        f"Got {len(self.model_reference)} models for {self.models_db_name}.",
                    ],
                ),
            )

    def download_model_reference(self):
        # We have to wipe it, as we are going to be adding it it instead of replacing it
        # We're not downloading now, as we need to be able to init without it
        self.model_reference = {}
        self.save_cached_reference_to_disk()

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
                logger.error(f"LORA download failed {e}")
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

    def _parse_civitai_lora_data(self, item):
        """Return a simplified dictionary with the information we actually need about a lora"""
        lora = {
            "name": "",
            "sha256": "",
            "filename": "",
            "id": "",
            "url": "",
            "triggers": [],
            "size_mb": 0,
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
                lora["filename"] = f'{Sanitizer.sanitise_filename(lora["name"])}.safetensors'
                lora["sha256"] = file.get("hashes", {}).get("SHA256")
                try:
                    lora["size_mb"] = round(file.get("sizeKB", 0) / 1024)
                except TypeError:
                    lora["size_mb"] = 144  # guess common case of 144MB, it's not critical here
                lora["url"] = file.get("downloadUrl", "")
                lora["triggers"] = triggers
                break
        # If we don't have everything required, fail
        if not lora.get("sha256") or not lora.get("filename") or not lora.get("url") or not lora.get("triggers"):
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
                time.sleep(2)
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
                        if hashdata.lower() == lora["sha256"].lower():
                            # we already have this lora, consider it downloaded
                            logger.debug(f"Already have LORA {lora['filename']}")
                            with self._mutex:
                                # We store as lower to allow case-insensitive search
                                self.model_reference[lora["name"].lower()] = lora
                                if self.is_default_cache_full():
                                    self.stop_downloading = True
                                else:
                                    # Normally this should never happen unless the user manually
                                    # reduced their max size in the meantime
                                    if self.is_adhoc_cache_full():
                                        self.delete_oldest_lora()
                                self.save_cached_reference_to_disk()
                            break

                    logger.info(f"Starting download of LORA {lora['filename']}")
                    response = requests.get(lora["url"], timeout=self.REQUEST_DOWNLOAD_TIMEOUT)
                    response.raise_for_status()
                    # Check the data hash
                    hash_object = hashlib.sha256()
                    hash_object.update(response.content)
                    sha256 = hash_object.hexdigest()
                    if sha256.lower() == lora["sha256"].lower():
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
                            self.model_reference[lora["name"].lower()] = lora
                            if len(self._adhoc_mutex) == 0:
                                if self.is_default_cache_full():
                                    self.stop_downloading = True
                            else:
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
        self._thread = threading.Thread(target=self._start_processing, daemon=True)
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
        return self.stop_downloading

    def are_download_threads_idle(self):
        # logger.debug([dthread["lora"] for dthread in self._download_threads.values()])
        for dthread in self._download_threads.values():
            if dthread["lora"] is not None:
                return False
        return True

    def fuzzy_find_lora(self, lora_name):
        # sname = Sanitizer.remove_version(lora_name).lower()
        sname = lora_name.lower()
        if sname in self.model_reference:
            return sname
        for lora in self.model_reference:
            if sname in lora:
                return lora
        for lora in self.model_reference:
            if fuzz.ratio(sname, lora) > 80:
                return lora
        return None

    # Using `get_model` instead of `get_lora` as it exists in the base class
    def get_model(self, model_name: str):
        """Returns the actual lora details dict for the specified model_name search string
        Returns None if lora name not found"""
        lora_name = self.fuzzy_find_lora(model_name)
        if not lora_name:
            return None
        return self.model_reference[lora_name]

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
        logger.debug([self.calculate_adhoc_loras_cache(), self._max_adhoc_disk])
        return self.calculate_adhoc_loras_cache() >= self._max_adhoc_disk

    def calculate_download_queue(self):
        total_queue = 0
        for lora in self._download_queue:
            total_queue += lora["size_mb"]
        return total_queue

    def find_oldest_adhoc_lora(self):
        oldest_lora: str = None
        oldest_datetime: datetime = None
        for lora in self._adhoc_loras:
            lora_datetime = datetime.strptime(self.model_reference[lora]["last_used"], "%Y-%m-%d %H:%M:%S")
            if not oldest_lora:
                oldest_lora = lora
                oldest_datetime = lora_datetime
                continue
            if oldest_datetime > lora_datetime:
                oldest_lora = lora
                oldest_datetime = lora_datetime
        return oldest_lora

    def delete_oldest_lora(self):
        oldest_lora = self.find_oldest_adhoc_lora()
        if not oldest_lora:
            return
        filename = os.path.join(self.modelFolderPath, self.model_reference[oldest_lora]["filename"])
        os.remove(filename)
        del self.model_reference[oldest_lora]
        del self._adhoc_loras[oldest_lora]

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
        for ulora in loras_to_delete:
            filename = os.path.join(self.modelFolderPath, ulora)
            os.remove(filename)
            logger.info(f"Deleted unused LoRa file: {filename}")
        return loras_to_delete

    def reset_adhoc_loras(self):
        """Compared the known loras from the previous run to the current one
        Adds any definitions as adhoc loras, until we have as many Mb as self._max_adhoc_disk"""
        while not self.are_downloads_complete():
            if self._stop_all_threads:
                logger.debug("Stopped processing thread")
                return
            time.sleep(0.2)
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._adhoc_loras = set()
        try:
            sorted_items = sorted(
                self._previous_model_reference.items(),
                key=lambda x: x[1].get("last_used", now),
                reverse=True,
            )
        except Exception as err:
            logger.error(err)
        while not self.is_adhoc_cache_full():
            prevlora_key, prevlora_value = sorted_items.pop()
            if prevlora_key in self.model_reference:
                continue
            self.model_reference[prevlora_key] = prevlora_value
            self._adhoc_loras.add(prevlora_key)
        for lora_key in self.model_reference:
            if lora_key in self._previous_model_reference:
                self.model_reference[lora_key]["last_used"] = self._previous_model_reference[lora_key].get(
                    "last_used",
                    now,
                )
        self._previous_model_reference = {}
        self.save_cached_reference_to_disk()

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
        lora["last_used"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def get_lora_last_use(self, lora_name):
        """Returns a dateimte object based on the "last_used" key in a lora entry"""
        lora = self.get_model(lora_name)
        return datetime.strptime(lora["last_used"], "%Y-%m-%d %H:%M:%S")

    @override
    def is_local_model(self, model_name):
        return self.fuzzy_find_lora(model_name) is not None

    @override
    def modelToRam(
        self,
        model_name: str,
        **kwargs,
    ) -> dict[str, typing.Any]:
        pass
