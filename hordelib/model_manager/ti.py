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

import requests
from fuzzywuzzy import fuzz
from horde_model_reference import LEGACY_REFERENCE_FOLDER
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


class TextualInversionModelManager(BaseModelManager):
    TI_API: str = "https://civitai.com/api/v1/models?types=TextualInversion&sort=Highest%20Rated&primaryFileOnly=true"
    HORDELING_API: str = "https://hordeling.aihorde.net/api/v1/embedding"
    MAX_RETRIES: int = 10 if not TESTS_ONGOING else 3
    MAX_DOWNLOAD_THREADS: int = 3
    """The number of threads to use for downloading (the max number of concurrent downloads)."""
    RETRY_DELAY: float = 3 if not TESTS_ONGOING else 0.2
    """The time to wait between retries in seconds"""
    REQUEST_METADATA_TIMEOUT: int = 30
    """The time to wait for a metadata request to complete in seconds"""
    REQUEST_DOWNLOAD_TIMEOUT: int = 30
    """The time to wait for a download request to complete in seconds"""
    THREAD_WAIT_TIME: int = 2
    """The time to wait between checking the download queue in seconds"""

    _file_lock: multiprocessing_lock | nullcontext

    def __init__(
        self,
        download_reference=False,
        multiprocessing_lock: multiprocessing_lock | None = None,
        civitai_api_token: str | None = None,
        **kwargs,
    ):
        self._data = None
        self._next_page_url = None
        self._file_lock = multiprocessing_lock or nullcontext()
        self._mutex = threading.Lock()
        self._file_count = 0
        self._download_threads = {}  # type: ignore # FIXME: add type
        self._download_queue = deque()  # type: ignore # FIXME: add type
        self._thread = None
        self.stop_downloading = True
        # Not yet handled, as we need a global reference to search through.
        self._previous_model_reference = {}  # type: ignore # FIXME: add type
        self._adhoc_tis = set()  # type: ignore # FIXME: add type
        # If false, this MM will only download SFW tis
        self.nsfw = True
        self._adhoc_reset_thread = None
        self._stop_all_threads = False
        self._index_ids = {}  # type: ignore # FIXME: add type
        self._index_orig_names = {}  # type: ignore # FIXME: add type
        self.total_retries_attempted = 0

        models_db_path = LEGACY_REFERENCE_FOLDER.joinpath("ti.json").resolve()
        super().__init__(
            model_category_name=MODEL_CATEGORY_NAMES.ti,
            download_reference=download_reference,
            models_db_path=models_db_path,
            civitai_api_token=civitai_api_token,
        )

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

        # TI are always stored to disk and the model reference created slowly through ad-hoc requests
        os.makedirs(self.model_folder_path, exist_ok=True)
        if self.models_db_path.exists():
            try:
                self.model_reference = json.loads((self.models_db_path).read_text())

                for ti in self.model_reference.values():
                    self._index_ids[ti["id"]] = ti["name"].lower().strip()
                    orig_name = ti.get("orig_name", ti["name"]).lower().strip()
                    self._index_orig_names[orig_name] = ti["name"].lower().strip()

                logger.info("Loaded model reference from disk.")
            except json.JSONDecodeError:
                logger.error(f"Could not load {self.models_db_name} model reference from disk! Bad JSON?")
                self.model_reference = {}
                self.save_cached_reference_to_disk()
        else:
            logger.info(f"Initiating new model reference {self.models_db_name} model reference to disk.")
            self.model_reference = {}
            self.save_cached_reference_to_disk()

    def download_model_reference(self):
        # We have to wipe it, as we are going to be adding it it instead of replacing it
        # We're not downloading now, as we need to be able to init without it
        self.model_reference = {}
        self.save_cached_reference_to_disk()

    def _add_ti_ids_to_download_queue(self, ti_ids, adhoc=False, version_compare=None):
        idsq = "&ids=".join([str(id) for id in ti_ids])
        url = f"https://civitai.com/api/v1/models?limit=100&ids={idsq}"
        data = self._get_json(url)
        if not data:
            logger.warning(f"metadata for Textual Inversion {ti_ids} could not be downloaded!")
            return
        for ti_data in data.get("items", []):
            ti = self._parse_civitai_ti_data(ti_data, adhoc=adhoc)
            # If we're comparing versions, then we don't download if the existing ti metadata matches
            # Instead we just refresh metadata information
            if not ti:
                continue
            if version_compare and ti["version_id"] == version_compare:
                logger.debug(
                    f"Downloaded metadata for Textual Inversion {ti['name']} "
                    f"('{ti['name']}') and found version match! Refreshing metadata.",
                )
                ti["last_checked"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self._add_ti_to_reference(ti)
                continue
            logger.debug(
                f"Downloaded metadata for Textual Inversions {ti['id']} ('{ti['name']}') and added to download queue",
            )
            self._download_ti(ti)

    def _get_json(self, url):
        retries = 0
        while retries <= self.MAX_RETRIES:
            response = None
            try:
                response = requests.get(url, timeout=self.REQUEST_METADATA_TIMEOUT)
                response.raise_for_status()
                # Attempt to decode the response to JSON
                return response.json()

            except (requests.HTTPError, requests.ConnectionError, requests.Timeout, json.JSONDecodeError):
                # CivitAI Errors when the model ID is too long
                if response is not None:
                    if response.status_code in [401, 404]:
                        return None
                    if response.status_code == 500:
                        retries += 3
                        logger.debug(
                            "CivitAI reported an internal error when downloading metadata. "
                            "Fewer retries will be attempted.",
                        )

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
            url = f"{self.TI_API}&nsfw={str(self.nsfw).lower()}"
        else:
            url = self._next_page_url

        # This may be the end of the road, unlikely but...
        if not url:
            logger.warning("End of Textual Inversion data reached")
            self.stop_downloading = True
        else:
            # Get the actual item data
            items = self._get_json(url)
            if items:
                self._data = items
                self._next_page_url = self._data.get("metadata", {}).get("nextPage", "")
            else:
                # We failed to get more items
                logger.error("Failed to download all Textual Inversion data even after retries.")
                self._data = None
                self._next_page_url = None  # give up

    def _parse_civitai_ti_data(self, item, adhoc=False):
        """Return a simplified dictionary with the information we actually need about a ti"""
        ti = {
            "name": "",
            "orig_name": "",
            "sha256": "",
            "filename": "",
            "id": "",
            "url": "",
            "triggers": [],
            "size_kb": 0,
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
        # logger.debug(json.dumps(version,indent=4))
        for file in version.get("files", {}):
            if file.get("primary", False):
                ti["name"] = Sanitizer.sanitise_model_name(item.get("name", ""))
                ti["orig_name"] = item.get("name", "")
                ti["id"] = item.get("id", 0)
                ti["filename"] = f'{ti["id"]}.safetensors'
                ti["sha256"] = file.get("hashes", {}).get("SHA256")
                try:
                    ti["size_kb"] = round(file.get("sizeKB", 0))
                except TypeError:
                    ti["size_kb"] = 24  # guess common case of 24Kb, it's not critical here
                ti["url"] = file.get("downloadUrl", "")
                ti["triggers"] = triggers
                ti["nsfw"] = item.get("nsfw", True)
                ti["baseModel"] = version.get("baseModel", "SD 1.5")
                ti["version_id"] = version.get("id", 0)
                break
        # If we don't have everything required, fail
        if ti["adhoc"] and not ti.get("sha256"):
            logger.debug(f"Rejecting Textual Inversion {ti.get('name')} because it doesn't have a sha256")
            return None
        if not ti.get("filename") or not ti.get("url"):
            logger.debug(f"Rejecting Textual Inversion {ti.get('name')} because it doesn't have a url")
            return None
        # We don't want to start downloading GBs of a single Textual Inversion.
        # We just ignore anything over 150Mb. Them's the breaks...
        if ti["adhoc"] and ti["size_kb"] > 20000:
            logger.debug(f"Rejecting Textual Inversion {ti.get('name')} because its size is over 20Mb.")
            return None
        if ti["adhoc"] and ti["nsfw"] and not self.nsfw:
            logger.debug(f"Rejecting Textual Inversion {ti.get('name')} because worker is SFW.")
            return None
        # Fixup A1111 centric triggers
        for i, trigger in enumerate(ti["triggers"]):
            if re.match("<ti:(.*):.*>", trigger):
                ti["triggers"][i] = re.sub("<ti:(.*):.*>", "\\1", trigger)
        return ti

    def _download_thread(self, thread_number):
        # We try to download the Textual Inversion. There are tens of thousands of these things, we aren't
        # picky if downloads fail, as they often will if civitai is the host, we just move on to
        # the next one
        logger.debug(f"Started Download Thread {thread_number}")
        while True:
            # Endlessly look for files to download and download them
            if self._stop_all_threads:
                logger.debug(f"Stopped Download Thread {thread_number}")
                return
            try:
                ti = self._download_queue.popleft()
                self._download_threads[thread_number]["ti"] = ti
            except IndexError:
                # Nothing in the queue
                self._download_threads[thread_number]["ti"] = None
                time.sleep(self.THREAD_WAIT_TIME)
                continue
            # Download the ti
            retries = 0
            while retries <= self.MAX_RETRIES:
                try:
                    # Just before we download this file, check if we already have it
                    filepath = os.path.join(self.model_folder_path, ti["filename"])
                    hashpath = f"{os.path.splitext(filepath)[0]}.sha256"
                    logger.debug(f"Retrieving TI metadata from Hordeling for ID: {ti['filename']}")
                    hordeling_response = requests.get(f"{self.HORDELING_API}/{ti['id']}", timeout=5)
                    if not hordeling_response.ok:
                        if hordeling_response.status_code == 404:
                            logger.debug(f"Textual Inversion: {ti['filename']} could not be found on AI Hordeling.")
                            break

                        if hordeling_response.status_code == 500:
                            retries += 2
                            logger.debug(
                                "AI Hordeing reported an internal error when downloading metadata. "
                                "Fewer retries will be attempted.",
                            )

                        hordeling_json = hordeling_response.json()
                        # We will retry
                        logger.debug(
                            "AI Hordeling reported error when downloading metadata "
                            f"for Textual Inversion: {ti['filename']}: "
                            f"{hordeling_json}"
                            f"Retry {retries}/{self.MAX_RETRIES}",
                        )

                        message = hordeling_json.get("message", "")
                        if message is not None and isinstance(message, str) and "hash" in message.lower():
                            logger.debug(f"Textual Inversion: {ti['filename']} hash mismatch reported.")
                            break

                    else:
                        hordeling_json = hordeling_response.json()
                        if hordeling_json.get("sha256"):
                            ti["sha256"] = hordeling_json["sha256"]
                        if os.path.exists(filepath) and os.path.exists(hashpath):
                            # Check the hash
                            with open(hashpath) as infile:
                                try:
                                    hashdata = infile.read().split()[0]
                                except (IndexError, OSError, PermissionError):
                                    hashdata = ""

                            if not ti.get("sha256") or hashdata.lower() == ti["sha256"].lower():
                                # we already have this ti, consider it downloaded
                                # the SHA256 might not exist when the ti has been selected in the curation list
                                # Where we allow them to skip it
                                if not ti.get("sha256"):
                                    logger.debug(
                                        f"Already have Textual Inversion: {ti['filename']}. "
                                        "Bypassing SHA256 check as there's none stored",
                                    )
                                else:
                                    logger.debug(f"Already have Textual Inversion: {ti['filename']}")
                                with self._mutex, self._file_lock:
                                    # We store as lower to allow case-insensitive search
                                    ti["last_checked"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                    self._add_ti_to_reference(ti)
                                    if self.is_default_cache_full():
                                        self.stop_downloading = True
                                    self.save_cached_reference_to_disk()
                                break

                        logger.info(f"Starting download of Textual Inversion: {ti['filename']}")

                        ti_url = hordeling_json["url"]
                        if self._civitai_api_token and self.is_model_url_from_civitai(ti_url):
                            ti_url += f"?token={self._civitai_api_token}"

                        response = requests.get(
                            ti_url,
                            timeout=self.REQUEST_DOWNLOAD_TIMEOUT,
                        )
                        response.raise_for_status()
                        # Check the data hash
                        hash_object = hashlib.sha256()
                        hash_object.update(response.content)
                        sha256 = hash_object.hexdigest()
                        if not ti.get("sha256") or sha256.lower() == ti["sha256"].lower():
                            # wow, we actually got a valid file, save it
                            with open(filepath, "wb") as outfile:
                                outfile.write(response.content)
                            # Save the hash file
                            with open(hashpath, "w") as outfile:
                                outfile.write(f"{sha256} *{ti['filename']}")

                            # Shout about it
                            logger.info(f"Downloaded Textual Inversion: {ti['filename']} ({ti['size_kb']} KB)")
                            # Maybe we're done
                            with self._mutex, self._file_lock:
                                # We store as lower to allow case-insensitive search
                                ti["last_used"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                ti["last_checked"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                self._add_ti_to_reference(ti)
                                if self.is_adhoc_cache_full():
                                    self.delete_oldest_ti()
                                self.save_cached_reference_to_disk()
                            break

                        # We will retry
                        logger.debug(
                            f"Downloaded Textual Inversion file {ti['filename']} didn't match hash. "
                            f"Retry {retries}/{self.MAX_RETRIES}",
                        )

                except (requests.HTTPError, requests.ConnectionError, requests.Timeout, json.JSONDecodeError) as e:
                    # We will retry
                    logger.debug(f"Error downloading {ti['filename']} {e}. Retry {retries}/{self.MAX_RETRIES}")

                except Exception as e:
                    # Failed badly, ignore and retry
                    logger.debug(f"Fatal error downloading {ti['filename']} {e}. Retry {retries}/{self.MAX_RETRIES}")

                retries += 1
                self.total_retries_attempted += 1
                if retries > self.MAX_RETRIES:
                    break  # fail

                time.sleep(self.RETRY_DELAY)

    def _download_ti(self, ti):
        with self._mutex, self._file_lock:
            # Start download threads if they aren't already started
            while len(self._download_threads) < self.MAX_DOWNLOAD_THREADS:
                thread_iter = len(self._download_threads)
                thread = threading.Thread(target=self._download_thread, daemon=True, args=(thread_iter,))
                self._download_threads[thread_iter] = {"thread": thread, "ti": None}
                thread.start()

            # Add this ti to the download queue
            self._download_queue.append(ti)

    def _process_items(self):
        # i.e. given a bunch of TI item metadata, download them
        if not self._data:
            logger.debug("No Textual Inversion data to process")
            return
        for item in self._data.get("items", []):
            ti = self._parse_civitai_ti_data(item)
            if ti:
                self._file_count += 1
                # We have valid ti data, download it
                self._download_ti(ti)

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

    def _add_ti_to_reference(self, ti):
        ti_key = ti["name"].lower().strip()
        if ti.get("adhoc", False):
            self._adhoc_tis.add(ti_key)
            # Once added to our set, we don't need to specify it was adhoc anymore
            del ti["adhoc"]
        self.model_reference[ti_key] = ti
        self._index_ids[ti["id"]] = ti_key
        orig_name = ti.get("orig_name", ti["name"]).lower().strip()
        self._index_orig_names[orig_name] = ti_key

    def wait_for_downloads(self, timeout=None):
        rtr = 0
        while not self.are_downloads_complete():
            time.sleep(0.5)
            rtr += 0.5
            if timeout and rtr > timeout:
                raise Exception(f"Textual Inversion downloads exceeded specified timeout ({timeout})")

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
        # logger.debug([dthread["ti"] for dthread in self._download_threads.values()])
        for dthread in self._download_threads.values():
            if dthread["ti"] is not None:
                return False
        return True

    def fuzzy_find_ti_key(self, ti_name):
        # sname = Sanitizer.remove_version(ti_name).lower()
        if isinstance(ti_name, int) or ti_name.isdigit():
            if int(ti_name) in self._index_ids:
                return self._index_ids[int(ti_name)]
            return None
        sname = ti_name.lower().strip()
        if sname in self.model_reference:
            return sname
        if sname in self._index_orig_names:
            return self._index_orig_names[sname].lower().strip()
        if Sanitizer.has_unicode(sname):
            for ti in self._index_orig_names:
                if sname in ti:
                    return self._index_orig_names[ti].lower().strip()
            # If a unicode name is not found in the orig_names index
            # it won't be found anywhere else, as unicode chars are converted to ascii in the keys
            # This saves us time doing unnecessary fuzzy searches
            return None
        for ti in self.model_reference:
            if sname in ti:
                return ti.lower().strip()
        for ti in self.model_reference:
            if fuzz.ratio(sname, ti) > 80:
                return ti.lower().strip()
        return None

    # Using `get_model` instead of `get_ti` as it exists in the base class
    def get_model_reference_info(self, model_name: str) -> dict | None:
        """Returns the actual ti details dict for the specified model_name search string
        Returns None if ti name not found"""
        ti_name = self.fuzzy_find_ti_key(model_name)
        if not ti_name:
            return None
        return self.model_reference.get(ti_name)

    def get_ti_filename(self, model_name: str):
        """Returns the actual ti filename for the specified model_name search string
        Returns None if ti name not found"""
        ti = self.get_model_reference_info(model_name)
        if not ti:
            return None
        return ti["filename"]

    def get_ti_name(self, model_name: str):
        """Returns the actual ti name for the specified model_name search string
        Returns None if ti name not found"""
        ti = self.get_model_reference_info(model_name)
        if not ti:
            return None
        return ti["name"]

    def get_ti_id(self, model_name: str):
        """Returns the civitai ti ID for the specified model_name search string
        Returns None if ti name not found"""
        ti = self.get_model_reference_info(model_name)
        if not ti:
            return None
        return ti["id"]

    def get_ti_triggers(self, model_name: str):
        """Returns a list of triggers for a specified ti name
        Returns an empty list if no triggers are found
        Returns None if ti name not found"""
        ti = self.get_model_reference_info(model_name)
        if not ti:
            return None
        triggers = ti.get("triggers")
        if triggers:
            return triggers
        # We don't `return ti.get("triggers", [])`` to avoid the returned list object being modified
        # and then we keep returning previous items
        return []

    def find_ti_trigger(self, model_name: str, trigger_search: str):
        """Searches for a specific trigger for a specified ti name
        Returns None if string not found even with fuzzy search"""
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

    def save_cached_reference_to_disk(self):
        with open(self.models_db_path, "w", encoding="utf-8", errors="ignore") as outfile:
            outfile.write(json.dumps(self.model_reference, indent=4))

    def calculate_downloaded_tis(self, mode=DOWNLOAD_SIZE_CHECK.everything):
        total_size = 0
        for ti in self.model_reference:
            if mode == DOWNLOAD_SIZE_CHECK.top and ti in self._adhoc_tis:
                continue
            if mode == DOWNLOAD_SIZE_CHECK.adhoc and ti not in self._adhoc_tis:
                continue
            total_size += self.model_reference[ti]["size_kb"]
        return total_size

    def is_default_cache_full(self):
        return False

    def is_adhoc_cache_full(self):
        return False

    def calculate_download_queue(self):
        total_queue = 0
        for ti in self._download_queue:
            total_queue += ti["size_kb"]
        return total_queue

    def find_oldest_adhoc_ti(self) -> str | None:
        oldest_ti: str | None = None
        oldest_datetime: datetime | None = None
        for ti in self._adhoc_tis:
            ti_datetime = datetime.strptime(self.model_reference[ti]["last_used"], "%Y-%m-%d %H:%M:%S")
            if not oldest_ti:
                oldest_ti = ti
                oldest_datetime = ti_datetime
                continue
            if oldest_datetime and oldest_datetime > ti_datetime:
                oldest_ti = ti
                oldest_datetime = ti_datetime
        return oldest_ti

    def delete_oldest_ti(self):
        oldest_ti = self._parse_civitai_ti_data()
        if not oldest_ti:
            return
        self.delete_ti(oldest_ti)

    def find_ti_from_filename(self, filename: str):
        for ti in self.model_reference:
            if self.model_reference[ti]["filename"] == filename:
                return ti
        return None

    def find_unused_tis(self):
        files = glob.glob(f"{self.model_folder_path}/*.safetensors")
        filesnames = set()
        for stfile in files:
            filename = os.path.basename(stfile)
            if not self.find_ti_from_filename(filename):
                filesnames.add(filename)
        return filesnames

    def delete_unused_tis(self, timeout=0):
        """Deletes downloaded Textual Inversions which do not appear in the model_reference
        By default protects the user by not running if are_downloads_complete() is not done
        """
        waited = 0
        while not self.are_downloads_complete():
            if waited >= timeout:
                raise Exception(
                    f"Waiting for current Textual Inversion downloads exceeded specified timeout ({timeout})",
                )
            waited += 0.2
            time.sleep(0.2)
        tis_to_delete = self.find_unused_tis()
        for ti_filename in tis_to_delete:
            self.delete_ti_files(ti_filename)
        return tis_to_delete

    def delete_ti_files(self, ti_filename: str):
        filename = os.path.join(self.model_folder_path, ti_filename)
        if not os.path.exists(filename):
            logger.warning(f"Could not find Textual Inversion file on disk to delete: {filename}")
            return
        os.remove(filename)
        logger.info(f"Deleted Textual Inversion file: {filename}")

    def delete_ti(self, ti_name: str):
        ti_info = self.get_model_reference_info(ti_name)
        if not ti_info:
            logger.warning(f"Could not find ti {ti_name} to delete")
            return

        self.delete_ti_files(ti_info["filename"])

        if ti_name in self._adhoc_tis:
            self._adhoc_tis.remove(ti_name)
        else:
            logger.warning(f"Could not find ti {ti_name} in adhoc tis to delete")

        if ti_info["id"] in self._index_ids:
            del self._index_ids[ti_info["id"]]
        else:
            logger.warning(f"Could not find ti {ti_name} in id index to delete")

        if ti_info["orig_name"].lower() in self._index_orig_names:
            del self._index_orig_names[ti_info["orig_name"].lower()]
        else:
            logger.warning(f"Could not find ti {ti_name} in orig_name index to delete")

        if ti_name in self.model_reference:
            del self.model_reference[ti_name]
        else:
            logger.warning(f"Could not find ti {ti_name} in model_reference to delete")
        self.save_cached_reference_to_disk()

    def ensure_ti_deleted(self, ti_name: str):
        ti_key = self.fuzzy_find_ti_key(ti_name)
        if not ti_key:
            return
        self.delete_ti(ti_key)

    # def reset_adhoc_tis(self):
    #     """Compared the known tis from the previous run to the current one
    #     Adds any definitions as adhoc tis, until we have as many Mb as self.max_adhoc_disk"""
    #     while not self.are_downloads_complete():
    #         if self._stop_all_threads:
    #             logger.debug("Stopped processing thread")
    #             return
    #         time.sleep(0.2)
    #     now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    #     self._adhoc_tis = set()
    #     sorted_items = []
    #     try:
    #         sorted_items = sorted(
    #             self._previous_model_reference.items(),
    #             key=lambda x: x[1].get("last_used", now),
    #             reverse=True,
    #         )
    #     except Exception as err:
    #         logger.error(err)
    #     while not self.is_adhoc_cache_full() and len(sorted_items) > 0:
    #         prevti_key, prevti_value = sorted_items.pop()
    #         if prevti_key in self.model_reference:
    #             continue
    #         # If False, it will initiates a redownload and call _add_ti_to_reference() later
    #         if self._check_for_refresh(prevti_key):
    #             self._add_ti_to_reference(prevti_value)
    #         self._adhoc_tis.add(prevti_key)
    #     for ti_key in self.model_reference:
    #         if ti_key in self._previous_model_reference:
    #             self.model_reference[ti_key]["last_used"] = self._previous_model_reference[ti_key].get(
    #                 "last_used",
    #                 now,
    #             )
    #     # Final assurance that all our tis have a last_used timestamp
    #     for ti in self.model_reference.values():
    #         if "last_used" not in ti:
    #             ti["last_used"] = now
    #     self._previous_model_reference = {}
    #     self.save_cached_reference_to_disk()

    def _check_for_refresh(self, ti_name: str):
        """Returns True if a refresh is needed
        and also initiates a refresh
        Else returns False
        """
        ti_details = self.get_model_reference_info(ti_name)
        if not ti_details:
            return True
        refresh = False
        if "last_checked" not in ti_details:
            refresh = True
        elif "baseModel" not in ti_details:
            refresh = True
        else:
            ti_datetime = datetime.strptime(ti_details["last_checked"], "%Y-%m-%d %H:%M:%S")
            if ti_datetime < datetime.now() - timedelta(days=1):
                refresh = True
        if refresh:
            logger.debug(f"Textual Inversion {ti_name} found needing refresh. Initiating metadata download...")
            self._add_ti_ids_to_download_queue([ti_details["id"]], ti_details.get("version_id", -1))
            return False
        return True

    # def check_for_valid

    # def is_adhoc_reset_complete(self):
    #     if self._adhoc_reset_thread and self._adhoc_reset_thread.is_alive():
    #         return False
    #     return True

    # def wait_for_adhoc_reset(self, timeout=15):
    #     rtr = 0
    #     while not self.is_adhoc_reset_complete():
    #         time.sleep(0.2)
    #         rtr += 0.2
    #         if timeout and rtr > timeout:
    #             raise Exception(f"Textual Inversion adhoc reset exceeded specified timeout ({timeout})")

    def stop_all(self):
        self._stop_all_threads = True

    def touch_ti(self, ti_name):
        """Updates the "last_used" key in a ti entry to current UTC time"""
        ti = self.get_model_reference_info(ti_name)
        if not ti:
            logger.warning(f"Could not find ti {ti_name} to touch")
            return
        ti["last_used"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def get_ti_last_use(self, ti_name):
        """Returns a dateimte object based on the "last_used" key in a ti entry"""
        ti = self.get_model_reference_info(ti_name)
        if not ti:
            logger.warning(f"Could not find ti {ti_name} to get last use")
            return None
        return datetime.strptime(ti["last_used"], "%Y-%m-%d %H:%M:%S")

    def fetch_adhoc_ti(self, ti_name, timeout=15):
        if isinstance(ti_name, int) or ti_name.isdigit():
            url = f"https://civitai.com/api/v1/models/{ti_name}"
        else:
            url = f"{self.TI_API}&nsfw={str(self.nsfw).lower()}&query={ti_name}"
        data = self._get_json(url)
        # CivitAI down
        if not data:
            return None
        if "items" in data:
            if len(data["items"]) == 0:
                return None
            ti = self._parse_civitai_ti_data(data["items"][0], adhoc=True)
        else:
            ti = self._parse_civitai_ti_data(data, adhoc=True)
        # For example epi_noiseoffset doesn't have sha256 so we ignore it
        # This avoid us faulting
        if not ti:
            return None
        # We double-check that somehow our search missed it but CivitAI searches differently and found it
        fuzzy_find = self.fuzzy_find_ti_key(ti["id"])
        if fuzzy_find:
            logger.error(fuzzy_find)
            return fuzzy_find
        self._download_ti(ti)
        # We need to wait a bit to make sure the threads pick up the download
        time.sleep(self.THREAD_WAIT_TIME)
        self.wait_for_downloads(timeout)
        return ti["name"].lower()

    def do_baselines_match(self, ti_name, model_details):
        self._check_for_refresh(ti_name)
        lota_details = self.get_model_reference_info(ti_name)
        return True  # FIXME
        if not lota_details:
            logger.warning(f"Could not find ti {ti_name} to check baselines")
            return False
        if "SD 1.5" in lota_details["baseModel"] and model_details["baseline"] == "stable diffusion 1":
            return True
        if "SD 2.1" in lota_details["baseModel"] and model_details["baseline"] == "stable diffusion 2":
            return True
        return False

    @override
    def is_local_model(self, model_name):
        return self.fuzzy_find_ti_key(model_name) is not None

    def get_available_models(self):
        """
        Returns the available model names
        """
        return list(self.model_reference.keys())
