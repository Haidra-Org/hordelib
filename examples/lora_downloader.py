import hashlib
import json
import os
import re
import string
import threading
import time
from collections import deque

import requests
from loguru import logger


class LoraDownloader:
    """
    Download the highest rated LORAs from Civitai:
        - Download in the background using multiple download threads
        - Hash validate files in memory before saving to disk
        - Don't download LORAs we already have
        - Robust download, clean up the data, retry, etc

    x = LoraDownload( <max_storage_to_use_in_mb> )
    x.download()  # returns immediately spawning background download threads

    # To check if done
    if x.done:
        ...

    # If you want the model meta data once we're done (also saved to loras.json)
    print(x.model_data)

        [
            {
                "sha256": "348071DB544B7242C5EDCB3306160D83BCDE66395153C1DAF38A575C5CEFD66E",
                "filename": "LowRA.safetensors",
                "url": "https://civitai.com/api/download/models/63006",
                "triggers": [
                    "dark theme"
                ],
                "size_mb": 72
            },
                ...etc...


    """

    LORA_API = "https://civitai.com/api/v1/models?types=LORA&sort=Highest%20Rated"
    MAX_RETRIES = 10
    MAX_DOWNLOAD_THREADS = 3  # max concurrent downloads
    RETRY_DELAY = 5  # seconds
    REQUEST_METADATA_TIMEOUT = 30  # seconds
    REQUEST_DOWNLOAD_TIMEOUT = 300  # seconds

    def __init__(self, allowed_storage):
        self._max_disk = allowed_storage
        self._data = None
        self._next_page_url = None
        self._mutex = threading.Lock()
        self._file_count = 0
        self._download_threads = []
        self._download_queue = deque()
        self._download_queue_mb = 0
        self._downloaded_mb = 0
        self._thread = None
        # FIXME this obviously needs to be the correct location
        self._download_dir = "f:/ai/models/loras"
        self.done = False
        self.model_data = []

    def _get_json(self, url):
        retries = 0
        while retries <= LoraDownloader.MAX_RETRIES:
            try:
                response = requests.get(url, timeout=LoraDownloader.REQUEST_METADATA_TIMEOUT)
                response.raise_for_status()
                # Attempt to decode the response to JSON
                return response.json()

            except (requests.HTTPError, requests.ConnectionError, requests.Timeout, json.JSONDecodeError):
                retries += 1
                if retries <= LoraDownloader.MAX_RETRIES:
                    time.sleep(LoraDownloader.RETRY_DELAY)
                else:
                    # Max retries exceeded, give up
                    return None

            except Exception as e:
                # Failed badly
                logger.error(f"LORA download failed {e}")
                return None
        return None

    def _get_more_items(self):
        if not self._data:
            url = LoraDownloader.LORA_API
        else:
            url = self._next_page_url

        # This may be the end of the road, unlikely but...
        if not url:
            logger.warning("End of LORA data reached")
            self.done = True
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

    def _sanitise_filename(self, filename):
        """Don't allow crazy filenames, there are a lot"""
        # First remove exotic unicode characters
        filename = filename.encode("ascii", "ignore").decode("ascii")
        # Now exploit characters
        valid_chars = f"-_.() {string.ascii_letters}{string.digits}"
        return "".join(c for c in filename if c in valid_chars)

    def _parse_civitai_lora_data(self, item):
        """Return a simplified dictionary with the information we actually need about a lora"""
        lora = {
            "sha256": "",
            "filename": "",
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
                lora["filename"] = self._sanitise_filename(file.get("name", ""))
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
            return None
        # Fixup A1111 centric triggers
        for i, trigger in enumerate(lora["triggers"]):
            if re.match("<lora:(.*):.*>", trigger):
                lora["triggers"][i] = re.sub("<lora:(.*):.*>", "\\1", trigger)
        return lora

    def _download_thread(self):
        # We try to download the LORA. There are tens of thousands of these things, we aren't
        # picky if downloads fail, as they often will if civitai is the host, we just move on to
        # the next one
        while True:
            # Endlessly look for files to download and download them
            try:
                lora = self._download_queue.popleft()
            except IndexError:
                # Nothing in the queue
                time.sleep(2)
                continue

            # Download the lora
            retries = 0
            while retries <= LoraDownloader.MAX_RETRIES:
                try:
                    # Just before we download this file, check if we already have it
                    filename = os.path.join(self._download_dir, lora["filename"])
                    hashfile = f"{os.path.splitext(filename)[0]}.sha256"
                    if os.path.exists(filename) and os.path.exists(hashfile):
                        # Check the hash
                        with open(hashfile) as infile:
                            try:
                                hashdata = infile.read().split()[0]
                            except (IndexError, OSError, PermissionError):
                                hashdata = ""
                        if hashdata.lower() == lora["sha256"].lower():
                            # we already have this lora, consider it downloaded
                            logger.debug(f"Already have LORA {lora['filename']}")
                            with self._mutex:
                                self._downloaded_mb += lora["size_mb"]
                                self.model_data.append(lora)
                                if self._downloaded_mb > self._max_disk:
                                    self.done = True
                                break

                    logger.info(f"Starting download of LORA {lora['filename']}")
                    response = requests.get(lora["url"], timeout=LoraDownloader.REQUEST_DOWNLOAD_TIMEOUT)
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
                        with open(hashfile, "w") as outfile:
                            outfile.write(f"{sha256} *{lora['filename']}")

                        # Shout about it
                        logger.info(f"Downloaded LORA {lora['filename']} ({lora['size_mb']} MB)")
                        # Maybe we're done
                        with self._mutex:
                            self.model_data.append(lora)
                            self._downloaded_mb += lora["size_mb"]
                            if self._downloaded_mb > self._max_disk:
                                self.done = True
                        break

                    logger.debug(f"Downloaded LORA file for {lora['filename']} didn't match hash")
                    # we will retry

                except (requests.HTTPError, requests.ConnectionError, requests.Timeout, json.JSONDecodeError) as e:
                    logger.debug(f"Error downloading {lora['filename']} {e}")
                    # we will retry

                except Exception as e:
                    # Failed badly, ignore
                    logger.debug(f"Fatal error downloading {lora['filename']} {e}")
                    # we will retry

                retries += 1
                logger.debug(f"Retry of LORA file download for {lora['filename']}")
                if retries > LoraDownloader.MAX_RETRIES:
                    break  # fail

                time.sleep(LoraDownloader.RETRY_DELAY)

    def _download_lora(self, lora):
        with self._mutex:
            # Start download threads if they aren't already started
            while len(self._download_threads) < LoraDownloader.MAX_DOWNLOAD_THREADS:
                thread = threading.Thread(target=self._download_thread, daemon=True)
                thread.start()
                self._download_threads.append(thread)

            # Add this lora to the download queue
            self._download_queue_mb += lora["size_mb"]
            self._download_queue.append(lora)

    def _process_items(self):
        # i.e. given a bunch of LORA item metadata, download them
        for item in self._data.get("items", []):
            lora = self._parse_civitai_lora_data(item)
            if lora:
                self._file_count += 1
                # Allow a queue of 20% larger than the max disk space as we'll lose some
                if self._download_queue_mb > self._max_disk * 1.2:
                    return
                # We have valid lora data, download it
                self._download_lora(lora)

    def _start_processing(self):
        self.done = False

        while not self.done:
            # Get some items to download
            self._get_more_items()

            # If we have some items to process, process them
            if self._data:
                self._process_items()

    def download(self, wait=False):
        """Start up a background thread downloading and return immediately"""

        # Don't start if we're already busy doing something
        if self._thread:
            return

        # Start processing in a background thread
        self._thread = threading.Thread(target=self._start_processing, daemon=True)
        self._thread.start()

        # Wait for completion of our threads if requested
        if wait:
            while self._thread.is_alive():
                time.sleep(0.5)

        # Save the final model data index
        filename = os.path.join(self._download_dir, "loras.json")
        with open(filename, "w", encoding="utf-8", errors="ignore") as outfile:
            outfile.write(json.dumps(self.model_data, indent=4))


if __name__ == "__main__":
    downloader = LoraDownloader(allowed_storage=1024)  # MB
    downloader.download(wait=True)
