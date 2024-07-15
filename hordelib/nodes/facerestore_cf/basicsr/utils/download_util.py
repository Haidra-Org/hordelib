import math
import os
from urllib.parse import urlparse

import requests
from torch.hub import download_url_to_file, get_dir
from tqdm import tqdm

from .misc import sizeof_fmt
from hordelib.shared_model_manager import SharedModelManager


def download_file_from_google_drive(file_id, save_path):
    """Download files from google drive.
    Ref:
    https://stackoverflow.com/questions/25010369/wget-curl-large-file-from-google-drive  # noqa E501
    Args:
        file_id (str): File id.
        save_path (str): Save path.
    """

    session = requests.Session()
    URL = "https://docs.google.com/uc?export=download"
    params = {"id": file_id}

    response = session.get(URL, params=params, stream=True)
    token = get_confirm_token(response)
    if token:
        params["confirm"] = token
        response = session.get(URL, params=params, stream=True)

    # get file size
    response_file_size = session.get(URL, params=params, stream=True, headers={"Range": "bytes=0-2"})
    print(response_file_size)
    if "Content-Range" in response_file_size.headers:
        file_size = int(response_file_size.headers["Content-Range"].split("/")[1])
    else:
        file_size = None

    save_response_content(response, save_path, file_size)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value
    return None


def save_response_content(response, destination, file_size=None, chunk_size=32768):
    if file_size is not None:
        pbar = tqdm(total=math.ceil(file_size / chunk_size), unit="chunk")

        readable_file_size = sizeof_fmt(file_size)
    else:
        pbar = None

    with open(destination, "wb") as f:
        downloaded_size = 0
        for chunk in response.iter_content(chunk_size):
            downloaded_size += chunk_size
            if pbar is not None:
                pbar.update(1)
                pbar.set_description(f"Download {sizeof_fmt(downloaded_size)} / {readable_file_size}")
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
        if pbar is not None:
            pbar.close()


def load_file_from_url(url, model_dir=None, progress=True, file_name=None):
    """Load file form http url, will download models if necessary.
    Ref:https://github.com/1adrianb/face-alignment/blob/master/face_alignment/utils.py
    Args:
        url (str): URL to be downloaded.
        model_dir (str): The path to save the downloaded model. Should be a full path. If None, use pytorch hub_dir.
            Default: None.
        progress (bool): Whether to show the download progress. Default: True.
        file_name (str): The downloaded file name. If None, use the file name in the url. Default: None.
    Returns:
        str: The path to the downloaded file.
    """
    return str(SharedModelManager.manager.gfpgan.model_folder_path / file_name)
