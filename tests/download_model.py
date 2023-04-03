# download_model.py
# Download a test model if we don't have one.
import requests
import os


def assert_test_model():
    url = "https://civitai.com/api/download/models/15236?type=Model&format=PickleTensor"
    filename = "model.ckpt"
    if os.path.exists(filename) and os.path.getsize(filename) > 2132000000:
        return

    print("Downloading test model, please please (2GB)...")
    response = requests.get(url, stream=True)

    if response.status_code == 200:
        with open(filename, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
    else:
        raise Exception(f"Request failed with status code: {response.status_code}")
