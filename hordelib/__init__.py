import os
import sys

VERSION = "0.0.1"
COMFYUI_VERSION = "9a27030519c6e1e2024df50cdc547f6a05d714bc"

current_file_path = os.path.abspath(__file__)
current_folder = os.path.dirname(current_file_path)
comfypath = os.path.join(current_folder, "ComfyUI")
sys.path.append(comfypath)  # noqa: E402


class HordelibException(Exception):
    pass
