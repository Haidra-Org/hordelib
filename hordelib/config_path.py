import os
import sys


def set_system_path():
    current_file_path = os.path.abspath(__file__)
    current_folder = os.path.dirname(current_file_path)
    comfypath = os.path.join(current_folder, "ComfyUI")
    sys.path.append(comfypath)  # noqa: E402
