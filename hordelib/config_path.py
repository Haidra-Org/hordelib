import os
import sys


def get_hordelib_path():
    current_file_path = os.path.abspath(__file__)
    return os.path.dirname(current_file_path)


def get_comfyui_path():
    return os.path.join(get_hordelib_path(), "ComfyUI")


def set_system_path():
    sys.path.append(get_comfyui_path())  # noqa: E402
