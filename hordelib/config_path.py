import os
import sys


def get_hordelib_path():
    """Returns the path hordelib is installed in."""
    current_file_path = os.path.abspath(__file__)
    return os.path.dirname(current_file_path)


def get_comfyui_path():
    """Returns the path to ComfyUI that hordelib installs and manages."""
    if os.path.exists(os.path.join(get_hordelib_path(), "_version.py")):
        # Packaged version
        return os.path.join(get_hordelib_path(), "_comfyui")
    else:
        # Development version
        return os.path.join(os.path.dirname(get_hordelib_path()), "ComfyUI")


def set_system_path():
    """Adds ComfyUI to the python path, as it is not a proper library."""
    sys.path.append(get_comfyui_path())
    sys.path.append(os.path.join(get_hordelib_path(), "nodes"))
