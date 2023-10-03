import sys
from pathlib import Path


def get_hordelib_path() -> Path:
    """Returns the path hordelib is installed in."""
    current_file_path = Path(__file__).resolve()
    return current_file_path.parent


def get_comfyui_path() -> Path:
    """Returns the path to ComfyUI that hordelib installs and manages."""
    hordelib_path = get_hordelib_path()
    if (hordelib_path / "_version.py").exists():
        # Packaged version
        return hordelib_path / "_comfyui"

    # Development version
    return hordelib_path.parent / "ComfyUI"


def set_system_path() -> None:
    """Adds ComfyUI to the python path, as it is not a proper library."""
    comfyui_path = get_comfyui_path()
    sys.path.append(str(comfyui_path))
    sys.path.append(str(get_hordelib_path() / "nodes"))
