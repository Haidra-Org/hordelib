import importlib
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
    paths_to_prepend = [
        str(comfyui_path),
        str(comfyui_path / "utils"),
        str(comfyui_path / "comfy"),
        str(get_hordelib_path() / "nodes"),
    ]

    # Prepend so vendored ComfyUI takes precedence over similarly named site-packages modules.
    for path in reversed(paths_to_prepend):
        if path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)

    # Ensure previously imported third-party modules do not shadow ComfyUI's utils package.
    sys.modules.pop("utils", None)
    comfy_utils = importlib.import_module("utils")
    # Guard against environments that ship a plain module called "utils".
    if not hasattr(comfy_utils, "__path__"):
        raise ImportError("Expected ComfyUI utils package to be importable; got plain module instead.")
