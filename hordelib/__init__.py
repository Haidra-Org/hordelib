import os
import sys
from . import install

VERSION = "0.0.4"
COMFYUI_VERSION = "27fc64ad469c07d8f84b2c2791a593f1cf2c7b59"

current_file_path = os.path.abspath(__file__)
current_folder = os.path.dirname(current_file_path)
comfypath = os.path.join(current_folder, "ComfyUI")
sys.path.append(comfypath)  # noqa: E402

installer = install.Installer()
installer.install(COMFYUI_VERSION)


class HordelibException(Exception):
    pass
