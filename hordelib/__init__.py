import os
import sys
from . import install

VERSION = "0.0.9"
COMFYUI_VERSION = "72f9235a491e7800b3a7686e4901729d371dabed"

current_file_path = os.path.abspath(__file__)
current_folder = os.path.dirname(current_file_path)
comfypath = os.path.join(current_folder, "ComfyUI")
sys.path.append(comfypath)  # noqa: E402

installer = install.Installer()
installer.install(COMFYUI_VERSION)


class HordelibException(Exception):
    pass
