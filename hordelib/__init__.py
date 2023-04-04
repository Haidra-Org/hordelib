import os
import sys

from hordelib import install
from hordelib.config_path import set_system_path
from hordelib.settings import WorkerSettings

VERSION = "0.0.10"
COMFYUI_VERSION = "1718730e80549c35ce3c5d3fb7926ce5654a2fdd"

set_system_path()

installer = install.Installer()
installer.install(COMFYUI_VERSION)


class HordelibException(Exception):
    pass
