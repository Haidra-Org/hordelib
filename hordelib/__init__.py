import os
import sys

from hordelib import install
from hordelib.config_path import set_system_path
from hordelib.settings import WorkerSettings

VERSION = "0.0.10"
COMFYUI_VERSION = "1ed6cadf1292fa7607317a438777e6e37fe2709d"

set_system_path()

installer = install.Installer()
installer.install(COMFYUI_VERSION)


class HordelibException(Exception):
    pass
