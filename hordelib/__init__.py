from hordelib import install
from hordelib.config_path import set_system_path
from hordelib.settings import WorkerSettings

VERSION = "0.0.10"
COMFYUI_VERSION = "0bb5f93b9291f088de8eee44cd98db86b44b1e6d"

set_system_path()

installer = install.Installer()
installer.install(COMFYUI_VERSION)


class HordelibException(Exception):
    pass
