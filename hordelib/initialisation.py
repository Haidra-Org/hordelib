# initialisation.py
# Initialise hordelib.
from hordelib import install_comfy
from hordelib.consts import COMFYUI_VERSION, DEFAULT_MODEL_MANAGERS
from hordelib.config_path import set_system_path


def initialise():
    # Ensure we have ComfyUI
    installer = install_comfy.Installer()
    installer.install(COMFYUI_VERSION)

    # Modify python path to include comfyui
    set_system_path()

    # Initialise model manager
    from hordelib.shared_model_manager import SharedModelManager
    SharedModelManager.loadModelManagers(**DEFAULT_MODEL_MANAGERS)
