# initialisation.py
# Initialise hordelib.
import os
import shutil
import sys

from loguru import logger

from hordelib import install_comfy
from hordelib.config_path import get_hordelib_path, set_system_path
from hordelib.consts import (
    COMFYUI_VERSION,
    DEFAULT_MODEL_MANAGERS,
    MODEL_CATEGORY_NAMES,
    RELEASE_VERSION,
)
from hordelib.utils.logger import HordeLog

_is_initialised = False


def initialise(
    # model_managers_to_load: dict[MODEL_CATEGORY_NAMES, bool] = DEFAULT_MODEL_MANAGERS,
    *,
    setup_logging=True,
    clear_logs=False,
    debug_logging=False,
):  # XXX # TODO Do we need `model_managers_to_load`?

    global _is_initialised

    # Wipe existing logs if requested
    if clear_logs and os.path.exists("./logs"):
        shutil.rmtree("./logs")

    # Setup logging if requested
    HordeLog.initialise(setup_logging=setup_logging)
    if debug_logging:
        HordeLog.set_logger_verbosity(5)
        HordeLog.quiesce_logger(0)

    # If developer mode, don't permit some things
    if not RELEASE_VERSION and " " in get_hordelib_path():
        # Our runtime patching can't handle this
        raise Exception(
            "Do not run this project in developer mode from a path that " "contains spaces in directory names.",
        )

    # Ensure we have ComfyUI
    logger.debug("Clearing command line args in sys.argv before ComfyUI load")
    sys_arg_bkp = sys.argv.copy()
    sys.argv = sys.argv[:1]
    installer = install_comfy.Installer()
    installer.install(COMFYUI_VERSION)

    # Modify python path to include comfyui
    set_system_path()

    # Initialise model manager
    from hordelib.shared_model_manager import SharedModelManager

    SharedModelManager()

    sys.argv = sys_arg_bkp

    _is_initialised = True


def is_initialised():
    return _is_initialised
