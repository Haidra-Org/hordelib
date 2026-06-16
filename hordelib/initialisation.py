# initialisation.py
# Initialise hordelib.
import os
import shutil
import sys

from loguru import logger

from hordelib.config_path import get_comfyui_path, get_hordelib_path, set_system_path
from hordelib.consts import RELEASE_VERSION
from hordelib.installation import EnvironmentInstaller, load_packaged_manifest
from hordelib.utils.logger import HordeLog

_is_initialised = False


def initialise(
    # model_managers_to_load: dict[MODEL_REFERENCE_CATEGORY, bool] = ...,
    *,
    setup_logging: bool | None = True,
    clear_logs=False,
    logging_verbosity=3,
    process_id: int | None = None,
    force_normal_vram_mode: bool = True,
    extra_comfyui_args: list[str] | None = None,
    disable_smart_memory: bool = False,
    do_not_load_model_mangers: bool = True,
    models_not_to_force_load: list[str] | None = None,
    reference_offline: bool | None = None,
):
    """Initialise hordelib. This is required before using any other hordelib functions.

    Args:
        setup_logging (bool | None, optional): Whether to use hordelib's loguru logging system. Defaults to True.
        clear_logs (bool, optional): Whether logs should be purged when loading loguru. Defaults to False.
        logging_verbosity (int, optional): The message level of logging. Defaults to 3.
        process_id (int | None, optional): If this is is being used in a child processes, the identifier. \
            Defaults to None.
        force_low_vram (bool, optional): Whether to forcibly disable ComfyUI's high/med vram modes. Defaults to False.
        extra_comfyui_args (list[str] | None, optional): Any additional CLI args for comfyui that should be used. \
            Defaults to None.
        models_not_to_force_load (list[str] | None, optional): Models that should not be force loaded, as \
            ``KNOWN_IMAGE_GENERATION_BASELINE`` members (preferred) or raw comfy class-name fragments. \
            **If this is `None`, the defaults are used.** If you wish to override the defaults, pass an empty list. \
            Defaults to None.
        reference_offline (bool | None, optional): If True, the model reference manager reads references \
            from local disk only and never downloads them (the caller/parent process owns downloading). \
            If None, defers to ``HORDE_MODEL_REFERENCE_OFFLINE``. Defaults to None.
    """
    global _is_initialised

    # Wipe existing logs if requested
    if clear_logs and os.path.exists("./logs"):
        shutil.rmtree("./logs")

    if setup_logging is not None:
        # Setup logging if requested
        HordeLog.initialise(
            setup_logging=setup_logging,
            process_id=process_id,
            verbosity_count=logging_verbosity,
        )

    # If developer mode, don't permit some things
    if not RELEASE_VERSION and " " in str(get_hordelib_path()):
        # Our runtime patching can't handle this
        raise Exception(
            "Do not run this project in developer mode from a path that contains spaces in directory names.",
        )

    # Ensure we have ComfyUI (and any manifest-pinned custom nodes)
    logger.debug("Clearing command line args in sys.argv before ComfyUI load")
    sys_arg_bkp = sys.argv.copy()
    sys.argv = sys.argv[:1]
    EnvironmentInstaller(load_packaged_manifest()).ensure(get_comfyui_path())

    # Tell comfyui_controlnet_aux where to store its annotator checkpoints before its package
    # is imported (which happens when ComfyUI loads custom nodes).
    from hordelib.settings import UserSettings

    os.environ.setdefault(
        "AUX_ANNOTATOR_CKPTS_PATH",
        str(UserSettings.get_model_directory() / "controlnet" / "annotators"),
    )
    os.environ.setdefault("AUX_USE_SYMLINKS", "False")

    # Modify python path to include comfyui
    set_system_path()

    # Fail fast (before ComfyUI imports torch) when the installed torch/torchvision (or a hand-installed
    # torchaudio) were built for different CUDA/CPU backends, with a message that names the fix.
    from hordelib.utils.torch_build import verify_torch_build_consistency

    verify_torch_build_consistency()

    import hordelib.comfy_horde

    hordelib.comfy_horde.do_comfy_import(
        force_normal_vram_mode=force_normal_vram_mode,
        extra_comfyui_args=extra_comfyui_args,
        disable_smart_memory=disable_smart_memory,
    )
    if models_not_to_force_load is not None:
        logger.debug("Overriding models_not_to_force_load: models={}", models_not_to_force_load)
        from hordelib.execution import comfy_patches

        # Entries may be horde baseline enum members (preferred) or raw comfy class fragments
        comfy_patches.models_not_to_force_load = comfy_patches.resolve_force_load_skip_entries(
            list(models_not_to_force_load),
        )

    vram_on_start_free = hordelib.comfy_horde.get_torch_free_vram_mb()
    vram_total = hordelib.comfy_horde.get_torch_total_vram_mb()
    vram_percent_used = round((vram_total - vram_on_start_free) / vram_total * 100, 2)
    message_addendum = "This will almost certainly cause issues. "
    message_addendum += "It is strongly recommended you close other applications before running the worker."
    if vram_on_start_free < 2000:
        logger.warning("You have less than 2GB of VRAM free. {}", message_addendum)

    if vram_percent_used > 60:
        logger.warning("VRAM percent used on start: percent={}%. {}", vram_percent_used, message_addendum)

    if vram_total < 4000:
        logger.warning("You have less than 4GB of VRAM total. It is likely that generations will happen very slowly.")

    # Initialise model manager
    from hordelib.shared_model_manager import SharedModelManager

    if reference_offline is not None:
        SharedModelManager._reference_offline = reference_offline

    SharedModelManager(do_not_load_model_mangers=do_not_load_model_mangers)

    sys.argv = sys_arg_bkp

    _is_initialised = True


def is_initialised():
    return _is_initialised
