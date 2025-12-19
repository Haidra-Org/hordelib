# comfy.py
# Wrapper around comfy to allow usage by the horde worker.
import asyncio
import contextlib
import copy
import gc
import glob
import hashlib
import io
import json
import os
import re
import time
import sys
import typing
import types
import uuid
import random
import threading
from pprint import pformat
import loguru
import requests
import psutil
from collections.abc import Callable

import torch
from loguru import logger

from hordelib.settings import UserSettings
from hordelib.utils.ioredirect import ComfyUIProgress, OutputCollector
from hordelib.config_path import get_hordelib_path

# Note It may not be abundantly clear with no context what is going on below, and I will attempt to clarify:
#
# The nature of packaging comfyui leads to a situation where some trickery takes place in
# `hordelib.initialise()` in order for the imports below to work. Calling that function is therefore a prerequisite
# to using any of the functions in this module. If you do not, you will get an exception as reaching into comfyui is
# impossible without it.
#
# As for having the imports below in a function, this is to ensure that `hordelib.initialise()` has done its magic. It
# in turn calls `do_comfy_import()`.
#
# There may be other ways to skin this cat, but this strategy minimizes certain kinds of hassle.
#
# If you tamper with the code in this module to bring the imports out of the function, you may find that you have
# broken, among myriad other things, the ability of pytest to do its test discovery because you will have lost the
# ability for modules which, directly or otherwise, import this module without having called `hordelib.initialise()`.
# Pytest discovery will come across those imports, valiantly attempt to import them and fail with a cryptic error
# message.
#
# Correspondingly, you will find that to be an enormous hassle if you are are trying to leverage pytest in any
# reasonability sophisticated way (outside of tox), and you will be forced to adopt solution below or something
# equally obtuse, or in lieu of either of those, abandon all hope and give up on the idea of attempting to develop any
# new features or author new tests for hordelib.
#
# Keen readers may have noticed that the aforementioned issues could be side stepped by simply calling
# `hordelib.initialise()` automatically, such as in `test/__init__.py` or in a `conftest.py`. You would be correct,
# but that would be a terrible idea as a general practice. It would mean that every time you saved a file in your
# editor, a number of heavyweight operations would be triggered, such as loading comfyui, while pytest discovery runs
# and that would cause slow and unpredictable behavior in your editor.
#
# This would be a nightmare to debug, as this author is able to attest to and is the reason this wall of text exists.
#
# Further, if you are like myself, and enjoy type hints, you will find that any modules have this file in their import
# chain will be un-importable in certain contexts and you would be unable to provide the relevant type hints.
#
# Having exercised a herculean amount of focus to read this far, I suggest you also glance at the code in
# `hordelib.initialise()` to get a sense of what is going on there, and if you're still confused, ask a hordelib dev
# who would be happy to share the burden of understanding.

_comfy_load_models_gpu: types.FunctionType
_comfy_current_loaded_models: list = None  # type: ignore

_comfy_nodes: types.ModuleType
_comfy_PromptExecutor: typing.Any
_comfy_validate_prompt: types.FunctionType
_comfy_CacheType: typing.Any

_comfy_folder_names_and_paths: dict[str, tuple[list[str], list[str] | set[str]]]
_comfy_supported_pt_extensions: set[str]

_comfy_load_checkpoint_guess_config: types.FunctionType

_comfy_get_torch_device: types.FunctionType
"""Will return the current torch device, typically the GPU."""
_comfy_get_free_memory: types.FunctionType
"""Will return the amount of free memory on the current torch device. This value can be misleading."""
_comfy_get_total_memory: types.FunctionType
"""Will return the total amount of memory on the current torch device."""
_comfy_load_torch_file: types.FunctionType
_comfy_model_loading: types.ModuleType
_comfy_free_memory: Callable[[float, torch.device, list], None]
"""Will aggressively unload models from memory"""
_comfy_cleanup_models: Callable
"""Will unload unused models from memory"""
_comfy_soft_empty_cache: Callable
"""Triggers comfyui and torch to empty their caches"""

_comfy_is_changed_cache_get: Callable
_comfy_model_patcher_load: Callable
_comfy_load_calculate_weight: Callable
_comfy_text_encoder_initial_device: Callable

_comfy_interrupt_current_processing: types.FunctionType

_canny: types.ModuleType
_hed: types.ModuleType
_leres: types.ModuleType
_midas: types.ModuleType
_mlsd: types.ModuleType
_openpose: types.ModuleType
_pidinet: types.ModuleType
_uniformer: types.ModuleType


# isort: off

import logging


class InterceptHandler(logging.Handler):
    """
    Add logging handler to augment python stdlib logging.

    Logs which would otherwise go to stdlib logging are redirected through
    loguru.
    """

    _ignored_message_contents: list[str]
    _ignored_libraries: list[str]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ignored_message_contents = {
            "lowvram: loaded module regularly",
            "lora key not loaded",
        }
        self._ignored_libraries = [
            "numba.core",
        ]

    def add_ignored_message_content(self, content: str) -> None:
        """Add a message content to ignore."""
        self._ignored_message_contents.append(content)

    def get_ignored_message_contents(self) -> list[str]:
        """Return the list of ignored message contents."""
        return self._ignored_message_contents.copy()

    def reset_ignored_message_contents(self) -> None:
        """Reset the list of ignored message contents."""
        self._ignored_message_contents = []

    def add_ignored_library(self, library: str) -> None:
        """Add a library to ignore."""
        self._ignored_libraries.append(library)

    def get_ignored_libraries(self) -> list[str]:
        """Return the list of ignored libraries."""
        return self._ignored_libraries.copy()

    def reset_ignored_libraries(self) -> None:
        """Reset the list of ignored libraries."""
        self._ignored_libraries = []

    @logger.catch(default=True, reraise=True)
    def emit(self, record):
        library = record.name
        for ignored_library in self._ignored_libraries:
            if ignored_library in library:
                return

        message = record.getMessage()
        for ignored_message_content in self._ignored_message_contents:
            if ignored_message_content in message:
                return

        # Get corresponding Loguru level if it exists.
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message.
        frame, depth = logging.currentframe(), 2
        while frame and (
            "loguru" in frame.f_code.co_filename
            or frame.f_code.co_filename == logging.__file__
            or frame.f_code.co_filename == __file__
        ):
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, message)


intercept_handler = InterceptHandler()
# ComfyUI uses stdlib logging, so we need to intercept it.
logging.basicConfig(handlers=[intercept_handler], level=0, force=True)


def do_comfy_import(
    force_normal_vram_mode: bool = False,
    extra_comfyui_args: list[str] | None = None,
    disable_smart_memory: bool = False,
    hijack_load_models_gpu: bool = False,
) -> None:
    """Import comfy modules and hijack certain functions for horde use case.
    Args:
        force_normal_vram_mode: If true, forces comfy to use normal VRAM mode using the comfyui command line arg.
        extra_comfyui_args: Extra command line arguments to pass to comfyui.
        disable_smart_memory: If true, disables comfy smart memory management using the comfyui command line arg.
        hijack_load_models_gpu: If true, hijacks comfy model loading to force full loads conditionally based on
            model type (see _load_models_gpu_hijack).
    """
    global _comfy_current_loaded_models
    global _comfy_load_models_gpu
    global _comfy_nodes, _comfy_PromptExecutor, _comfy_validate_prompt, _comfy_CacheType
    global _comfy_folder_names_and_paths, _comfy_supported_pt_extensions
    global _comfy_load_checkpoint_guess_config
    global _comfy_get_torch_device, _comfy_get_free_memory, _comfy_get_total_memory
    global _comfy_load_torch_file, _comfy_model_loading, _comfy_text_encoder_initial_device
    global _comfy_free_memory, _comfy_cleanup_models, _comfy_soft_empty_cache
    global _canny, _hed, _leres, _midas, _mlsd, _openpose, _pidinet, _uniformer

    global _comfy_interrupt_current_processing

    if disable_smart_memory:
        logger.info("Disabling smart memory")
        sys.argv.append("--disable-smart-memory")

    if force_normal_vram_mode:
        logger.info("Forcing normal vram mode")
        sys.argv.append("--normalvram")

    if extra_comfyui_args is not None:
        sys.argv.extend(extra_comfyui_args)

    logger.info(f"hijack_load_models_gpu is set to {hijack_load_models_gpu}")

    # Note these imports are intentionally somewhat obfuscated as a reminder to other modules
    # that they should never call through this module into comfy directly. All calls into
    # comfy should be abstracted through functions in this module.
    output_collector = OutputCollector()
    with contextlib.redirect_stdout(output_collector), contextlib.redirect_stderr(output_collector):
        from comfy.options import enable_args_parsing

        enable_args_parsing()
        import execution
        from execution import nodes as _comfy_nodes
        from execution import PromptExecutor as _comfy_PromptExecutor
        from execution import validate_prompt as _comfy_validate_prompt
        from execution import CacheType as _comfy_CacheType

        # from execution import recursive_output_delete_if_changed
        from execution import IsChangedCache

        global _comfy_is_changed_cache_get
        _comfy_is_changed_cache_get = IsChangedCache.get

        IsChangedCache.get = IsChangedCache_get_hijack  # type: ignore

        from folder_paths import folder_names_and_paths as _comfy_folder_names_and_paths
        from folder_paths import supported_pt_extensions as _comfy_supported_pt_extensions
        from comfy.sd import load_checkpoint_guess_config as _comfy_load_checkpoint_guess_config
        from comfy.model_management import current_loaded_models as _comfy_current_loaded_models
        from comfy.model_management import load_models_gpu

        _comfy_load_models_gpu = load_models_gpu  # type: ignore
        import comfy.model_management

        if hijack_load_models_gpu:
            comfy.model_management.load_models_gpu = _load_models_gpu_hijack  # type: ignore

        from comfy.model_management import get_torch_device as _comfy_get_torch_device
        from comfy.model_management import get_free_memory as _comfy_get_free_memory
        from comfy.model_management import get_total_memory as _comfy_get_total_memory
        from comfy.model_management import free_memory as _comfy_free_memory
        from comfy.model_management import cleanup_models as _comfy_cleanup_models
        from comfy.model_management import soft_empty_cache as _comfy_soft_empty_cache
        from comfy.model_management import interrupt_current_processing as _comfy_interrupt_current_processing
        from comfy.model_management import text_encoder_initial_device as _comfy_text_encoder_initial_device

        comfy.model_management.text_encoder_initial_device = text_encoder_initial_device_hijack  # type: ignore

        from comfy.utils import load_torch_file as _comfy_load_torch_file
        from comfy_extras.chainner_models import model_loading as _comfy_model_loading

        from comfy.model_patcher import ModelPatcher

        global _comfy_model_patcher_load
        _comfy_model_patcher_load = ModelPatcher.load
        ModelPatcher.load = _model_patcher_load_hijack  # type: ignore

        global _comfy_load_calculate_weight
        import comfy.lora
        from comfy.lora import calculate_weight as _comfy_load_calculate_weight

        comfy.lora.calculate_weight = _calculate_weight_hijack  # type: ignore

        from hordelib.nodes.comfy_controlnet_preprocessors import (
            canny as _canny,
            hed as _hed,
            leres as _leres,
            midas as _midas,
            mlsd as _mlsd,
            openpose as _openpose,
            pidinet as _pidinet,
            uniformer as _uniformer,
        )

        logger.info("Comfy_Horde initialised")

    log_free_ram()
    output_collector.replay()


# isort: on
models_not_to_force_load: list = [
    "cascade",
    "sdxl",
    "flux",
    "qwen_image",
    "z_image_turbo",
]  # other possible values could be `basemodel` or `sd1`
"""Models which should not be forced to load in the comfy model loading hijack.

Possible values include `cascade`, `sdxl`, `basemodel`, `sd1` or any other comfyui classname
which can be passed to comfyui's `load_models_gpu` function (as a `ModelPatcher.model`).
"""

disable_force_loading: bool = False


def _do_not_force_load_model_in_patcher(model_patcher):
    model_name_lower = str(type(model_patcher.model)).lower()
    if "clip" in model_name_lower:
        return False

    for model in models_not_to_force_load:
        if model in model_name_lower:
            return True

    return False


def _load_models_gpu_hijack(*args, **kwargs):
    """Intercepts the comfy load_models_gpu function to force full load.

    ComfyUI is too conservative in its loading to GPU for the worker/horde use case where we can have
    multiple ComfyUI instances running on the same GPU. This function forces a full load of the model
    and the worker/horde-engine takes responsibility for managing the memory or the problems this may
    cause.
    """
    found_model_to_skip = False
    for model_patcher in args[0]:
        found_model_to_skip = _do_not_force_load_model_in_patcher(model_patcher)
        if found_model_to_skip:
            break

    global _comfy_current_loaded_models
    if found_model_to_skip:
        logger.debug("Not overriding model load")
        kwargs["memory_required"] = 1e30
        _comfy_load_models_gpu(*args, **kwargs)
        return

    if "force_full_load" in kwargs:
        kwargs.pop("force_full_load")

    kwargs["force_full_load"] = True
    _comfy_load_models_gpu(*args, **kwargs)


def _model_patcher_load_hijack(*args, **kwargs):
    """Intercepts the comfy ModelPatcher.load function to force full load.

    See _load_models_gpu_hijack for more information
    """
    global _comfy_model_patcher_load

    model_patcher = args[0]
    if _do_not_force_load_model_in_patcher(model_patcher):
        logger.debug("Not overriding model load")
        _comfy_model_patcher_load(*args, **kwargs)
        return

    if "full_load" in kwargs:
        kwargs.pop("full_load")

    kwargs["full_load"] = True
    _comfy_model_patcher_load(*args, **kwargs)


def _calculate_weight_hijack(*args, **kwargs):
    global _comfy_load_calculate_weight
    patches = args[0]

    for p in patches:
        v = p[1]
        if not isinstance(v, tuple):
            continue
        patch_type = v[0]
        if patch_type != "diff":
            continue
        if len(v) == 2 and isinstance(v[1], list):
            for idx, val in enumerate(v[1]):
                if val is None:
                    v[1][idx] = {"pad_weight": False}
                    # logger.debug(f"Setting pad_weight to False for {v[0]} on {p} in {patches}")
                    break

    return _comfy_load_calculate_weight(*args, **kwargs)


_last_pipeline_settings_hash = ""

import PIL.Image


def default_json_serializer_pil_image(obj):
    if isinstance(obj, PIL.Image.Image):
        return str(hash(obj.__str__()))
    return obj


async def IsChangedCache_get_hijack(self, *args, **kwargs):
    global _comfy_is_changed_cache_get
    result = await _comfy_is_changed_cache_get(self, *args, **kwargs)

    global _last_pipeline_settings_hash

    prompt = self.dynprompt.original_prompt

    pipeline_settings_hash = hashlib.md5(
        json.dumps(prompt, default=default_json_serializer_pil_image).encode(),
    ).hexdigest()

    if pipeline_settings_hash != _last_pipeline_settings_hash:
        _last_pipeline_settings_hash = pipeline_settings_hash
        logger.debug(f"Pipeline settings changed: {pipeline_settings_hash}")
        logger.debug(f"Cache length: {len(self.outputs_cache.cache)}")
        logger.debug(f"Subcache length: {len(self.outputs_cache.subcaches)}")

        logger.debug(f"IsChangedCache.dynprompt.all_node_ids: {self.dynprompt.all_node_ids()}")

    if result:
        logger.debug(f"IsChangedCache.get: {result}")

    return result


def text_encoder_initial_device_hijack(*args, **kwargs):
    # This ensures clip models are loaded on the CPU first
    return torch.device("cpu")


def clear_gc_and_torch_cache() -> None:
    """Clear the garbage collector and the PyTorch cache."""
    gc.collect()
    from torch.cuda import empty_cache

    empty_cache()


def unload_all_models_vram():
    global _comfy_current_loaded_models

    log_free_ram()

    from hordelib.shared_model_manager import SharedModelManager

    logger.debug("In unload_all_models_vram")
    logger.debug(f"{len(SharedModelManager.manager._models_in_ram)} models cached in shared model manager")

    logger.debug(f"{len(_comfy_current_loaded_models)} models loaded in comfy")

    logger.debug("Freeing memory on all devices")
    _comfy_free_memory(1e30, _comfy_get_torch_device())
    log_free_ram()

    logger.debug("Cleaning up models")
    with torch.no_grad():
        try:
            _comfy_soft_empty_cache()
            log_free_ram()
        except Exception as e:
            logger.error(f"Exception during comfy unload: {e}")
            _comfy_cleanup_models()
            _comfy_soft_empty_cache()

    logger.debug(f"{len(SharedModelManager.manager._models_in_ram)} models cached in shared model manager")
    logger.debug(f"{len(_comfy_current_loaded_models)} models loaded in comfy")

    clear_gc_and_torch_cache()
    log_free_ram()


def unload_all_models_ram():
    global _comfy_current_loaded_models

    log_free_ram()
    from hordelib.shared_model_manager import SharedModelManager

    logger.debug("In unload_all_models_ram")
    logger.debug(f"{len(SharedModelManager.manager._models_in_ram)} models cached in shared model manager")

    SharedModelManager.manager._models_in_ram = {}
    logger.debug(f"{len(_comfy_current_loaded_models)} models loaded in comfy")
    all_devices = set()
    for model in _comfy_current_loaded_models:
        all_devices.add(model.device)

    all_devices.add(get_torch_device())

    with torch.no_grad():
        for device in all_devices:
            logger.debug(f"Freeing memory on device {device}")
            _comfy_free_memory(1e30, device)

        log_free_ram()
        logger.debug("Freeing memory on CPU")
        _comfy_free_memory(1e30, torch.device("cpu"))

        log_free_ram()

        logger.debug("Cleaning up models")
        _comfy_cleanup_models()
        log_free_ram()

        logger.debug("Soft emptying cache")
        _comfy_soft_empty_cache()
        log_free_ram()

    logger.debug(f"{len(SharedModelManager.manager._models_in_ram)} models cached in shared model manager")
    logger.debug(f"{len(_comfy_current_loaded_models)} models loaded in comfy")

    clear_gc_and_torch_cache()
    log_free_ram()


def get_torch_device():
    return _comfy_get_torch_device()


def get_torch_total_vram_mb():
    return round(_comfy_get_total_memory() / (1024 * 1024))


def get_torch_free_vram_mb():
    return round(_comfy_get_free_memory() / (1024 * 1024))


def log_free_ram():
    logger.debug(
        f"Free VRAM: {get_torch_free_vram_mb():0.0f} MB, "
        f"Free RAM: {psutil.virtual_memory().available / (1024 * 1024):0.0f} MB",
    )


def interrupt_comfyui_processing():
    logger.warning("Interrupting comfyui processing")
    _comfy_interrupt_current_processing()


class Comfy_Horde:
    """Handles horde-specific behavior against ComfyUI."""

    # We save pipelines from the ComfyUI GUI. This is very convenient as it
    # makes it super easy to load and edit them in the future. We allow the pipeline
    # design with standard nodes, then at runtime we dynamically replace these
    # node types with our own where we need to. This allows easy previewing in ComfyUI
    # which our custom nodes don't allow.
    NODE_REPLACEMENTS = {
        "CheckpointLoaderSimple": "HordeCheckpointLoader",
        "UNETLoader": "HordeCheckpointLoader",
        # "UpscaleModelLoader": "HordeUpscaleModelLoader",
        "SaveImage": "HordeImageOutput",
        "LoadImage": "HordeImageLoader",
        # "DiffControlNetLoader": "HordeDiffControlNetLoader",
        "LoraLoader": "HordeLoraLoader",
    }
    """A mapping of ComfyUI node types to Horde node types."""

    # We may wish some ComfyUI standard nodes had different names for consistency. Here
    # we can dynamically rename some node input parameter names.
    NODE_PARAMETER_REPLACEMENTS: dict[str, dict[str, str]] = {
        #     "HordeCheckpointLoader": {
        #         # We name this "model_name" as then we can use the same generic code in our model loaders
        #         "ckpt_name": "model_name",
        #     },
    }
    """A mapping of ComfyUI node types which map to a dictionary of node input parameter names to rename."""

    GC_TIME = 30
    """The approximate number of seconds to force garbage collection."""

    pipelines: dict
    images: list[dict[str, io.BytesIO]] | None

    def __init__(
        self,
        *,
        comfyui_callback: typing.Callable[[str, dict, str], None] | None = None,
        aggressive_unloading: bool = True,
    ) -> None:
        """Initialise the Comfy_Horde object.

        This must be called before using any of the functions in this module as it sets critical ComfyUI
        globals and settings.
        """
        if _comfy_current_loaded_models is None:
            raise RuntimeError("hordelib.initialise() must be called before using comfy_horde.")
        self.pipelines = {}
        self._exit_time = 0
        self._callers = 0
        self._gc_timer = time.time()
        self._counter_mutex = threading.Lock()
        self.images = None

        # Set comfyui paths for checkpoints, loras, etc
        self._set_comfyui_paths()

        # Load our pipelines
        self._load_pipelines()

        # Load our custom nodes
        self._load_custom_nodes()

        self._comfyui_callback = comfyui_callback
        self.aggressive_unloading = aggressive_unloading

    def _set_comfyui_paths(self) -> None:
        # These set the default paths for comfyui to look for models and embeddings. From within hordelib,
        # we aren't ever going to use the default ones, and this may help lubricate any further changes.
        _comfy_folder_names_and_paths["loras"] = (
            [
                _comfy_folder_names_and_paths["loras"][0][0],
                str(UserSettings.get_model_directory() / "lora"),
            ],
            _comfy_supported_pt_extensions,
        )
        _comfy_folder_names_and_paths["embeddings"] = (
            [
                _comfy_folder_names_and_paths["embeddings"][0][0],
                str(UserSettings.get_model_directory() / "ti"),
            ],
            _comfy_supported_pt_extensions,
        )

        _comfy_folder_names_and_paths["checkpoints"] = (
            [
                _comfy_folder_names_and_paths["checkpoints"][0][0],
                str(UserSettings.get_model_directory() / "compvis"),
            ],
            _comfy_supported_pt_extensions,
        )

        _comfy_folder_names_and_paths["diffusion_models"] = (
            [
                _comfy_folder_names_and_paths["diffusion_models"][0][0],
                str(UserSettings.get_model_directory() / "compvis"),
            ],
            _comfy_supported_pt_extensions,
        )

        _comfy_folder_names_and_paths["vae"] = (
            [
                _comfy_folder_names_and_paths["vae"][0][0],
                str(UserSettings.get_model_directory() / "vae"),
            ],
            _comfy_supported_pt_extensions,
        )

        _comfy_folder_names_and_paths["text_encoders"] = (
            [
                _comfy_folder_names_and_paths["text_encoders"][0][0],
                str(UserSettings.get_model_directory() / "text_encoders"),
            ],
            _comfy_supported_pt_extensions,
        )

        _comfy_folder_names_and_paths["upscale_models"] = (
            [
                _comfy_folder_names_and_paths["upscale_models"][0][0],
                str(UserSettings.get_model_directory() / "esrgan"),
                str(UserSettings.get_model_directory() / "gfpgan"),
                str(UserSettings.get_model_directory() / "codeformer"),
            ],
            _comfy_supported_pt_extensions,
        )

        _comfy_folder_names_and_paths["facerestore_models"] = (
            [
                str(UserSettings.get_model_directory() / "gfpgan"),
                str(UserSettings.get_model_directory() / "codeformer"),
            ],
            _comfy_supported_pt_extensions,
        )

        _comfy_folder_names_and_paths["controlnet"] = (
            [
                _comfy_folder_names_and_paths["controlnet"][0][0],
                str(UserSettings.get_model_directory() / "controlnet"),
            ],
            _comfy_supported_pt_extensions,
        )

        # Set custom node path
        _comfy_folder_names_and_paths["custom_nodes"] = (
            [
                _comfy_folder_names_and_paths["custom_nodes"][0][0],
                str(get_hordelib_path() / "nodes"),
            ],
            [],
        )

    def _this_dir(self, filename: str, subdir="") -> str:
        """Return the path to a file in the same directory as this file.

        Args:
            filename (str): The filename to return the path to.
            subdir (str, optional): The subdirectory to look in. Defaults to "".

        Returns:
            str: The path to the file.
        """
        target_dir = os.path.dirname(os.path.realpath(__file__))
        if subdir:
            target_dir = os.path.join(target_dir, subdir)
        return os.path.join(target_dir, filename)

    def _load_custom_nodes(self) -> None:
        """Force ComfyUI to load its normal custom nodes and the horde custom nodes."""
        asyncio.run(_comfy_nodes.init_extra_nodes(init_custom_nodes=True))

    def _get_executor(self):
        """Return the ComfyUI PromptExecutor object."""
        # This class (`Comfy_Horde`) uses duck typing to intercept calls intended for
        # ComfyUI's `PromptServer` class. In particular, it intercepts calls to
        # `PromptServer.send_sync`. See `Comfy_Horde.send_sync` for more details.
        # return _comfy_PromptExecutor(self)
        return _comfy_PromptExecutor(self, cache_type=_comfy_CacheType.CLASSIC, cache_args={"lru": 0, "ram": 0})

    def get_pipeline_data(self, pipeline_name):
        pipeline_data = copy.deepcopy(self.pipelines.get(pipeline_name, {}))
        if pipeline_data:
            logger.info(f"Running pipeline {pipeline_name}")
        return pipeline_data

    def _fix_pipeline_types(self, data: dict) -> dict:
        """Replace comfyui standard node types with hordelib node types.

        Args:
            data (dict): The pipeline data.

        Returns:
            dict: The pipeline resulting from the replacement.
        """
        # We have a list of nodes and each node has a class type, which we may want to change
        for nodename, node in data.items():
            if ("class_type" in node) and (node["class_type"] in Comfy_Horde.NODE_REPLACEMENTS):
                logger.debug(
                    (
                        f"Changed type {data[nodename]['class_type']} to "
                        f"{Comfy_Horde.NODE_REPLACEMENTS[node['class_type']]}"
                    ),
                )
                data[nodename]["class_type"] = Comfy_Horde.NODE_REPLACEMENTS[node["class_type"]]
        # Now we've fixed up node types, check for any node input parameter rename needed
        for nodename, node in data.items():
            if ("class_type" in node) and (node["class_type"] in Comfy_Horde.NODE_PARAMETER_REPLACEMENTS):
                for oldname, newname in Comfy_Horde.NODE_PARAMETER_REPLACEMENTS[node["class_type"]].items():
                    if "inputs" in node and oldname in node["inputs"]:
                        node["inputs"][newname] = node["inputs"][oldname]
                        # del node["inputs"][oldname]
                logger.debug(f"Renamed node input {nodename}.{oldname} to {newname}")

        return data

    def _fix_node_names(self, data: dict) -> dict:
        """Rename nodes to the "title" set in the design file.

        Args:
            data (dict): The pipeline data.
            design (dict): The design data.

        Returns:
            dict: The pipeline resulting from the renaming.
        """
        # We have a list of nodes, attempt to rename them to the "title" set
        # in the design file. These must be unique names.
        newnodes = {}
        renames = {}
        for nodename, nodedata in data.items():
            newname = nodename
            if nodedata.get("_meta", {}).get("title"):
                newname = nodedata["_meta"]["title"]
            renames[nodename] = newname
            newnodes[newname] = nodedata

        # Now we've renamed the node names, change any references to them also
        for node in newnodes.values():
            if "inputs" in node:
                for _, input in node["inputs"].items():
                    if isinstance(input, list) and input and input[0] in renames:
                        input[0] = renames[input[0]]
        return newnodes

    # We are passed a valid comfy pipeline and a design file from the comfyui web app.
    # Why?
    #
    # 1. We replace some nodes with our own hordelib nodes, for example "CheckpointLoaderSimple"
    #    with "HordeCheckpointLoader".
    # 2. We replace unfriendly node names like "3" and "7" with friendly names taken from the
    #    "title" attribute in the webui so we can have nicer parameter names when we call the
    #    inference pipeline.
    #
    # Note that point 1 does not actually need a design file, and point 2 is not technically
    # essential.
    #
    # Note also that the format of the design files from web app is expected to change at a fast
    # pace. This is why the only thing that partially relies on that format, is in fact, optional.
    def _patch_pipeline(self, data: dict) -> dict:
        """Patch the pipeline data with the design data."""
        # FIXME: This can now be done through the _meta.title key included with each API export.
        # First replace comfyui standard types with hordelib node types
        data = self._fix_pipeline_types(data)
        # Now try to find better parameter names
        return self._fix_node_names(data)

    def _load_pipeline(self, filename: str) -> bool | None:
        """
        Load a single inference pipeline from a file.

        Args:
            filename (str): The path to the pipeline file.

        Returns:
            bool | None: True if the pipeline was loaded successfully, False if it was not, None if there was an error.
        """
        # Check if the file exists
        if not os.path.exists(filename):
            logger.error(f"No such inference pipeline file: {filename}")
            return None

        try:
            # Open the pipeline file
            with open(filename) as jsonfile:
                # Extract the pipeline name from the filename
                pipeline_name_regex_matches = re.match(r".*pipeline_(.*)\.json", filename)
                if pipeline_name_regex_matches is None:
                    logger.error(f"Regex parsing failed for: {filename}")
                    return None

                pipeline_name = pipeline_name_regex_matches[1]
                # Load the pipeline data from the file
                pipeline_data = json.loads(jsonfile.read())
                # Check if there is a design file for this pipeline
                logger.debug(f"Patching pipeline {pipeline_name}")
                pipeline_data = self._patch_pipeline(pipeline_data)
                # Add the pipeline data to the pipelines dictionary
                self.pipelines[pipeline_name] = pipeline_data
                logger.debug(f"Loaded inference pipeline: {pipeline_name}")
                return True
        except (OSError, ValueError):
            logger.error(f"Invalid inference pipeline file: {filename}")
            return None

    def _load_pipelines(self) -> int:
        """Load all of the inference pipelines from the pipelines directory matching `pipeline_*.json`.

        Returns:
            int: The number of pipelines loaded.
        """
        files = glob.glob(self._this_dir("pipeline_*.json", subdir="pipelines"))
        loaded_count = 0
        for file in files:
            if self._load_pipeline(file):
                loaded_count += 1
        return loaded_count

    def _set(self, dct, **kwargs) -> None:
        """Set the named parameter to the named value in the pipeline template.

        Allows inputs to be missing from the key name, if it is we insert it.
        """
        num_skipped = 0

        for key, value in kwargs.items():
            keys = key.split(".")
            skip = False
            if "inputs" not in keys:
                keys.insert(1, "inputs")
            current = dct

            for k in keys[:-1]:
                if k not in current:
                    # logger.debug(f"Attempt to set parameter not defined in this template: {key}")
                    skip = True
                    num_skipped += 1
                    break

                current = current[k]

            if not skip:
                if not current.get(keys[-1]):
                    logger.debug(
                        f"Attempt to set parameter CREATED parameter '{key}'",
                    )
                current[keys[-1]] = value
        logger.debug(f"Attempted to set {len(kwargs)} parameters, skipped {num_skipped}")

    # Connect the named input to the named node (output).
    # Used for dynamic switching of pipeline graphs
    @classmethod
    def reconnect_input(cls, dct, input, output):
        # logger.debug(f"Request to reconnect input {input} to output {output}")

        # First check the output even exists
        if output not in dct.keys():
            logger.debug(
                f"Can not reconnect input {input} to {output} as {output} does not exist",
            )
            return None

        keys = input.split(".")
        if "inputs" not in keys:
            keys.insert(1, "inputs")
        current = dct
        for k in keys:
            if k not in current:
                logger.debug(f"Attempt to reconnect unknown input {input}")
                return None

            current = current[k]

        logger.debug(f"Request completed to reconnect input {input} to output {output}")
        current[0] = output
        return True

    _comfyui_callback: typing.Callable[[str, dict, str], None] | None = None

    # This is the callback handler for comfy async events.
    def send_sync(self, label: str, data: dict, sid: str | None = None) -> None:
        # Get receive image outputs via this async mechanism
        output = data.get("output", None)
        logger.debug([label, output])
        images_received = None
        if output is not None and "images" in output:
            images_received = output.get("images", None)

        if images_received is not None:
            if len(images_received) == 0:
                logger.warning("Received no output images from comfyui")

            for image_info in images_received:
                if not isinstance(image_info, dict):
                    logger.error(f"Received non dict output from comfyui: {image_info}")
                    continue
                for key, value in image_info.items():
                    if key == "imagedata" and isinstance(value, io.BytesIO):
                        if self.images is None:
                            self.images = []
                        self.images.append(image_info)
                    elif key == "type":
                        logger.debug(f"Received image type {value}")
                    else:
                        logger.error(f"Received unexpected image output from comfyui: {key}:{value}")
            logger.debug("Received output image(s) from comfyui")
        else:
            if self._comfyui_callback is not None and sid is not None:
                self._comfyui_callback(label, data, sid)

            if label == "execution_error":
                logger.error(f"{label}, {data}, {sid}")
                # Reset images on error so that we receive expected None input and can raise an exception
                self.images = None
            elif label != "executing":
                pass
                # logger.debug(f"{label}, {data}, {sid}")
            else:
                node_name = data.get("node", "")
                logger.debug(f"{label} comfyui node: {node_name}")
                if node_name == "vae_decode":
                    logger.info("Decoding image from VAE. This may take a while for large images.")

    # Execute the named pipeline and pass the pipeline the parameter provided.
    # For the horde we assume the pipeline returns an array of images.
    def _run_pipeline(
        self,
        pipeline: dict,
        params: dict,
        comfyui_progress_callback: typing.Callable[[ComfyUIProgress, str], None] | None = None,
    ) -> list[dict] | None:
        if _comfy_current_loaded_models is None:
            raise RuntimeError("hordelib.initialise() must be called before using comfy_horde.")
        # Wipe any previous images, if they exist.
        self.images = None

        # Set the pipeline parameters
        self._set(pipeline, **params)

        # Create (or retrieve) our prompt executor
        inference = self._get_executor()

        # This is useful for dumping the entire pipeline to the terminal when
        # developing and debugging new pipelines. A badly structured pipeline
        # file just results in a cryptic error from comfy
        # if True:  # This isn't here, Tazlin :)
        #     with open("pipeline_debug.json", "w") as outfile:
        #         default = lambda o: f"<<non-serializable: {type(o).__qualname__}>>"
        #         outfile.write(json.dumps(pipeline, indent=4, default=default))
        # pretty_pipeline = pformat(pipeline)
        # logger.warning(pretty_pipeline)

        # The client_id parameter used to only be for debugging, but is now required for all requests.
        # We pretend we are a web client and want async callbacks.
        stdio = OutputCollector(comfyui_progress_callback=comfyui_progress_callback)
        with contextlib.redirect_stdout(stdio), contextlib.redirect_stderr(stdio):
            # validate_prompt from comfy returns [bool, str, list]
            # Which gives us these nice hardcoded list indexes, which valid[2] is the output node list
            self.client_id = str(uuid.uuid4())
            valid = asyncio.run(_comfy_validate_prompt(1, pipeline, None))
            import folder_paths

            if "embeddings" in folder_paths.filename_list_cache:
                del folder_paths.filename_list_cache["embeddings"]

            try:
                with logger.catch(reraise=True):
                    inference.execute(pipeline, self.client_id, {"client_id": self.client_id}, valid[2])
            except Exception as e:
                logger.exception(f"Exception during comfy execute: {e}")
            finally:
                if self.aggressive_unloading:
                    global _comfy_cleanup_models
                    logger.debug("Cleaning up models")
                    _comfy_cleanup_models()
                    _comfy_soft_empty_cache()

        stdio.replay()

        # # Check if there are any resource to clean up
        # cleanup()
        # if time.time() - self._gc_timer > Comfy_Horde.GC_TIME:
        #     self._gc_timer = time.time()
        #     garbage_collect()
        log_free_ram()
        return self.images

    # Run a pipeline that returns an image in pixel space
    def run_image_pipeline(
        self,
        pipeline,
        params: dict,
        comfyui_progress_callback: typing.Callable[[ComfyUIProgress, str], None] | None = None,
    ) -> list[dict[str, typing.Any]]:
        # From the horde point of view, let us assume the output we are interested in
        # is always in a HordeImageOutput node named "output_image". This is an array of
        # dicts of the form:
        # [ {
        #     "imagedata": <BytesIO>,
        #     "type": "PNG"
        #   },
        # ]
        # See node_image_output.py

        # We may be passed a pipeline name or a pipeline data structure
        if isinstance(pipeline, str):
            # Grab pipeline data structure
            pipeline_data = self.get_pipeline_data(pipeline)
            # Sanity
            if not pipeline_data:
                logger.error(f"Unknown inference pipeline: {pipeline}")
                raise ValueError("Unknown inference pipeline")
        else:
            pipeline_data = pipeline

        # If no callers for a while, announce it
        if self._callers == 0 and self._exit_time:
            idle_time = time.time() - self._exit_time
            if idle_time > 1 and UserSettings.enable_idle_time_warning.active:
                logger.warning(f"No job ran for {round(idle_time, 3)} seconds")

        result = self._run_pipeline(pipeline_data, params, comfyui_progress_callback)

        if result:
            return result

        raise RuntimeError("Pipeline failed to run")


ANNOTATOR_MODEL_SHA_LOOKUP: dict[str, str] = {
    "body_pose_model.pth": "25a948c16078b0f08e236bda51a385d855ef4c153598947c28c0d47ed94bb746",
    "dpt_hybrid-midas-501f0c75.pt": "501f0c75b3bca7daec6b3682c5054c09b366765aef6fa3a09d03a5cb4b230853",
    "hand_pose_model.pth": "b76b00d1750901abd07b9f9d8c98cc3385b8fe834a26d4b4f0aad439e75fc600",
    "mlsd_large_512_fp32.pth": "5696f168eb2c30d4374bbfd45436f7415bb4d88da29bea97eea0101520fba082",
    "network-bsds500.pth": "58a858782f5fa3e0ca3dc92e7a1a609add93987d77be3dfa54f8f8419d881a94",
    "res101.pth": "1d696b2ef3e8336b057d0c15bc82d2fecef821bfebe5ef9d7671a5ec5dde520b",
    "upernet_global_small.pth": "bebfa1264c10381e389d8065056baaadbdadee8ddc6e36770d1ec339dc84d970",
}
"""The annotator precomputed SHA hashes; the dict is in the form of `{"filename": "hash", ...}."""


def download_all_controlnet_annotators() -> bool:
    """Will start the download of all the models needed for the controlnet annotators."""
    annotator_init_funcs = [
        _canny.CannyDetector,
        _hed.HEDdetector,
        _midas.MidasDetector,
        _mlsd.MLSDdetector,
        _openpose.OpenposeDetector,
        _uniformer.UniformerDetector,
        _leres.download_model_if_not_existed,
        _pidinet.download_if_not_existed,
    ]

    try:
        logger.info(
            f"Downloading {len(annotator_init_funcs)} controlnet annotators if required. Please wait.",
        )
        for i, annotator_init_func in enumerate(annotator_init_funcs):
            # Give some basic progress indication
            logger.info(
                f"{i+1} of {len(annotator_init_funcs)}",
            )
            annotator_init_func()
        return True
    except (OSError, requests.exceptions.RequestException) as e:
        logger.error(f"Failed to download annotator: {e}")

    return False
