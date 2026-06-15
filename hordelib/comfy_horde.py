# comfy.py
# Wrapper around comfy to allow usage by the horde worker.
import asyncio
import contextlib
import importlib
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
import logfire
import loguru
import psutil
from collections.abc import Callable

import torch
from loguru import logger

from hordelib.settings import UserSettings
from hordelib.utils.ioredirect import ComfyUIProgress, OutputCollector
from hordelib.config_path import get_hordelib_path
from hordelib.execution import comfy_patches
from hordelib.execution.graph_utils import GraphDict

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

_comfy_current_loaded_models: list = None  # type: ignore

_comfy_nodes: types.ModuleType
_comfy_PromptExecutor: typing.Any
_comfy_validate_prompt: types.FunctionType

_comfy_execution: types.ModuleType
_comfy_nodes_images: types.ModuleType

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

_comfy_interrupt_current_processing: types.FunctionType


# isort: off

import inspect
import logging


class InterceptHandler(logging.Handler):
    """
    Add logging handler to augment python stdlib logging.

    Logs which would otherwise go to stdlib logging are redirected through
    loguru.

    The handler walks the call stack (skipping stdlib logging internals and frozen importlib
    frames) so loguru attributes each message to its original caller. The stdlib-computed
    source location is additionally attached as bound context (``stdlib_*`` keys).

    Performance note: Profiling shows that the logging overhead is dominated by loguru's internal
    processing (~48%), LogRecord creation (~19%), and file operations (~17%). The handler's emit
    method accounts for only ~5% of overhead. See docs/intercept_handler_final_report.md for
    detailed analysis.
    """

    _ignored_message_contents: set[str]
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
        self._ignored_message_contents.add(content)

    def get_ignored_message_contents(self) -> list[str]:
        """Return the list of ignored message contents."""
        return list(self._ignored_message_contents)

    def reset_ignored_message_contents(self) -> None:
        """Reset the list of ignored message contents."""
        self._ignored_message_contents.clear()

    def add_ignored_library(self, library: str) -> None:
        """Add a library to ignore."""
        self._ignored_libraries.append(library)

    def get_ignored_libraries(self) -> list[str]:
        """Return the list of ignored libraries."""
        return self._ignored_libraries.copy()

    def reset_ignored_libraries(self) -> None:
        """Reset the list of ignored libraries."""
        self._ignored_libraries.clear()

    @logger.catch(default=True, reraise=True)
    def emit(self, record: logging.LogRecord) -> None:
        """
        Emit a log record to loguru.

        This method filters messages and forwards them to loguru, walking the call stack so
        loguru attributes the message to the original caller rather than this handler. Frozen
        importlib frames are skipped too, so logs emitted during module import are attributed
        to the importing module instead of ``importlib._bootstrap``.

        Args:
            record: The LogRecord containing message, level, and source location
        """
        # Filter by library name
        library = record.name
        for ignored_library in self._ignored_libraries:
            if ignored_library in library:
                return

        # Filter by message content
        message = record.getMessage()
        for ignored_message_content in self._ignored_message_contents:
            if ignored_message_content in message:
                return

        # Get corresponding Loguru level if it exists
        level: str | int
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Keep the stdlib-computed source location available as bound context regardless of
        # what the frame walk below resolves to.
        logger_context = logger.bind(
            stdlib_pathname=record.pathname,
            stdlib_lineno=record.lineno,
            stdlib_funcname=record.funcName,
            stdlib_loggername=record.name,
        )

        frame = inspect.currentframe()
        depth = 0
        while frame:
            filename = frame.f_code.co_filename
            module_name = frame.f_globals.get("__name__", "")
            is_logging_internal = filename == logging.__file__
            # The @logger.catch decorator on this method inserts loguru frames into the chain
            is_loguru_internal = module_name == "loguru" or module_name.startswith("loguru.")
            is_frozen_importlib = "importlib" in filename and "_bootstrap" in filename
            if depth > 0 and not (is_logging_internal or is_loguru_internal or is_frozen_importlib):
                break
            frame = frame.f_back
            depth += 1

        logger_context.opt(depth=depth, exception=record.exc_info).log(level, message)


intercept_handler = InterceptHandler()
# ComfyUI uses stdlib logging, so we need to intercept it.
logging.basicConfig(handlers=[intercept_handler], level=0, force=True)


def do_comfy_import(
    force_normal_vram_mode: bool = False,
    extra_comfyui_args: list[str] | None = None,
    disable_smart_memory: bool = False,
) -> None:
    global _comfy_current_loaded_models
    global _comfy_execution
    global _comfy_nodes, _comfy_PromptExecutor, _comfy_validate_prompt
    global _comfy_nodes_images
    global _comfy_folder_names_and_paths, _comfy_supported_pt_extensions
    global _comfy_load_checkpoint_guess_config
    global _comfy_get_torch_device, _comfy_get_free_memory, _comfy_get_total_memory
    global _comfy_load_torch_file, _comfy_model_loading
    global _comfy_free_memory, _comfy_cleanup_models, _comfy_soft_empty_cache

    global _comfy_interrupt_current_processing

    if disable_smart_memory:
        logger.info("Disabling smart memory")
        sys.argv.append("--disable-smart-memory")

    if force_normal_vram_mode:
        # ComfyUI removed the `--normalvram` flag; normal VRAM mode is now the default
        # (overridable only via --gpu-only/--highvram/--lowvram/--novram/--cpu), so there
        # is nothing to force anymore.
        logger.info("Normal vram mode requested (ComfyUI default; no flag needed)")

    if extra_comfyui_args is not None:
        sys.argv.extend(extra_comfyui_args)

    # Note these imports are intentionally somewhat obfuscated as a reminder to other modules
    # that they should never call through this module into comfy directly. All calls into
    # comfy should be abstracted through functions in this module.
    output_collector = OutputCollector()
    with contextlib.redirect_stdout(output_collector), contextlib.redirect_stderr(output_collector):
        from comfy.options import enable_args_parsing

        enable_args_parsing()
        import execution

        global _comfy_execution
        _comfy_execution = execution
        from execution import nodes as _comfy_nodes
        from execution import PromptExecutor as _comfy_PromptExecutor
        from execution import validate_prompt as _comfy_validate_prompt

        # from execution import recursive_output_delete_if_changed
        from execution import IsChangedCache

        comfy_patches.register_execution_module(execution)
        comfy_patches.capture_and_patch(
            "is_changed_cache_get",
            IsChangedCache,
            "get",
            comfy_patches.IsChangedCache_get_hijack,
        )

        from folder_paths import folder_names_and_paths as _comfy_folder_names_and_paths
        from folder_paths import supported_pt_extensions as _comfy_supported_pt_extensions
        from comfy.sd import load_checkpoint_guess_config as _comfy_load_checkpoint_guess_config
        from comfy.model_management import current_loaded_models as _comfy_current_loaded_models
        import comfy.model_management

        comfy_patches.capture_and_patch(
            "load_models_gpu",
            comfy.model_management,
            "load_models_gpu",
            comfy_patches._load_models_gpu_hijack,
        )
        # Fail fast if the comfy model_base class names the force-load hijack relies on have
        # drifted from this ComfyUI version, rather than silently force-loading an oversized model.
        comfy_patches.assert_force_load_class_names_exist()
        from comfy.model_management import get_torch_device as _comfy_get_torch_device
        from comfy.model_management import get_free_memory as _comfy_get_free_memory
        from comfy.model_management import get_total_memory as _comfy_get_total_memory
        from comfy.model_management import free_memory as _comfy_free_memory
        from comfy.model_management import cleanup_models as _comfy_cleanup_models
        from comfy.model_management import soft_empty_cache as _comfy_soft_empty_cache
        from comfy.model_management import interrupt_current_processing as _comfy_interrupt_current_processing

        comfy_patches.capture_and_patch(
            "text_encoder_initial_device",
            comfy.model_management,
            "text_encoder_initial_device",
            comfy_patches.text_encoder_initial_device_hijack,
        )

        from comfy.utils import load_torch_file as _comfy_load_torch_file
        from comfy_extras.chainner_models import model_loading as _comfy_model_loading

        from comfy.model_patcher import ModelPatcher

        comfy_patches.capture_and_patch(
            "model_patcher_load",
            ModelPatcher,
            "load",
            comfy_patches._model_patcher_load_hijack,
        )

        import comfy.lora

        comfy_patches.capture_and_patch(
            "lora_calculate_weight",
            comfy.lora,
            "calculate_weight",
            comfy_patches._calculate_weight_hijack,
        )

        # Ensure comfy_extras nodes register themselves before pipelines run.
        # _comfy_nodes_images = importlib.import_module("comfy_extras.nodes_images")
        import comfy_extras.nodes_images as _comfy_nodes_images

        # NOTE: The vendored comfy_controlnet_preprocessors package has been removed. ControlNet
        # preprocessing is unavailable until the comfyui_controlnet_aux migration lands (refactor P3).

        logger.info("Comfy_Horde initialised")

    # Now that ComfyUI is fully imported, apply deep instrumentation
    try:
        from hordelib.integrations.logfire_setup import initialize_comfy_internals

        initialize_comfy_internals()
    except Exception as e:
        logger.warning("Failed to initialize ComfyUI instrumentation", error=str(e))

    log_free_ram()
    output_collector.replay()


# isort: on


def clear_gc_and_torch_cache() -> None:
    """Clear the garbage collector and the PyTorch cache."""
    gc.collect()
    from torch.cuda import empty_cache

    empty_cache()


def pin_models_in_vram() -> bool:
    """Keep loaded models resident in VRAM (ComfyUI HIGH_VRAM mode).

    By default ComfyUI runs NORMAL_VRAM, which offloads the model to system RAM between
    jobs (to make room for the VAE etc.) and transfers it back to VRAM on the next job — a
    per-job RAM->VRAM cost that dominates non-sampling time on small jobs. HIGH_VRAM keeps
    everything resident so back-to-back jobs skip that reload.

    Only safe when the model(s) comfortably fit in VRAM, so the worker gates this behind
    ``high_memory_mode``. Returns True if the mode was set.
    """
    try:
        from comfy import model_management as mm

        mm.vram_state = mm.VRAMState.HIGH_VRAM
        logger.info("Pinned ComfyUI to HIGH_VRAM: models stay resident across jobs (no per-job RAM->VRAM reload)")
        return True
    except Exception as e:
        logger.warning(f"Could not pin models in VRAM: {e}")
        return False


def unload_all_models_vram():
    global _comfy_current_loaded_models

    log_free_ram()

    from hordelib.shared_model_manager import SharedModelManager

    logger.debug("In unload_all_models_vram")
    logger.debug(
        "Models cached in shared model manager: count={}",
        len(SharedModelManager.manager._models_in_ram),
    )

    logger.debug("Models loaded in comfy: count={}", len(_comfy_current_loaded_models))

    logger.debug("Freeing memory on all devices")
    _comfy_free_memory(1e30, _comfy_get_torch_device())
    log_free_ram()

    logger.debug("Cleaning up models")
    with torch.no_grad():
        try:
            with logfire.span("comfy.soft_empty_cache"):
                _comfy_soft_empty_cache()
            log_free_ram()
        except Exception:
            logger.exception("Exception during comfy unload")
            with logfire.span("comfy.cleanup_models_fallback"):
                _comfy_cleanup_models()
                _comfy_soft_empty_cache()

    logger.debug(
        "Models cached in shared model manager: count={}",
        len(SharedModelManager.manager._models_in_ram),
    )
    logger.debug("Models loaded in comfy: count={}", len(_comfy_current_loaded_models))
    logger.info(
        "comfy.models_after_unload",
        cached_models=len(SharedModelManager.manager._models_in_ram),
        loaded_models=len(_comfy_current_loaded_models),
    )

    with logfire.span("comfy.gc_and_torch_cache"):
        clear_gc_and_torch_cache()
    log_free_ram()


def unload_all_models_ram():
    global _comfy_current_loaded_models

    log_free_ram()
    from hordelib.shared_model_manager import SharedModelManager

    logger.debug("In unload_all_models_ram")
    logger.debug(
        "Models cached in shared model manager: count={}",
        len(SharedModelManager.manager._models_in_ram),
    )

    SharedModelManager.manager._models_in_ram = {}
    logger.debug("Models loaded in comfy: count={}", len(_comfy_current_loaded_models))
    all_devices = set()
    for model in _comfy_current_loaded_models:
        all_devices.add(model.device)

    all_devices.add(get_torch_device())

    with torch.no_grad():
        for device in all_devices:
            logger.debug("Freeing memory on device: device={}", device)
            _comfy_free_memory(1e30, device)

        log_free_ram()
        logger.debug("Freeing memory on CPU")
        _comfy_free_memory(1e30, torch.device("cpu"))

        log_free_ram()

        logger.debug("Cleaning up models")
        with logfire.span("comfy.cleanup_all_models"):
            _comfy_cleanup_models()
        log_free_ram()

        logger.debug("Soft emptying cache")
        _comfy_soft_empty_cache()
        log_free_ram()

    logger.debug(
        "Models cached in shared model manager: count={}",
        len(SharedModelManager.manager._models_in_ram),
    )
    logger.debug("Models loaded in comfy: count={}", len(_comfy_current_loaded_models))

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


def get_node_class(class_type: str) -> type:
    """Return a registered ComfyUI node class by its class_type name.

    Only valid after initialise() and after custom nodes have been loaded
    (which happens when the first Comfy_Horde instance is constructed).
    """
    if _comfy_nodes is None:
        raise RuntimeError("hordelib.initialise() must be called before looking up node classes.")
    node_class = _comfy_nodes.NODE_CLASS_MAPPINGS.get(class_type)
    if node_class is None:
        raise RuntimeError(
            f"ComfyUI node class '{class_type}' is not registered. "
            "Custom nodes may not have been loaded yet (construct Comfy_Horde/HordeLib first).",
        )
    return node_class


# Module-level metrics for ComfyUI pipeline performance tracking
pipeline_duration_histogram = logfire.metric_histogram(
    "comfy.pipeline.duration_ms",
    unit="ms",
    description="ComfyUI pipeline execution duration",
)


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
        self._exit_time = 0
        self._callers = 0
        self._gc_timer = time.time()
        self._counter_mutex = threading.Lock()
        self.images = None

        # Set comfyui paths for checkpoints, loras, etc
        self._set_comfyui_paths()

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
        from comfy.cli_args import args

        cache_lru: int = getattr(args, "cache_lru", 0)
        # `--cache-ram` is a list of 0-2 thresholds in GB (active, inactive) in current ComfyUI
        cache_ram_arg: list[float] = getattr(args, "cache_ram", [])

        # Unlike upstream (which defaults to RAM_PRESSURE), hordelib defaults to CLASSIC: the worker
        # manages memory aggressively itself and re-runs near-identical pipelines back to back.
        cache_type = _comfy_execution.CacheType.CLASSIC
        if cache_lru > 0:
            cache_type = _comfy_execution.CacheType.LRU
        elif getattr(args, "cache_none", False):
            cache_type = _comfy_execution.CacheType.NONE
        elif cache_ram_arg:
            cache_type = _comfy_execution.CacheType.RAM_PRESSURE

        # Mirrors ComfyUI main.prompt_worker's cache_args construction
        cache_ram = 0.0
        cache_ram_inactive = 0.0
        if cache_type == _comfy_execution.CacheType.RAM_PRESSURE:
            import comfy.model_management

            total_ram = comfy.model_management.total_ram
            cache_ram = min(10.0, max(2.0, total_ram * 0.10 / 1024.0))
            cache_ram_inactive = min(96.0, total_ram / 1024.0)
            if len(cache_ram_arg) > 0:
                cache_ram = cache_ram_arg[0]
            if len(cache_ram_arg) > 1:
                cache_ram_inactive = cache_ram_arg[1]

        cache_args = {"lru": cache_lru, "ram": cache_ram, "ram_inactive": cache_ram_inactive}

        return _comfy_PromptExecutor(self, cache_type=cache_type, cache_args=cache_args)

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
                    logger.debug("Template parameter created: key={}", key)
                current[keys[-1]] = value
        logger.debug(
            "Attempted to set parameters: requested_count={}, skipped_count={}",
            len(kwargs),
            num_skipped,
        )

    # Connect the named input to the named node (output).
    # Used for dynamic switching of pipeline graphs
    @classmethod
    def reconnect_input(cls, dct, input, output):
        # logger.debug(f"Request to reconnect input {input} to output {output}")

        # First check the output even exists
        if output not in dct.keys():
            logger.debug(
                "Cannot reconnect input to missing output",
                input_name=input,
                output_name=output,
            )
            return None

        keys = input.split(".")
        if "inputs" not in keys:
            keys.insert(1, "inputs")
        current = dct
        for k in keys:
            if k not in current:
                logger.debug("Attempt to reconnect unknown input: input_name={}", input)
                return None

            current = current[k]

        logger.debug("Reconnected input to output", input_name=input, output_name=output)
        current[0] = output
        return True

    _comfyui_callback: typing.Callable[[str, dict, str], None] | None = None

    # This is the callback handler for comfy async events.
    def send_sync(self, label: str, data: dict, sid: str | None = None) -> None:
        # Log all execution state changes for debugging
        logger.debug(
            "ComfyUI callback",
            label=label,
            client_id=sid,
            has_output=data.get("output") is not None,
            data_keys=list(data.keys()),
        )

        # Get receive image outputs via this async mechanism
        output = data.get("output", None)
        logger.debug(
            "ComfyUI callback raw output",
            label=label,
            output_present=output is not None,
            output_type=type(output).__name__ if output is not None else None,
        )
        images_received = None
        if output is not None and "images" in output:
            images_received = output.get("images", None)

        if images_received is not None:
            if len(images_received) == 0:
                logger.warning("Received no output images from comfyui")

            for image_info in images_received:
                if not isinstance(image_info, dict):
                    logger.error("Received non dict output from comfyui: output={}", image_info)
                    continue
                for key, value in image_info.items():
                    if key == "imagedata" and isinstance(value, io.BytesIO):
                        if self.images is None:
                            self.images = []
                        self.images.append(image_info)
                    elif key == "type":
                        logger.debug("Received image type: value={}", value)
                    else:
                        logger.error(
                            "Received unexpected image output from comfyui",
                            key=key,
                            value=value,
                        )
            logger.debug("Received output image(s) from comfyui")
        else:
            if self._comfyui_callback is not None and sid is not None:
                self._comfyui_callback(label, data, sid)

            if label == "execution_error":
                # Extract detailed error information
                node_id = data.get("node_id", "unknown")
                node_type = data.get("node_type", "unknown")
                exception_message = data.get("exception_message", "")
                exception_type = data.get("exception_type", "")
                traceback_text = data.get("traceback", "")

                logger.error(
                    "ComfyUI execution error",
                    node_id=node_id,
                    node_type=node_type,
                    exception_message=exception_message,
                )

                logger.error(
                    "ComfyUI node execution failed",
                    label=label,
                    client_id=sid,
                    node_id=node_id,
                    node_type=node_type,
                    exception_type=exception_type,
                    exception_message=exception_message,
                    traceback=traceback_text if traceback_text else None,
                    full_error_data=data,
                )

                # Reset images on error so that we receive expected None input and can raise an exception
                self.images = None
            elif label == "execution_success":
                # Log successful execution
                logger.info(
                    "ComfyUI execution completed successfully",
                    client_id=sid,
                    has_images=self.images is not None,
                    image_count=len(self.images) if self.images else 0,
                )
            elif label == "execution_cached":
                # Comfy emits execution_cached with an empty node list when nothing was
                # actually cached, so only treat a non-empty list as noteworthy.
                cached_nodes = data.get("nodes", [])
                logger.info(
                    "ComfyUI execution fully cached",
                    client_id=sid,
                    cached_nodes=cached_nodes,
                )
                if cached_nodes:
                    logger.warning(
                        "All nodes were cached - this may indicate a problem if new images were expected",
                        cached_nodes=cached_nodes,
                    )
            elif label != "executing":
                pass
                # logger.debug(f"{label}, {data}, {sid}")
            else:
                node_name = data.get("node", "")
                logger.debug("ComfyUI executing node", label=label, node_name=node_name)
                if node_name == "vae_decode":
                    logger.info("Decoding image from VAE. This may take a while for large images.")

    # Execute a fully materialized graph, applying any remaining dotted params.
    # For the horde we assume the pipeline returns an array of images.
    @logfire.instrument("comfy.execute_graph", extract_args=False)
    def _run_pipeline(
        self,
        pipeline: GraphDict,
        params: dict[str, typing.Any],
        comfyui_progress_callback: typing.Callable[[ComfyUIProgress, str], None] | None = None,
    ) -> list[dict[str, typing.Any]] | None:
        start_time = time.time()
        _t0 = time.perf_counter()
        _validate_seconds = 0.0
        _execute_seconds = 0.0
        _t_pre_execute = _t0
        _t_post_execute = _t0

        if _comfy_current_loaded_models is None:
            raise RuntimeError("hordelib.initialise() must be called before using comfy_horde.")

        # Generate client_id early for logging
        self.client_id = str(uuid.uuid4())

        logger.info(
            "ComfyUI pipeline execution starting",
            client_id=self.client_id,
            aggressive_unload=self.aggressive_unloading,
            node_count=len(pipeline),
        )

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

        # Progress goes through ComfyUI's native hook when installed (one stream, no tqdm
        # parsing); the OutputCollector then only captures/replays log output. Without the
        # hook, the collector's tqdm parser remains the progress source.
        from hordelib.execution.progress_hook import is_native_hook_installed, set_run_progress_callback

        use_native_progress = is_native_hook_installed()
        if use_native_progress:
            set_run_progress_callback(comfyui_progress_callback)

        # The client_id parameter used to only be for debugging, but is now required for all requests.
        # We pretend we are a web client and want async callbacks.
        stdio = OutputCollector(
            comfyui_progress_callback=None if use_native_progress else comfyui_progress_callback,
        )
        with contextlib.redirect_stdout(stdio), contextlib.redirect_stderr(stdio):
            # Log pipeline structure before validation for debugging
            pipeline_node_types = {
                node_id: node_info.get("class_type", "unknown") for node_id, node_info in pipeline.items()
            }
            logger.debug(
                "Pipeline structure before validation",
                node_count=len(pipeline),
                node_types=pipeline_node_types,
            )

            # validate_prompt from comfy returns [bool, str, list]
            # Which gives us these nice hardcoded list indexes, which valid[2] is the output node list
            _t_validate = time.perf_counter()
            valid = asyncio.run(_comfy_validate_prompt(1, pipeline, None))
            _validate_seconds = time.perf_counter() - _t_validate

            # Log validation results with structured data
            validation_status = valid[0] if len(valid) > 0 else None
            validation_error = valid[1] if len(valid) > 1 else None
            output_nodes = valid[2] if len(valid) > 2 else []
            node_errors = valid[3] if len(valid) > 3 else {}

            logger.info(
                "Pipeline validation result",
                is_valid=validation_status,
                error_message=validation_error if validation_error else None,
                error_details=validation_error.get("details") if isinstance(validation_error, dict) else None,
                error_type=validation_error.get("type") if isinstance(validation_error, dict) else None,
                output_node_count=len(output_nodes) if output_nodes else 0,
                output_nodes=output_nodes if output_nodes else [],
                node_error_count=len(node_errors) if node_errors else 0,
            )

            if not validation_status:
                logger.error(
                    "Pipeline validation failed",
                    validation_error=validation_error,
                    client_id=self.client_id,
                    node_errors=node_errors,
                    pipeline_node_types=pipeline_node_types,
                )
                logger.error(
                    "Pipeline validation failed summary",
                    validation_error=validation_error,
                    node_count=len(pipeline),
                    node_types=list(set(pipeline_node_types.values())),
                )
            import folder_paths

            if "embeddings" in folder_paths.filename_list_cache:
                del folder_paths.filename_list_cache["embeddings"]

            _t_pre_execute = time.perf_counter()
            try:
                with logger.catch(reraise=True):
                    inference.execute(pipeline, self.client_id, {"client_id": self.client_id}, valid[2])
            except Exception as exc:
                logger.exception("Exception during comfy execute")
                logger.error("ComfyUI execution failed", error=str(exc))
            finally:
                _t_post_execute = time.perf_counter()
                _execute_seconds = _t_post_execute - _t_pre_execute
                if use_native_progress:
                    set_run_progress_callback(None)
                if self.aggressive_unloading:
                    global _comfy_cleanup_models
                    logger.debug("Cleaning up models")
                    with logfire.span("comfy.cleanup"):
                        _comfy_cleanup_models()
                        _comfy_soft_empty_cache()

        stdio.replay()

        # Per-job phase attribution (additive instrumentation; must never break a run). "setup"
        # is everything before graph execution (prompt set, executor build, validate); "validate"
        # is the asyncio validate_prompt subset of setup; "execute" is the graph run (sampling +
        # VAE + CLIP encode live inside it and are recorded separately); "finalize" is stdout
        # replay and teardown. The worker-side result hand-off shows up as the inference window
        # minus the sum of these.
        try:
            from hordelib.metrics import get_metrics_collector

            _mc = get_metrics_collector()
            _mc.record_phase("pipeline_setup", _t_pre_execute - _t0)
            _mc.record_phase("pipeline_validate", _validate_seconds)
            _mc.record_phase("pipeline_execute", _execute_seconds)
            _mc.record_phase("pipeline_finalize", time.perf_counter() - _t_post_execute)
        except Exception:  # noqa: BLE001 - instrumentation must never break a run
            pass

        # Record pipeline duration
        duration_ms = (time.time() - start_time) * 1000
        pipeline_duration_histogram.record(duration_ms)

        # Validate execution results
        images_generated = self.images is not None and len(self.images) > 0 if self.images else False
        image_count = len(self.images) if self.images else 0

        logger.info(
            "ComfyUI pipeline execution complete",
            duration_ms=duration_ms,
            images_generated=images_generated,
            image_count=image_count,
            client_id=self.client_id,
        )

        # # Check if there are any resource to clean up
        # cleanup()
        # if time.time() - self._gc_timer > Comfy_Horde.GC_TIME:
        #     self._gc_timer = time.time()
        #     garbage_collect()
        log_free_ram()
        return self.images

    # Run a pipeline that returns an image in pixel space
    @logfire.instrument("comfy.run_pipeline", extract_args=False)
    def run_image_pipeline(
        self,
        pipeline: GraphDict,
        params: dict[str, typing.Any],
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
        if not isinstance(pipeline, dict):
            raise TypeError(
                f"run_image_pipeline expects a materialized graph dict, got {type(pipeline).__name__!r}. "
                "Named pipelines were removed; materialize a graph via the pipeline registry or "
                "hordelib.pipeline.graph.ComfyGraph instead.",
            )

        logger.info(
            "Pipeline starting",
            params_keys=list(params.keys()),
        )

        # If no callers for a while, announce it
        if self._callers == 0 and self._exit_time:
            idle_time = time.time() - self._exit_time
            if idle_time > 1 and UserSettings.enable_idle_time_warning.active:
                logger.warning("No job ran recently", idle_seconds=round(idle_time, 3))

        result = self._run_pipeline(pipeline, params, comfyui_progress_callback)

        if result:
            return result

        # Pipeline failed - provide detailed error context
        logger.error(
            "Pipeline execution failed - no images produced",
            result_is_none=result is None,
            result_type=type(result).__name__ if result is not None else "None",
            params_provided=list(params.keys()),
            param_count=len(params),
        )

        # Log the most critical params for debugging
        model_name = params.get("model_loader.horde_model_name", "unknown")
        steps = params.get("sampler.steps", "unknown")
        resolution = f"{params.get('empty_latent_image.width', '?')}x{params.get('empty_latent_image.height', '?')}"

        logger.error(
            "Pipeline failed to produce images",
            model_name=model_name,
            steps=steps,
            resolution=resolution,
        )

        raise RuntimeError(
            f"Pipeline failed to run - no images were produced. "
            f"Model: {model_name}, Steps: {steps}, Resolution: {resolution}",
        )


# Backwards-compatible re-exports: the monkeypatch machinery now lives in
# hordelib.execution.comfy_patches.
from hordelib.execution.comfy_patches import (
    get_monkeypatch_names,
    get_monkeypatch_state,
    set_monkeypatch_state,
    temporary_monkeypatch_state,
)


def __getattr__(name: str):
    # Forward reads of moved attributes so existing imports keep working for one release.
    if name in ("models_not_to_force_load", "disable_force_loading"):
        from hordelib.execution import comfy_patches

        return getattr(comfy_patches, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
