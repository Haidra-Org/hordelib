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
from hordelib.execution.comfy_events import (
    HORDE_MODEL_NAME_INPUT,
    MODEL_LOADER_NODE_TITLE,
    VAE_DECODE_NODE_TITLE,
    ExecutingEvent,
    ExecutionCachedEvent,
    ExecutionErrorEvent,
    ExecutionInterruptedEvent,
    ExecutionSuccessEvent,
    ValidationResult,
    parse_event,
)
from hordelib.execution.graph_utils import GraphDict, apply_dotted_params
from hordelib.execution.interface import DEFAULT_IMAGE_OUTPUTS, OutputSpec
from hordelib.execution.model_dirs import ModelCategory, invalidate_filename_cache, register_horde_model_paths
from hordelib.execution.results import PipelineRunResult, collect_output_entries
from hordelib.execution.server_shim import HeadlessComfyServer
from hordelib.pipeline.graph import HORDE_NODE_REPLACEMENTS
from hordelib.utils.memory_trim import trim_host_memory
from hordelib.utils.torch_memory import clear_accelerator_cache

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

_comfy_load_checkpoint_guess_config: types.FunctionType

_comfy_get_torch_device: types.FunctionType
"""Will return the current torch device, typically the GPU."""
_comfy_get_free_memory: types.FunctionType
"""Will return the amount of free memory on the current torch device. This value can be misleading."""
_comfy_get_total_memory: types.FunctionType
"""Will return the total amount of memory on the current torch device."""
_comfy_load_torch_file: types.FunctionType
_comfy_model_loading: types.ModuleType
_comfy_free_memory: Callable[..., None]
"""Will aggressively unload models from memory.

Called as ``(memory_required, device)``; the pristine signature (with its optional
keep-loaded/pinning parameters) is pinned by ``tests/test_comfy_contract_drift.py``.
"""
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

# Some comfy stdlib loggers emit DEBUG records on the per-forward-pass hot path (e.g.
# comfy_kitchen.dispatch logs a "Backend ... selected" line on every RoPE application,
# thousands per job). Raise their level so the records are never created, keeping them out
# of every sink at near-zero cost. Add further per-op loggers here if they surface.
_NOISY_COMFY_LOGGERS = ("comfy_kitchen.dispatch",)
for _noisy_logger_name in _NOISY_COMFY_LOGGERS:
    logging.getLogger(_noisy_logger_name).setLevel(logging.INFO)


def do_comfy_import(
    force_normal_vram_mode: bool = False,
    extra_comfyui_args: list[str] | None = None,
    disable_smart_memory: bool = False,
) -> None:
    global _comfy_current_loaded_models
    global _comfy_execution
    global _comfy_nodes, _comfy_PromptExecutor, _comfy_validate_prompt
    global _comfy_nodes_images
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

    # A CPU-only torch build cannot reach any accelerator, but ComfyUI's cpu_state defaults to GPU and
    # only flips to CPU on the --cpu flag (never from torch.cuda.is_available() being False). Without
    # --cpu it takes the CUDA branch in get_torch_device() and dies with "No CUDA GPUs are available"
    # during hordelib's startup VRAM probe. Inject the flag here, the single point every entry path
    # (initialise() and direct do_comfy_import callers) funnels through.
    from hordelib.utils.torch_memory import torch_build_is_cpu_only

    if torch_build_is_cpu_only():
        if extra_comfyui_args is None:
            extra_comfyui_args = ["--cpu"]
        elif "--cpu" not in extra_comfyui_args:
            extra_comfyui_args = [*extra_comfyui_args, "--cpu"]
        logger.info("CPU-only torch build detected; starting ComfyUI in CPU mode (--cpu).")

    if extra_comfyui_args is not None:
        sys.argv.extend(extra_comfyui_args)

    # ComfyUI imports torchaudio eagerly in several node modules loaded just below. torchaudio is an
    # optional, non-default dependency (no cu132 wheel; audio unsupported), so install a lazy stub when
    # it is absent, BEFORE comfy is imported, so those imports succeed for image/video work.
    from hordelib.utils.torch_build import ensure_torchaudio_importable

    ensure_torchaudio_importable()

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


def clear_gc_and_torch_cache(trim_host: bool = False) -> None:
    """Clear the garbage collector and the active backend's device cache.

    When ``trim_host`` is set, additionally ask the OS to reclaim this process's freed heap and cold
    mmap-faulted pages (see :func:`hordelib.utils.memory_trim.trim_host_memory`) after the collect and
    device-cache clear, so the process's measured host residency reflects live data. The trim defaults off
    so existing callers keep their exact behavior; enable it only at unload or idle boundaries, since
    reclaimed cold pages refault on demand.
    """
    gc.collect()
    clear_accelerator_cache()
    if trim_host:
        trim_host_memory()


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


def unload_all_models_vram() -> None:
    global _comfy_current_loaded_models

    from hordelib.metrics import get_metrics_collector

    _unload_start = time.perf_counter()

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

    # Surface the VRAM-eviction cost to the in-process collector so an embedder can attribute the
    # between-jobs reload churn (this is the "unload" half; the reload is already timed as
    # disk_to_ram/ram_to_vram). Recorded under the job in progress when the unload happens, or the
    # next job on this process when it is an idle eviction.
    get_metrics_collector().record_phase("model_unload", time.perf_counter() - _unload_start)


def unload_all_models_ram() -> None:
    global _comfy_current_loaded_models

    log_free_ram()
    from hordelib.shared_model_manager import SharedModelManager

    logger.debug("In unload_all_models_ram")
    logger.debug(
        "Models cached in shared model manager: count={}",
        len(SharedModelManager.manager._models_in_ram),
    )

    SharedModelManager.manager._models_in_ram.evict_all()
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

    # RAM unload is a terminal boundary: the weights just freed here are cold, so trim the host working
    # set to return their pages to the OS rather than let them ratchet the process's resident set upward.
    clear_gc_and_torch_cache(trim_host=True)
    log_free_ram()


def get_torch_device() -> torch.device:
    return _comfy_get_torch_device()


def get_torch_total_vram_mb() -> int:
    return round(_comfy_get_total_memory() / (1024 * 1024))


def get_torch_free_vram_mb() -> int:
    return round(_comfy_get_free_memory() / (1024 * 1024))


def get_torch_device_free_vram_mb() -> int:
    """Device-wide free VRAM (mem_get_info), excluding this process's reclaimable torch cache.

    See :func:`hordelib.utils.torch_memory.get_torch_device_free_vram_mb`. This is the honest figure for
    over-commit/streaming decisions; ``get_torch_free_vram_mb`` (comfy's view) adds back a single
    process's cache and so over-states cross-process headroom.
    """
    from hordelib.utils.torch_memory import get_torch_device_free_vram_mb as _device_free

    return _device_free()


_LAST_LOW_VRAM_WARN_TIME: float = 0.0
"""Monotonic-ish wall time of the last low-free-VRAM warning, so a job that runs its whole length below the
inference reserve warns periodically rather than on every (very frequent) log_free_ram call."""
_LOW_VRAM_WARN_INTERVAL_SECONDS: float = 15.0


def log_free_ram() -> None:
    global _LAST_LOW_VRAM_WARN_TIME

    # Report the *device-wide* free VRAM (mem_get_info), not comfy's get_free_memory: the latter adds back
    # this process's reserved-but-inactive torch allocator cache, over-stating free by several GB and
    # hiding VRAM held by other (including leaked) processes. The driver's system-memory fallback triggers
    # on the device-wide figure, so that is the honest number for a streaming/over-commit warning.
    free_vram_mb = get_torch_device_free_vram_mb()
    comfy_free_vram_mb = get_torch_free_vram_mb()
    reclaimable_cache_mb = max(0.0, comfy_free_vram_mb - free_vram_mb)
    free_ram_mb = psutil.virtual_memory().available / (1024 * 1024)

    offloaded_mb = 0
    try:
        from hordelib.utils.torch_memory import get_loaded_weights_offloaded_mb

        offloaded_mb = get_loaded_weights_offloaded_mb()
    except Exception:
        offloaded_mb = 0
    offloaded_note = f", Weights offloaded to RAM: {offloaded_mb:0.0f} MB" if offloaded_mb > 0 else ""
    # Surface the gap between the device-wide free and comfy's view so a live run shows how much of the
    # apparent headroom is merely this process's reclaimable cache (returned by empty_cache), not room a
    # sibling model load could actually use.
    cache_note = (
        f" (+{reclaimable_cache_mb:0.0f} MB reclaimable torch cache; comfy sees {comfy_free_vram_mb:0.0f})"
        if reclaimable_cache_mb >= 256
        else ""
    )

    logger.debug(f"Free VRAM: {free_vram_mb:0.0f} MB{cache_note}, Free RAM: {free_ram_mb:0.0f} MB{offloaded_note}")

    # A driver-level system-memory fallback (e.g. the Windows WDDM shared-memory pool, on by default) spills
    # device allocations to host RAM once free VRAM nears zero, collapsing the sampling rate the same way
    # ComfyUI's own weight offloading does but *without* appearing in its offload accounting. ComfyUI can
    # report a model as fully resident while the driver pages its per-step activations. So warn purely on
    # measured free VRAM falling below the inference working-set reserve: below that floor a sampling step has
    # no room for its activations and something (ComfyUI or the driver) must stream over the bus. Throttled so
    # a job that stays under the floor for its whole run periodically reports rather than flooding the log.
    try:
        from hordelib.vram_planning import compute_inference_reserve_mb

        reserve_mb = compute_inference_reserve_mb(get_torch_total_vram_mb())
    except Exception:
        return
    if free_vram_mb < reserve_mb:
        now = time.time()
        if now - _LAST_LOW_VRAM_WARN_TIME >= _LOW_VRAM_WARN_INTERVAL_SECONDS:
            _LAST_LOW_VRAM_WARN_TIME = now
            if offloaded_mb > 0:
                cause = f"ComfyUI reports {offloaded_mb:0.0f} MB of weights offloaded"
            else:
                cause = "ComfyUI reports no offload, so the GPU driver's system-memory fallback is the likely cause"
            logger.warning(
                f"Free VRAM {free_vram_mb:0.0f} MB is below the {reserve_mb:0.0f} MB inference reserve: "
                f"sampling activations will stream from host RAM and run several times slower ({cause}).",
            )


def interrupt_comfyui_processing() -> None:
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


def _extract_graph_input(pipeline: GraphDict, node_title: str, input_name: str) -> typing.Any | None:
    """Return a node input value from a materialized graph, or None if absent."""
    node = pipeline.get(node_title)
    if not isinstance(node, dict):
        return None
    inputs = node.get("inputs")
    if not isinstance(inputs, dict):
        return None
    return inputs.get(input_name)


# Module-level metrics for ComfyUI pipeline performance tracking
pipeline_duration_histogram = logfire.metric_histogram(
    "comfy.pipeline.duration_ms",
    unit="ms",
    description="ComfyUI pipeline execution duration",
)


class Comfy_Horde:
    """Handles horde-specific behavior against ComfyUI."""

    NODE_REPLACEMENTS = HORDE_NODE_REPLACEMENTS
    """ComfyUI standard node types replaced by Horde-specific implementations at graph load.

    The single source of truth lives in :mod:`hordelib.pipeline.graph`; this alias remains for
    any external readers of the historical class attribute.
    """

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

        # The named server contract ComfyUI's executor runs against when embedded; events
        # land in send_sync below.
        self._server_shim = HeadlessComfyServer(event_listener=self.send_sync)

        # Register horde model directories (and the custom-node path) with ComfyUI.
        register_horde_model_paths()

        # Load our custom nodes
        self._load_custom_nodes()

        self._comfyui_callback = comfyui_callback
        self.aggressive_unloading = aggressive_unloading

    def _load_custom_nodes(self) -> None:
        """Force ComfyUI to load its normal custom nodes and the horde custom nodes."""
        asyncio.run(_comfy_nodes.init_extra_nodes(init_custom_nodes=True))

    def _get_executor(self) -> typing.Any:
        """Return a fresh ComfyUI PromptExecutor for one run.

        A new executor (and so a new cache set) is built per run on purpose: cross-run node
        caching would pin tensors in RAM/VRAM against the worker's aggressive unload policy,
        and executor construction is cheap.

        The executor runs against the HeadlessComfyServer shim in place of ComfyUI's
        PromptServer; its events arrive in send_sync below.
        """
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

        return _comfy_PromptExecutor(self._server_shim, cache_type=cache_type, cache_args=cache_args)

    _comfyui_callback: typing.Callable[[str, dict, str], None] | None = None

    def send_sync(self, label: str, data: dict, sid: str | None = None) -> None:
        """Receive one execution event from ComfyUI, via the HeadlessComfyServer shim.

        Artifact collection reads the executor's ``history_result`` after the run (see
        ``_collect_run_result``); this channel only logs typed event context and forwards the
        raw event to the external callback.
        """
        logger.debug(
            "ComfyUI callback",
            label=label,
            client_id=sid,
            data_keys=list(data.keys()),
        )

        if self._comfyui_callback is not None and sid is not None:
            self._comfyui_callback(label, data, sid)

        event = parse_event(label, data)

        if isinstance(event, ExecutionErrorEvent):
            logger.error(
                "ComfyUI node execution failed",
                client_id=sid,
                node_id=event.node_id,
                node_type=event.node_type,
                exception_type=event.exception_type,
                exception_message=event.exception_message,
                traceback=event.traceback if event.traceback else None,
            )
        elif isinstance(event, ExecutionInterruptedEvent):
            logger.warning(
                "ComfyUI execution interrupted",
                client_id=sid,
                node_id=event.node_id,
                node_type=event.node_type,
            )
        elif isinstance(event, ExecutionSuccessEvent):
            logger.info("ComfyUI execution completed successfully", client_id=sid)
        elif isinstance(event, ExecutionCachedEvent):
            # Comfy emits execution_cached with an empty node list when nothing was
            # actually cached, so only treat a non-empty list as noteworthy.
            if event.nodes:
                logger.info(
                    "ComfyUI execution fully cached",
                    client_id=sid,
                    cached_nodes=event.nodes,
                )
                logger.warning(
                    "All nodes were cached - this may indicate a problem if new images were expected",
                    cached_nodes=event.nodes,
                )
        elif isinstance(event, ExecutingEvent):
            logger.debug("ComfyUI executing node", node_name=event.node)
            if event.node == VAE_DECODE_NODE_TITLE:
                logger.info("Decoding image from VAE. This may take a while for large images.")

    @staticmethod
    def _collect_run_result(executor: typing.Any) -> PipelineRunResult:
        """Build the typed run result from the executor's post-run attributes.

        ComfyUI's ``PromptExecutor`` exposes ``success``, ``status_messages`` and
        ``history_result`` after ``execute()`` returns; ``history_result`` is absent only
        when an exception escaped the executor entirely (there is no finally-assignment),
        which is treated as failure.
        """
        history_result = getattr(executor, "history_result", None)
        success = bool(getattr(executor, "success", False)) and history_result is not None

        entries: list[dict[str, typing.Any]] = []
        if history_result is not None:
            entries = collect_output_entries(history_result.get("outputs", {}))

        error_event: ExecutionErrorEvent | None = None
        for event_label, event_data in getattr(executor, "status_messages", []):
            parsed_event = parse_event(event_label, event_data)
            if isinstance(parsed_event, ExecutionErrorEvent):
                error_event = parsed_event
                break
            if isinstance(parsed_event, ExecutionInterruptedEvent):
                # Interrupts share the failure path; there is no exception context to carry.
                break

        return PipelineRunResult(success=success, entries=entries, error=error_event)

    # Execute a fully materialized graph, applying any remaining dotted params.
    @logfire.instrument("comfy.execute_graph", extract_args=False)
    def _run_pipeline(
        self,
        pipeline: GraphDict,
        params: dict[str, typing.Any],
        comfyui_progress_callback: typing.Callable[[ComfyUIProgress, str], None] | None = None,
        *,
        defer_vram_unload: bool = False,
    ) -> PipelineRunResult:
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

        # Set any remaining dotted parameters on the materialized graph
        apply_dotted_params(pipeline, params)

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

            _t_validate = time.perf_counter()
            validation = ValidationResult.from_comfy(asyncio.run(_comfy_validate_prompt(1, pipeline, None)))
            _validate_seconds = time.perf_counter() - _t_validate

            logger.info(
                "Pipeline validation result",
                is_valid=validation.is_valid,
                error_message=validation.error if validation.error else None,
                error_details=validation.error.get("details") if validation.error else None,
                error_type=validation.error.get("type") if validation.error else None,
                output_node_count=len(validation.output_node_ids),
                output_nodes=validation.output_node_ids,
                node_error_count=len(validation.node_errors),
            )

            if not validation.is_valid:
                logger.error(
                    "Pipeline validation failed",
                    validation_error=validation.error,
                    client_id=self.client_id,
                    node_errors=validation.node_errors,
                    pipeline_node_types=pipeline_node_types,
                )
                logger.error(
                    "Pipeline validation failed summary",
                    validation_error=validation.error,
                    node_count=len(pipeline),
                    node_types=list(set(pipeline_node_types.values())),
                )
            # Textual inversions can be downloaded between jobs; force a rescan so the run
            # sees embeddings that appeared since the last listing.
            invalidate_filename_cache(ModelCategory.EMBEDDINGS)

            _t_pre_execute = time.perf_counter()
            try:
                with logger.catch(reraise=True):
                    # client_id in extra_data is required: ComfyUI only delivers cached output
                    # nodes into history_result when the server has a client_id (pinned by
                    # tests/test_comfy_contract_drift.py).
                    inference.execute(
                        pipeline,
                        self.client_id,
                        {"client_id": self.client_id},
                        validation.output_node_ids,
                    )
            except Exception as exc:
                logger.exception("Exception during comfy execute")
                logger.error("ComfyUI execution failed", error=str(exc))
            finally:
                _t_post_execute = time.perf_counter()
                _execute_seconds = _t_post_execute - _t_pre_execute
                if use_native_progress:
                    set_run_progress_callback(None)
                # ``aggressive_unloading`` evicts the just-used model from VRAM after every job so N
                # sibling ComfyUI instances sharing one GPU never collectively over-commit. ``defer_vram_unload``
                # lets the host keep the model resident across this job when it knows the same model runs next
                # and the VRAM budget allows it, so the back-to-back force-reload (the dominant non-sampling cost
                # on small jobs) is skipped. The host owns the safety decision; here we only honor it.
                if self.aggressive_unloading and not defer_vram_unload:
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
        except Exception:
            pass

        # Record pipeline duration
        duration_ms = (time.time() - start_time) * 1000
        pipeline_duration_histogram.record(duration_ms)

        run_result = self._collect_run_result(inference)

        logger.info(
            "ComfyUI pipeline execution complete",
            duration_ms=duration_ms,
            success=run_result.success,
            artifact_count=len(run_result.entries),
            client_id=self.client_id,
        )

        log_free_ram()
        return run_result

    # Run a materialized pipeline graph and collect its declared outputs
    @logfire.instrument("comfy.run_pipeline", extract_args=False)
    def run_pipeline(
        self,
        pipeline: GraphDict,
        params: dict[str, typing.Any],
        comfyui_progress_callback: typing.Callable[[ComfyUIProgress, str], None] | None = None,
        *,
        outputs: tuple[OutputSpec, ...] = DEFAULT_IMAGE_OUTPUTS,
        defer_vram_unload: bool = False,
    ) -> list[dict[str, typing.Any]]:
        # Results are collected from the declared output nodes as dicts of the form
        # {"imagedata": <BytesIO>, "type": "PNG", "source_node": <node title>}, read from the
        # executor's history_result; see hordelib.execution.results and node_image_output.py.
        if not isinstance(pipeline, dict):
            raise TypeError(
                f"run_pipeline expects a materialized graph dict, got {type(pipeline).__name__!r}. "
                "Named pipelines were removed; materialize a graph via the pipeline registry or "
                "hordelib.pipeline.graph.ComfyGraph instead.",
            )

        logger.info(
            "Pipeline starting",
            params_keys=list(params.keys()),
            declared_outputs=[output.node for output in outputs],
        )

        run_result = self._run_pipeline(
            pipeline,
            params,
            comfyui_progress_callback,
            defer_vram_unload=defer_vram_unload,
        )

        produced_nodes = run_result.produced_nodes
        missing_nodes = [output.node for output in outputs if output.node not in produced_nodes]

        if run_result.success and run_result.entries and not missing_nodes:
            declared_nodes = {output.node for output in outputs}
            undeclared_nodes = produced_nodes - declared_nodes
            if undeclared_nodes:
                # Behavior-preserving fallback: collected but flagged, so stray SaveImage-style
                # nodes surface during the migration to declared outputs.
                logger.warning(
                    "Pipeline produced results from undeclared output nodes",
                    undeclared_nodes=sorted(undeclared_nodes),
                    declared_nodes=sorted(declared_nodes),
                )
            return run_result.entries

        model_name = _extract_graph_input(pipeline, MODEL_LOADER_NODE_TITLE, HORDE_MODEL_NAME_INPUT) or "unknown"
        error_summary = f" Error: {run_result.error.summary()}" if run_result.error else ""
        logger.error(
            "Pipeline execution failed - declared outputs missing",
            success=run_result.success,
            produced_count=len(run_result.entries),
            missing_output_nodes=missing_nodes,
            model_name=model_name,
            error_summary=run_result.error.summary() if run_result.error else None,
        )
        raise RuntimeError(
            f"Pipeline failed to run - declared output node(s) {missing_nodes} produced no results. "
            f"Model: {model_name}.{error_summary}",
        )


# Backwards-compatible re-exports: the monkeypatch machinery now lives in
# hordelib.execution.comfy_patches.
from hordelib.execution.comfy_patches import (
    get_monkeypatch_names,
    get_monkeypatch_state,
    set_monkeypatch_state,
    temporary_monkeypatch_state,
)


def __getattr__(name: str) -> typing.Any:
    # Forward reads of moved attributes so existing imports keep working for one release.
    if name in ("models_not_to_force_load", "disable_force_loading"):
        from hordelib.execution import comfy_patches

        return getattr(comfy_patches, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
