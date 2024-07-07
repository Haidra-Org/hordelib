# comfy.py
# Wrapper around comfy to allow usage by the horde worker.
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
import requests
import psutil

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
# broken, among other things, the ability of pytest to do its test discovery because you will have lost the ability for
# modules which, directly or otherwise, import this module without having called `hordelib.initialise()`. Pytest
# discovery will come across those imports, valiantly attempt to import them and fail with a cryptic error message.
#
# Correspondingly, you will find that to be an enormous hassle if you are are trying to leverage pytest in any
# reasonability sophisticated way (outside of tox), and you will be forced to adopt solution below or something
# equally obtuse, or in lieu of either of those, abandon all hope and give up on the idea of attempting to develop any
# new features or author new tests for hordelib.
#
# Keen readers may have noticed that the aforementioned issues could be side stepped by simply calling
# `hordelib.initialise()` automatically, such as in `test/__init__.py` or in a `conftest.py`. You would be correct,
# but that would be a terrible idea if you ever intended to make alterations to the patch file, as each time you
# triggered pytest discovery which could be as frequently as *every time you save a file* (such as with VSCode), and
# you would enter a situation where the patch was automatically being applied at times you may not intend.
#
# This would be a nightmare to debug, as this author is able to attest to.
#
# Further, if you are like myself, and enjoy type hints, you will find that any modules have this file in their import
# chain will be un-importable in certain contexts and you would be unable to provide the relevant type hints.
#
# Having read this, I suggest you glance at the code in `hordelib.initialise()` to get a sense of what is going on
# there, and if you're still confused, ask a hordelib dev who would be happy to share the burden of understanding.

_comfy_load_models_gpu: types.FunctionType
_comfy_current_loaded_models: list = None  # type: ignore

_comfy_nodes: types.ModuleType
_comfy_PromptExecutor: typing.Any
_comfy_validate_prompt: types.FunctionType

_comfy_folder_names_and_paths: dict[str, tuple[list[str], list[str] | set[str]]]
_comfy_supported_pt_extensions: set[str]

_comfy_load_checkpoint_guess_config: types.FunctionType

_comfy_get_torch_device: types.FunctionType
_comfy_get_free_memory: types.FunctionType
_comfy_get_total_memory: types.FunctionType
_comfy_load_torch_file: types.FunctionType
_comfy_model_loading: types.ModuleType
_comfy_free_memory: types.FunctionType
_comfy_cleanup_models: types.FunctionType
_comfy_soft_empty_cache: types.FunctionType

_comfy_recursive_output_delete_if_changed: types.FunctionType

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

    @logger.catch(default=True, reraise=True)
    def emit(self, record):
        # Get corresponding Loguru level if it exists.
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message.
        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


# ComfyUI uses stdlib logging, so we need to intercept it.
logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)


def do_comfy_import(
    force_normal_vram_mode: bool = False,
    extra_comfyui_args: list[str] | None = None,
    disable_smart_memory: bool = False,
) -> None:
    global _comfy_current_loaded_models
    global _comfy_load_models_gpu
    global _comfy_nodes, _comfy_PromptExecutor, _comfy_validate_prompt
    global _comfy_recursive_output_delete_if_changed
    global _comfy_folder_names_and_paths, _comfy_supported_pt_extensions
    global _comfy_load_checkpoint_guess_config
    global _comfy_get_torch_device, _comfy_get_free_memory, _comfy_get_total_memory
    global _comfy_load_torch_file, _comfy_model_loading
    global _comfy_free_memory, _comfy_cleanup_models, _comfy_soft_empty_cache
    global _canny, _hed, _leres, _midas, _mlsd, _openpose, _pidinet, _uniformer

    if disable_smart_memory:
        logger.info("Disabling smart memory")
        sys.argv.append("--disable-smart-memory")

    if force_normal_vram_mode:
        logger.info("Forcing normal vram mode")
        sys.argv.append("--normalvram")

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
        from execution import nodes as _comfy_nodes
        from execution import PromptExecutor as _comfy_PromptExecutor
        from execution import validate_prompt as _comfy_validate_prompt
        from execution import recursive_output_delete_if_changed

        _comfy_recursive_output_delete_if_changed = recursive_output_delete_if_changed  # type: ignore
        execution.recursive_output_delete_if_changed = recursive_output_delete_if_changed_hijack
        from folder_paths import folder_names_and_paths as _comfy_folder_names_and_paths  # type: ignore
        from folder_paths import supported_pt_extensions as _comfy_supported_pt_extensions  # type: ignore
        from comfy.sd import load_checkpoint_guess_config as _comfy_load_checkpoint_guess_config
        from comfy.model_management import current_loaded_models as _comfy_current_loaded_models
        from comfy.model_management import load_models_gpu as _comfy_load_models_gpu
        from comfy.model_management import get_torch_device as _comfy_get_torch_device
        from comfy.model_management import get_free_memory as _comfy_get_free_memory
        from comfy.model_management import get_total_memory as _comfy_get_total_memory
        from comfy.model_management import free_memory as _comfy_free_memory
        from comfy.model_management import cleanup_models as _comfy_cleanup_models
        from comfy.model_management import soft_empty_cache as _comfy_soft_empty_cache
        from comfy.utils import load_torch_file as _comfy_load_torch_file
        from comfy_extras.chainner_models import model_loading as _comfy_model_loading  # type: ignore
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

        import comfy.model_management

        # comfy.model_management.vram_state = comfy.model_management.VRAMState.HIGH_VRAM
        # comfy.model_management.set_vram_to = comfy.model_management.VRAMState.HIGH_VRAM

        logger.info("Comfy_Horde initialised")

        # def always_cpu(parameters, dtype):
        # return torch.device("cpu")

        # comfy.model_management.unet_inital_load_device = always_cpu
        # comfy.model_management.DISABLE_SMART_MEMORY = True
        # comfy.model_management.lowvram_available = True

        # comfy.model_management.unet_offload_device = _unet_offload_device_hijack

    log_free_ram()
    output_collector.replay()


# isort: on

_last_pipeline_settings_hash = ""


def recursive_output_delete_if_changed_hijack(prompt: dict, old_prompt, outputs, current_item):
    global _last_pipeline_settings_hash
    if current_item == "prompt":
        try:
            pipeline_settings_hash = hashlib.md5(json.dumps(prompt).encode("utf-8")).hexdigest()
            logger.debug(f"pipeline_settings_hash: {pipeline_settings_hash}")

            if pipeline_settings_hash != _last_pipeline_settings_hash:
                _last_pipeline_settings_hash = pipeline_settings_hash
                logger.debug("Pipeline settings changed")

            if old_prompt:
                old_pipeline_settings_hash = hashlib.md5(json.dumps(old_prompt).encode("utf-8")).hexdigest()
                logger.debug(f"old_pipeline_settings_hash: {old_pipeline_settings_hash}")
                if pipeline_settings_hash != old_pipeline_settings_hash:
                    logger.debug("Pipeline settings changed from old_prompt")
        except TypeError:
            logger.debug("could not print hash due to source image in payload")
    if current_item == "prompt" or current_item == "negative_prompt":
        try:
            prompt_text = prompt[current_item]["inputs"]["text"]
            prompt_hash = hashlib.md5(prompt_text.encode("utf-8")).hexdigest()
            logger.debug(f"{current_item} hash: {prompt_hash}")
        except KeyError:
            pass

    global _comfy_recursive_output_delete_if_changed
    return _comfy_recursive_output_delete_if_changed(prompt, old_prompt, outputs, current_item)


# def cleanup():
# _comfy_soft_empty_cache()


def unload_all_models_vram():
    from hordelib.shared_model_manager import SharedModelManager

    logger.debug("In unload_all_models_vram")

    logger.debug(f"{len(_comfy_current_loaded_models)} models loaded in comfy")
    # _comfy_free_memory(_comfy_get_total_memory(), _comfy_get_torch_device())
    with torch.no_grad():
        try:
            for model in _comfy_current_loaded_models:
                model.model_unload()
            _comfy_soft_empty_cache()
        except Exception as e:
            logger.error(f"Exception during comfy unload: {e}")
            _comfy_cleanup_models()
            _comfy_soft_empty_cache()
    logger.debug(f"{len(_comfy_current_loaded_models)} models loaded in comfy")


def unload_all_models_ram():
    from hordelib.shared_model_manager import SharedModelManager

    logger.debug("In unload_all_models_ram")

    SharedModelManager.manager._models_in_ram = {}
    logger.debug(f"{len(_comfy_current_loaded_models)} models loaded in comfy")
    with torch.no_grad():
        _comfy_free_memory(_comfy_get_total_memory(), _comfy_get_torch_device())
        _comfy_cleanup_models()
        _comfy_soft_empty_cache()
    logger.debug(f"{len(_comfy_current_loaded_models)} models loaded in comfy")


def get_torch_device():
    return _comfy_get_torch_device()


def get_torch_total_vram_mb():
    return round(_comfy_get_total_memory() / (1024 * 1024))


def get_torch_free_vram_mb():
    return round(_comfy_get_free_memory() / (1024 * 1024))


def log_free_ram():
    logger.debug(f"Free VRAM: {get_torch_free_vram_mb():0.0f} MB")
    logger.debug(f"Free RAM: {psutil.virtual_memory().available / (1024 * 1024):0.0f} MB")


class Comfy_Horde:
    """Handles horde-specific behavior against ComfyUI."""

    # We save pipelines from the ComfyUI GUI. This is very convenient as it
    # makes it super easy to load and edit them in the future. We allow the pipeline
    # design with standard nodes, then at runtime we dynamically replace these
    # node types with our own where we need to. This allows easy previewing in ComfyUI
    # which our custom nodes don't allow.
    NODE_REPLACEMENTS = {
        "CheckpointLoaderSimple": "HordeCheckpointLoader",
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

        stdio = OutputCollector()
        with contextlib.redirect_stdout(stdio):
            # Load our custom nodes
            self._load_custom_nodes()
        stdio.replay()

        self._comfyui_callback = comfyui_callback

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

        _comfy_folder_names_and_paths["upscale_models"] = (
            [
                _comfy_folder_names_and_paths["upscale_models"][0][0],
                str(UserSettings.get_model_directory() / "esrgan"),
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
        _comfy_nodes.init_extra_nodes(init_custom_nodes=True)

    def _get_executor(self):
        """Return the ComfyUI PromptExecutor object."""
        # This class (`Comfy_Horde`) uses duck typing to intercept calls intended for
        # ComfyUI's `PromptServer` class. In particular, it intercepts calls to
        # `PromptServer.send_sync`. See `Comfy_Horde.send_sync` for more details.
        return _comfy_PromptExecutor(self)

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
    def send_sync(self, label: str, data: dict, _id: str) -> None:
        # Get receive image outputs via this async mechanism
        if "output" in data and "images" in data["output"]:
            images_received = data["output"]["images"]
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
            if self._comfyui_callback is not None:
                self._comfyui_callback(label, data, _id)

            if label == "execution_error":
                logger.error(f"{label}, {data}, {_id}")
                # Reset images on error so that we receive expected None input and can raise an exception
                self.images = None
            elif label != "executing":
                logger.debug(f"{label}, {data}, {_id}")
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

        # The client_id parameter here is just so we receive comfy callbacks for debugging.
        # We pretend we are a web client and want async callbacks.
        stdio = OutputCollector(comfyui_progress_callback=comfyui_progress_callback)
        with contextlib.redirect_stdout(stdio), contextlib.redirect_stderr(stdio):
            # validate_prompt from comfy returns [bool, str, list]
            # Which gives us these nice hardcoded list indexes, which valid[2] is the output node list
            self.client_id = str(uuid.uuid4())
            valid = _comfy_validate_prompt(pipeline)
            import folder_paths

            if "embeddings" in folder_paths.filename_list_cache:
                del folder_paths.filename_list_cache["embeddings"]

            try:
                with logger.catch(reraise=True):
                    inference.execute(pipeline, self.client_id, {"client_id": self.client_id}, valid[2])
            except Exception as e:
                logger.exception(f"Exception during comfy execute: {e}")

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
