# comfy.py
# Wrapper around comfy to allow usage by the horde worker.
import contextlib
import copy
import gc
import glob
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

import torch
from loguru import logger

from hordelib.settings import UserSettings
from hordelib.utils.ioredirect import OutputCollector
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


_comfy_nodes: types.ModuleType
_comfy_PromptExecutor: types.ModuleType
_comfy_validate_prompt: types.FunctionType
_comfy_folder_paths: types.ModuleType
_comfy_load_checkpoint_guess_config: types.FunctionType
_comfy_load_controlnet: types.FunctionType
_comfy_model_manager: types.ModuleType | None = None
_comfy_get_torch_device: types.FunctionType
_comfy_get_free_memory: types.FunctionType
_comfy_load_torch_file: types.FunctionType
_comfy_model_loading: types.ModuleType
_canny: types.ModuleType
_hed: types.ModuleType
_leres: types.ModuleType
_midas: types.ModuleType
_mlsd: types.ModuleType
_openpose: types.ModuleType
_pidinet: types.ModuleType
_uniformer: types.ModuleType


# isort: off
def do_comfy_import():
    global _comfy_nodes, _comfy_PromptExecutor, _comfy_validate_prompt, _comfy_folder_paths
    global _comfy_load_checkpoint_guess_config, _comfy_load_controlnet, _comfy_model_manager
    global _comfy_get_torch_device, _comfy_get_free_memory, _comfy_load_torch_file, _comfy_model_loading
    global _canny, _hed, _leres, _midas, _mlsd, _openpose, _pidinet, _uniformer

    # Note these imports are intentionally somewhat obfuscated as a reminder to other modules
    # that they should never call through this module into comfy directly. All calls into
    # comfy should be abstracted through functions in this module.

    from execution import nodes as _comfy_nodes  # type: ignore
    from execution import PromptExecutor as _comfy_PromptExecutor  # type: ignore
    from execution import validate_prompt as _comfy_validate_prompt  # type: ignore
    from folder_paths import folder_names_and_paths as _comfy_folder_paths  # type: ignore
    from comfy.sd import load_checkpoint_guess_config as _comfy_load_checkpoint_guess_config  # type: ignore
    from comfy.sd import load_controlnet as _comfy_load_controlnet  # type: ignore
    from comfy.model_management import model_manager as _comfy_model_manager  # type: ignore
    from comfy.model_management import get_torch_device as _comfy_get_torch_device  # type: ignore
    from comfy.model_management import get_free_memory as _comfy_get_free_memory  # type: ignore
    from comfy.utils import load_torch_file as _comfy_load_torch_file  # type: ignore
    from comfy_extras.chainner_models import model_loading as _comfy_model_loading  # type: ignore
    from hordelib.nodes.comfy_controlnet_preprocessors import (
        canny as _canny,  # type: ignore
        hed as _hed,  # type: ignore
        leres as _leres,  # type: ignore
        midas as _midas,  # type: ignore
        mlsd as _mlsd,  # type: ignore
        openpose as _openpose,  # type: ignore
        pidinet as _pidinet,  # type: ignore
        uniformer as _uniformer,  # type: ignore
    )


# isort: on

__models_to_release: dict[str, dict[str, typing.Any]] = {}
__model_load_mutex = threading.Lock()


def cleanup():
    locked = _comfy_model_manager.sampler_mutex.acquire(timeout=0)  # type: ignore
    if locked:
        # Do we have any models waiting to be released?
        if not __models_to_release:
            _comfy_model_manager.sampler_mutex.release()  # type: ignore
            return

        # Can we release any of them?
        for model_name, model_data in __models_to_release.copy().items():
            if is_model_in_use(model_data["model"]):
                # We're in the middle of using it, nothing we can do
                continue
            # Unload the model from the GPU
            logger.debug(f"Unloading {model_name} from GPU")
            unload_model_from_gpu(model_data["model"])
            # Free ram
            if "model" in model_data:
                del model_data["model"]
            if "clip" in model_data:
                del model_data["clip"]
            if "vae" in model_data:
                del model_data["vae"]
            if "clipVisionModel" in model_data:
                del model_data["clipVisionModel"]
            del model_data
            with __model_load_mutex:
                del __models_to_release[model_name]
            gc.collect()
            logger.debug(f"Removal of model {model_name} completed")

        _comfy_model_manager.sampler_mutex.release()  # type: ignore


def remove_model_from_memory(model_name, model_data):
    logger.debug(f"Comfy_Horde received request to unload {model_name}")
    with __model_load_mutex:
        if model_name not in __models_to_release:
            logger.debug(f"Model {model_name} queued for GPU/RAM unload")
            __models_to_release[model_name] = model_data


def get_models_on_gpu():
    return _comfy_model_manager.get_models_on_gpu()  # type: ignore


def get_torch_device():
    return _comfy_get_torch_device()


def get_torch_free_vram_mb():
    return round(_comfy_get_free_memory() / (1024 * 1024))


def unload_model_from_gpu(model):
    _comfy_model_manager.unload_model(model)  # type: ignore
    garbage_collect()


def garbage_collect():
    gc.collect()
    if not torch.cuda.is_available():
        return None
    if torch.version.cuda:
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def load_model_to_gpu(model):
    # Don't bother if there isn't any space
    if not _comfy_model_manager.have_free_vram():  # type: ignore
        return
    # Load the model to the GPU. This would normally be done just before
    # the model is used for sampling, the caller must want more free ram.
    return _comfy_model_manager.load_model_gpu(model)  # type: ignore


def is_model_in_use(model):
    return _comfy_model_manager.is_model_in_use(model)  # type: ignore


def load_torch_file(filename):
    result = _comfy_load_torch_file(filename)
    return result


def load_state_dict(state_dict):
    result = _comfy_model_loading.load_state_dict(state_dict)
    return result


def horde_load_checkpoint(
    ckpt_path: str,
    output_vae: bool = True,
    output_clip: bool = True,
    embeddings_path: str | None = None,
) -> dict[str, typing.Any]:  # XXX # FIXME 'any'
    # XXX Needs docstring
    # XXX # TODO One day this signature should be generic, and not comfy specific
    # XXX # This can remain a comfy call, but the rest of the code should be able
    # XXX # to pretend it isn't
    # Redirect IO
    try:
        stdio = OutputCollector()
        with contextlib.redirect_stdout(stdio):
            (modelPatcher, clipModel, vae, clipVisionModel) = _comfy_load_checkpoint_guess_config(
                ckpt_path=ckpt_path,
                output_vae=output_vae,
                output_clip=output_clip,
                embedding_directory=embeddings_path,
            )
    except RuntimeError:
        # Failed, hard to tell why, bad checkpoint, not enough ram, for example
        raise

    return {
        "model": modelPatcher,
        "clip": clipModel,
        "vae": vae,
        "clipVisionModel": clipVisionModel,
    }


def horde_load_controlnet(controlnet_path: str, target_model):  # XXX Needs docstring
    # Redirect IO
    stdio = OutputCollector()
    with contextlib.redirect_stdout(stdio):
        controlnet = _comfy_load_controlnet(ckpt_path=controlnet_path, model=target_model)
    return controlnet


class Comfy_Horde:
    """Handles horde-specific behavior against ComfyUI."""

    # We save pipelines from the ComfyUI GUI. This is very convenient as it
    # makes it super easy to load and edit them in the future. We allow the pipeline
    # design with standard nodes, then at runtime we dynamically replace these
    # node types with our own where we need to. This allows easy previewing in ComfyUI
    # which our custom nodes don't allow.
    NODE_REPLACEMENTS = {
        "CheckpointLoaderSimple": "HordeCheckpointLoader",
        "UpscaleModelLoader": "HordeUpscaleModelLoader",
        "SaveImage": "HordeImageOutput",
        "LoadImage": "HordeImageLoader",
        "DiffControlNetLoader": "HordeDiffControlNetLoader",
        "LoraLoader": "HordeLoraLoader",
    }

    # We may wish some ComfyUI standard nodes had different names for consistency. Here
    # we can dynamically rename some node input parameter names.
    NODE_PARAMETER_REPLACEMENTS = {
        "HordeCheckpointLoader": {
            # We name this "model_name" as then we can use the same generic code in our model loaders
            "ckpt_name": "model_name",
        },
    }

    # Approximate number of seconds to force garbage collection
    GC_TIME = 30

    _property_mutex = threading.Lock()

    # We maintain one "client_id" per thread
    @property
    def client_id(self):
        with Comfy_Horde._property_mutex:
            tid = threading.current_thread().ident
            return self._client_id.get(tid)

    @client_id.setter
    def client_id(self, client_id):
        with Comfy_Horde._property_mutex:
            tid = threading.current_thread().ident
            self._client_id[tid] = client_id

    # We maintain one "images" for outputs per thread
    @property
    def images(self):
        with Comfy_Horde._property_mutex:
            tid = threading.current_thread().ident
            images = self._images.get(tid)
            if tid in self._images:
                del self._images[tid]
            return images

    @images.setter
    def images(self, images):
        with Comfy_Horde._property_mutex:
            tid = threading.current_thread().ident
            self._images[tid] = images

    def __init__(self) -> None:
        if _comfy_model_manager is None:
            raise RuntimeError("hordelib.initialise() must be called before using comfy_horde.")
        self._client_id = {}  # type: ignore
        self._images = {}  # type: ignore
        self.pipelines = {}  # type: ignore
        self._model_locks = []  # type: ignore
        self._model_lock_mutex = threading.Lock()
        self._exit_time = 0
        self._callers = 0
        self._gc_timer = time.time()
        self._counter_mutex = threading.Lock()
        # FIXME Temporary hack to set the model dir for LORAs
        _comfy_folder_paths["loras"] = (  # type: ignore
            [os.path.join(UserSettings.get_model_directory(), "loras")],
            [".safetensors"],
        )
        # Set custom node path
        _comfy_folder_paths["custom_nodes"] = ([os.path.join(get_hordelib_path(), "nodes")], [])  # type: ignore
        # Load our pipelines
        self._load_pipelines()

        stdio = OutputCollector()
        with contextlib.redirect_stdout(stdio):
            # Load our custom nodes
            self._load_custom_nodes()
        stdio.replay()

    def lock_models(self, model_names):
        """Lock the given model, or list of models.

        Return True if the models could be locked, False indicates they are in use already
        """
        already_locked = False
        with self._model_lock_mutex:
            if isinstance(model_names, str):
                model_names = [model_names]
            for model in model_names:
                if model in self._model_locks:
                    already_locked = True
                    break
            if not already_locked:
                self._model_locks.extend(model_names)

        return not already_locked

    def unlock_models(self, model_names):
        """The reverse of _lock_models(). Although this can never fail so returns nothing."""
        with self._model_lock_mutex:
            if isinstance(model_names, str):
                model_names = [model_names]
            for model in model_names:
                if model in self._model_locks:
                    self._model_locks.remove(model)

    def _this_dir(self, filename: str, subdir="") -> str:
        target_dir = os.path.dirname(os.path.realpath(__file__))
        if subdir:
            target_dir = os.path.join(target_dir, subdir)
        return os.path.join(target_dir, filename)

    def _load_custom_nodes(self) -> None:
        _comfy_nodes.init_custom_nodes()

    def _get_executor(self, pipeline):
        executor = _comfy_PromptExecutor(self)  # type: ignore
        return executor

    def get_pipeline_data(self, pipeline_name):
        pipeline_data = copy.deepcopy(self.pipelines.get(pipeline_name, {}))
        if pipeline_data:
            logger.info(f"Running pipeline {pipeline_name}")
        return pipeline_data

    def _fix_pipeline_types(self, data: dict) -> dict:
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
                        del node["inputs"][oldname]
                logger.debug(f"Renamed node input {nodename}.{oldname} to {newname}")  # type: ignore # FIXME

        return data

    def _fix_node_names(self, data: dict, design: dict) -> dict:
        # We have a list of nodes, attempt to rename them to the "title" set
        # in the design file. These must be unique names.
        newnodes = {}
        renames = {}
        nodes = design["nodes"]
        for nodename, oldnode in data.items():
            newname = nodename
            for node in nodes:
                if str(node["id"]) == str(nodename) and "title" in node:
                    newname = node["title"]
                    break
            renames[nodename] = newname
            newnodes[newname] = oldnode
        # Now we've renamed the node names, change any references to them also
        for node in newnodes.values():
            if "inputs" in node:
                for _, input in node["inputs"].items():
                    if type(input) is list and input and input[0] in renames:
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
    def _patch_pipeline(self, data: dict, design: dict) -> dict:
        # First replace comfyui standard types with hordelib node types
        data = self._fix_pipeline_types(data)
        # Now try to find better parameter names
        data = self._fix_node_names(data, design)
        return data

    def _load_pipeline(self, filename: str) -> bool | None:
        if not os.path.exists(filename):
            logger.error(f"No such inference pipeline file: {filename}")
            return None

        try:
            with open(filename) as jsonfile:
                pipeline_name_rematches = re.match(r".*pipeline_(.*)\.json", filename)
                if pipeline_name_rematches is None:
                    logger.error(f"Regex parsing failed for: {filename}")
                    return None

                pipeline_name = pipeline_name_rematches[1]
                data = json.loads(jsonfile.read())
                # Do we have a design file for this pipeline?
                design = os.path.join(
                    os.path.dirname(os.path.dirname(filename)),
                    "pipeline_designs",
                    os.path.basename(filename),
                )
                # If we do have a design pipeline, use it to patch the pipeline we loaded.
                if os.path.exists(design):
                    logger.debug(f"Patching pipeline {pipeline_name}")
                    with open(design) as designfile:
                        designdata = json.loads(designfile.read())
                    data = self._patch_pipeline(data, designdata)
                self.pipelines[pipeline_name] = data
                logger.debug(f"Loaded inference pipeline: {pipeline_name}")
                return True
        except (OSError, ValueError):
            logger.error(f"Invalid inference pipeline file: {filename}")
            return None

    def _load_pipelines(self) -> int:
        files = glob.glob(self._this_dir("pipeline_*.json", subdir="pipelines"))
        loaded_count = 0
        for file in files:
            if self._load_pipeline(file):
                loaded_count += 1
        return loaded_count

    # Inject parameters into a pre-configured pipeline
    # We allow "inputs" to be missing from the key name, if it is we insert it.
    def _set(self, dct, **kwargs) -> None:
        for key, value in kwargs.items():
            keys = key.split(".")
            skip = False
            if "inputs" not in keys:
                keys.insert(1, "inputs")
            current = dct

            for k in keys[:-1]:
                if k not in current:
                    logger.debug(f"Attempt to set unknown pipeline parameter {key}")
                    skip = True
                    break

                current = current[k]

            if not skip:
                if not current.get(keys[-1]):
                    logger.debug(
                        f"Attempt to set parameter CREATED parameter '{key}'",
                    )
                current[keys[-1]] = value

    # Connect the named input to the named node (output).
    # Used for dynamic switching of pipeline graphs
    @classmethod
    def reconnect_input(cls, dct, input, output):
        logger.debug(f"Request to reconnect input {input} to output {output}")

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

    # This is the callback handler for comfy async events.
    def send_sync(self, label, data, _id):
        # Get receive image outputs via this async mechanism
        if "output" in data and "images" in data["output"]:
            self.images = data["output"]["images"]
            logger.debug("Received output image(s) from comfyui")
        else:
            logger.debug(f"{label}, {data}, {_id}")

    # Execute the named pipeline and pass the pipeline the parameter provided.
    # For the horde we assume the pipeline returns an array of images.
    def _run_pipeline(self, pipeline: dict, params: dict) -> dict | None:
        if _comfy_model_manager is None:
            raise RuntimeError("hordelib.initialise() must be called before using comfy_horde.")

        # Update user settings
        _comfy_model_manager.set_user_reserved_vram(UserSettings.get_vram_to_leave_free_mb())
        _comfy_model_manager.set_batch_optimisations(UserSettings.enable_batch_optimisations.active)

        # Set the pipeline parameters
        self._set(pipeline, **params)

        # Create (or retrieve) our prompt executive
        inference = self._get_executor(pipeline)

        # This is useful for dumping the entire pipeline to the terminal when
        # developing and debugging new pipelines. A badly structured pipeline
        # file just results in a cryptic error from comfy
        pretty_pipeline = pformat(pipeline)
        if False:  # This isn't here Tazlin :)
            logger.error(pretty_pipeline)

        # The client_id parameter here is just so we receive comfy callbacks for debugging.
        # We pretend we are a web client and want async callbacks.
        stdio = OutputCollector()
        with contextlib.redirect_stdout(stdio), contextlib.redirect_stderr(stdio):
            # validate_prompt from comfy returns [bool, str, list]
            # Which gives us these nice hardcoded list indexes, which valid[2] is the output node list
            self.client_id = str(uuid.uuid4())
            valid = _comfy_validate_prompt(pipeline, True)
            inference.execute(pipeline, self.client_id, {"client_id": self.client_id}, valid[2])

        stdio.replay()

        # Check if there are any resource to clean up
        cleanup()
        if time.time() - self._gc_timer > Comfy_Horde.GC_TIME:
            self._gc_timer = time.time()
            garbage_collect()

        return self.images

    # Run a pipeline that returns an image in pixel space
    def run_image_pipeline(self, pipeline, params: dict) -> list[dict] | None:
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
                return None
        else:
            pipeline_data = pipeline

        # If no callers for a while, announce it
        if self._callers == 0 and self._exit_time:
            idle_time = time.time() - self._exit_time
            if idle_time > 1 and UserSettings.enable_idle_time_warning.active:
                logger.warning(f"No job ran for {round(idle_time, 3)} seconds")

        # We have just been entered
        with self._counter_mutex:
            self._callers += 1

        result = self._run_pipeline(pipeline_data, params)

        # We are being exited
        with self._counter_mutex:
            self._exit_time = time.time()  # type: ignore
            self._callers -= 1

        if result:
            return result  # type: ignore

        return None


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
