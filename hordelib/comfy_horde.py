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
import random
import threading
from pprint import pformat

import torch
from loguru import logger

from hordelib.settings import UserSettings
from hordelib.utils.ioredirect import OutputCollector
from hordelib.config_path import get_hordelib_path

# Note these imports are intentionally somewhat obfuscated as a reminder to other modules
# that they should never call through this module into comfy directly. All calls into
# comfy should be abstracted through functions in this module.
# isort: off
from execution import nodes as _comfy_nodes
from execution import PromptExecutor as _comfy_PromptExecutor
from folder_paths import folder_names_and_paths as _comfy_folder_paths
from comfy.sd import load_checkpoint_guess_config as __comfy_load_checkpoint_guess_config
from comfy.sd import load_controlnet as __comfy_load_controlnet
from comfy.model_management import model_manager as _comfy_model_manager
from comfy.model_management import get_torch_device as __comfy_get_torch_device
from comfy.utils import load_torch_file as __comfy_load_torch_file
from comfy_extras.chainner_models import model_loading as _comfy_model_loading

# isort: on

__models_to_release = {}


def cleanup():
    with _comfy_model_manager.sampler_mutex:
        # Do we have any models waiting to be released?
        if not __models_to_release:
            return

        # Can we release any of them?
        for model_name, model_data in __models_to_release.copy().items():
            if is_model_in_use(model_data["model"]):
                # We're in the middle of using it, nothing we can do
                continue
            # Unload the model from the GPU
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
            del __models_to_release[model_name]
            gc.collect()


def remove_model_from_memory(model_name, model_data):
    with _comfy_model_manager.sampler_mutex:
        if model_name not in __models_to_release:
            __models_to_release[model_name] = model_data


def get_models_on_gpu():
    return _comfy_model_manager.get_models_on_gpu()


def get_torch_device():
    return __comfy_get_torch_device()


def unload_model_from_gpu(model):
    _comfy_model_manager.unload_model(model)
    gc.collect()
    if not torch.cuda.is_available():
        return None
    if torch.version.cuda:
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def is_model_in_use(model):
    return _comfy_model_manager.is_model_in_use(model)


def load_torch_file(filename):
    result = __comfy_load_torch_file(filename)
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
            (modelPatcher, clipModel, vae, clipVisionModel) = __comfy_load_checkpoint_guess_config(
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
        controlnet = __comfy_load_controlnet(ckpt_path=controlnet_path, model=target_model)
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
    }

    # We may wish some ComfyUI standard nodes had different names for consistency. Here
    # we can dynamically rename some node input parameter names.
    NODE_PARAMETER_REPLACEMENTS = {
        "HordeCheckpointLoader": {
            # We name this "model_name" as then we can use the same generic code in our model loaders
            "ckpt_name": "model_name",
        },
    }

    # Enforce min/max bounds on some parameters
    PARAMETER_BOUNDS = {
        "sampler.cfg": (1, 100),
        "sampler.denoise": (0.1, 1.0),
        "empty_latent_image.height": (64, 8192),
        "empty_latent_image.width": (64, 8192),
        "sampler.steps": (1, 500),
        "empty_latent_image.batch_size": (1, 500),
        "clip_skip.stop_at_clip_layer": (-10, -1),
        "latent_upscale.width": (64, 8192),
        "latent_upscale.height": (64, 8192),
    }

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

    def __init__(self) -> None:
        self._client_id = {}
        self.pipelines = {}
        self.exit_time = 0
        # Set custom node path
        _comfy_folder_paths["custom_nodes"] = ([os.path.join(get_hordelib_path(), "nodes")], [])
        # Load our pipelines
        self._load_pipelines()

        stdio = OutputCollector()
        with contextlib.redirect_stdout(stdio):
            # Load our custom nodes
            self._load_custom_nodes()
        stdio.replay()

    def _this_dir(self, filename: str, subdir="") -> str:
        target_dir = os.path.dirname(os.path.realpath(__file__))
        if subdir:
            target_dir = os.path.join(target_dir, subdir)
        return os.path.join(target_dir, filename)

    def _load_custom_nodes(self) -> None:
        _comfy_nodes.init_custom_nodes()

    def _get_executor(self, pipeline):
        executor = _comfy_PromptExecutor(self)
        return executor

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
                logger.debug(f"Renamed node input {nodename}.{oldname} to {newname}")

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

    def _assert_parameter_bounds(self, params):
        for key, value in params.items():
            if key in self.PARAMETER_BOUNDS:
                pmin, pmax = self.PARAMETER_BOUNDS[key]
                if value < pmin:
                    value = pmin
                elif value > pmax:
                    value = pmax
                params[key] = value

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

    # This is the callback handler for comfy async events. We only use it for debugging.
    def send_sync(self, p1, p2, p3):
        logger.debug(f"{p1}, {p2}, {p3}")

    # Execute the named pipeline and pass the pipeline the parameter provided.
    # For the horde we assume the pipeline returns an array of images.
    def run_pipeline(self, pipeline_name: str, params: dict) -> dict | None:
        # Sanity
        if pipeline_name not in self.pipelines:
            logger.error(f"Unknown inference pipeline: {pipeline_name}")
            return None

        # Update user settings
        _comfy_model_manager.set_user_reserved_vram(UserSettings.vram_to_leave_free_mb)

        logger.info(f"Running pipeline {pipeline_name}")

        # Grab a copy of the pipeline
        pipeline = copy.deepcopy(self.pipelines[pipeline_name])

        # Inject our model manager if required
        from hordelib.shared_model_manager import SharedModelManager

        if "model_loader.model_manager" not in params:
            logger.debug("Injecting model manager")
            params["model_loader.model_manager"] = SharedModelManager

        # If we have a source image, use that rather than noise (i.e. img2img)
        # XXX This probably shouldn't be here. But for the moment, it works.
        if params.get("image_loader.image"):
            self.reconnect_input(pipeline, "sampler.latent_image", "vae_encode")

        # XXX This shouldn't be here either, but it's not clear to me yet where the
        # XXX correct place for dynamic connection of nodes is. Need to do a few more
        # XXX pipelines to see.
        if params.get("control_type"):
            # Inject control net model manager
            if "controlnet_model_loader.model_manager" not in params:
                logger.debug("Injecting controlnet model manager")
                params["controlnet_model_loader.model_manager"] = SharedModelManager
            # Connect to the correct pre-processor node
            if params.get("return_control_map", False):
                # Connect annotator to output image directly
                self.reconnect_input(
                    pipeline,
                    "output_image.images",
                    params["control_type"],
                )
            else:
                # Connect annotator to controlnet apply node
                self.reconnect_input(
                    pipeline,
                    "controlnet_apply.image",
                    params["control_type"],
                )

        # Enforce our parameter bounds
        self._assert_parameter_bounds(params)

        # Set the pipeline parameters
        self._set(pipeline, **params)

        # Create (or retrieve) our prompt executive
        inference = self._get_executor(pipeline_name)

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
            inference.execute(pipeline, extra_data={"client_id": random.randint(0, sys.maxsize)})

        stdio.replay()

        # Check if there are any resource to clean up
        cleanup()

        return inference.outputs

    # Run a pipeline that returns an image in pixel space
    def run_image_pipeline(self, pipeline_name: str, params: dict) -> list[dict] | None:
        # From the horde point of view, let us assume the output we are interested in
        # is always in a HordeImageOutput node named "output_image". This is an array of
        # dicts of the form:
        # [ {
        #     "imagedata": <BytesIO>,
        #     "type": "PNG"
        #   },
        # ]
        # See node_image_output.py
        if self.exit_time:
            idle_time = time.time() - self.exit_time
            if idle_time > 1:
                logger.warning(f"No job ran for {round(idle_time, 3)} seconds")

        result = self.run_pipeline(pipeline_name, params)
        self.exit_time = time.time()

        if result:
            return result["output_image"]["images"]

        return None
