# comfy.py
# Wrapper around comfy to allow usage by the horde worker.
import copy
import glob
import json
import os
import re
from io import BytesIO

from loguru import logger
from PIL import Image

from hordelib.config_path import get_comfyui_path

# Do not change the order of these imports
# fmt: off
import execution
from comfy.sd import load_checkpoint_guess_config
from comfy.utils import load_torch_file
from comfy_extras.chainner_models import model_loading
# fmt: on


class Comfy_Horde:
    """Handles horde-specific behavior against ComfyUI."""

    # Lookup of ComfyUI standard nodes to hordelib custom nodes
    NODE_REPLACEMENTS = {
        "CheckpointLoaderSimple": "HordeCheckpointLoader",
        "UpscaleModelLoader": "HordeUpscaleModelLoader",
        "SaveImage": "HordeImageOutput",
        "LoadImage": "HordeImageLoader",
    }

    def __init__(self) -> None:
        self.client_id = None  # used for receiving comfyUI async events
        self.pipelines = {}
        self.unit_testing = os.getenv("HORDELIB_TESTING", "")

        # Load our pipelines
        self._load_pipelines()

    def _this_dir(self, filename: str, subdir="") -> str:
        target_dir = os.path.dirname(os.path.realpath(__file__))
        if subdir:
            target_dir = os.path.join(target_dir, subdir)
        return os.path.join(target_dir, filename)

    def _load_node(self, filename: str) -> None:
        try:
            execution.nodes.load_custom_node(self._this_dir(filename, subdir="nodes"))
        except Exception:
            logger.error(f"Failed to load custom pipeline node: {filename}")
            return
        logger.debug(f"Loaded custom pipeline node: {filename}")

    def _load_custom_nodes(self) -> None:
        # Load standard nodes stored in odd locations first
        self._load_extra_nodes()
        # Now load our own nodes
        files = glob.glob(self._this_dir("node_*.py", subdir="nodes"))
        for file in files:
            self._load_node(os.path.basename(file))

    def _load_comfy_node(self, filename: str) -> None:
        try:
            pathname = os.path.join(get_comfyui_path(), "comfy_extras", filename)
            execution.nodes.load_custom_node(pathname)
        except Exception:
            logger.error(f"Failed to load comfy extra node: {filename}")
            return
        logger.debug(f"Loaded comfy extra node: {filename}")

    # Load the comfy nodes that comfy stores in a different location from it's other nodes...
    def _load_extra_nodes(self) -> None:
        self._load_comfy_node("nodes_upscale_model.py")

    def _fix_pipeline_types(self, data: dict) -> dict:
        # We have a list of nodes and each node has a class type, which we may want to change
        for nodename, node in data.items():
            if ("class_type" in node) and (
                node["class_type"] in Comfy_Horde.NODE_REPLACEMENTS
            ):
                logger.debug(f"Changed type {data[nodename]['class_type']} to {Comfy_Horde.NODE_REPLACEMENTS[node['class_type']]}")
                data[nodename]["class_type"] = Comfy_Horde.NODE_REPLACEMENTS[
                    node["class_type"]
                ]
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
                    if type(input) is list:
                        if input and input[0] in renames:
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
            if "inputs" not in keys:
                keys.insert(1, "inputs")
            current = dct

            for k in keys[:-1]:
                if k not in current:
                    logger.error(f"Attempt to set unknown pipeline parameter {key}")
                    break

                current = current[k]

            current[keys[-1]] = value

    # Connect the named input to the named node (output).
    # Used for dynamic switching of pipeline graphs
    @classmethod
    def reconnect_input(cls, dct, input, output):
        keys = input.split(".")
        if "inputs" not in keys:
            keys.insert(1, "inputs")
        current = dct
        for k in keys:
            if k not in current:
                logger.error(f"Attempt to reconnect unknown input {input}")
                return

            current = current[k]

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

        logger.debug(f"Running pipeline {pipeline_name}")

        # Grab a copy of the pipeline
        pipeline = copy.copy(self.pipelines[pipeline_name])

        # Inject our model manager if required
        from hordelib.shared_model_manager import SharedModelManager
        if "model_loader.model_manager" not in params:
            logger.debug("Injecting model manager")
            params["model_loader.model_manager"] = SharedModelManager

        # If we have a source image, use that rather than noise (i.e. img2img)
        # XXX This probably shouldn't be here. But for the moment, it works.
        if "image_loader.image" in params:
            self.reconnect_input(pipeline, "sampler.latent_image", "vae_encode")

        # Set the pipeline parameters
        self._set(pipeline, **params)

        # Fake it!
        if self.unit_testing:
            img = Image.new("RGB", (64, 64), (0, 0, 0))
            byte_stream = BytesIO()
            img.save(byte_stream, format="PNG", compress_level=4)
            byte_stream.seek(0)

            return {
                "output_image": {"images": [{"imagedata": byte_stream, "type": "PNG"}]}
            }

        # Run it!
        inference = execution.PromptExecutor(self)
        # Load our custom nodes
        self._load_custom_nodes()
        # The client_id parameter here is just so we receive comfy callbacks for debugging.
        # We essential pretend we are a web client and want async callbacks.
        inference.execute(pipeline, extra_data={"client_id": 1})

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
        result = self.run_pipeline(pipeline_name, params)

        if result:
            return result["output_image"]["images"]

        return None
