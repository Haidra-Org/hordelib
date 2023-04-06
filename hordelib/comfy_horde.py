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

# Do not change the order of these imports
# fmt: off
import execution
from comfy.sd import load_checkpoint_guess_config
# fmt: on

class Comfy_Horde:
    """Handles horde-specific behavior against ComfyUI."""

    # Lookup of ComfyUI standard nodes to hordelib custom nodes
    NODE_REPLACEMENTS = {
        "CheckpointLoaderSimple": "HordeCheckpointLoader",
        "SaveImage": "HordeImageOutput",
    }

    def __init__(self) -> None:
        self.pipelines = {}
        self.unit_testing = os.getenv("HORDELIB_TESTING", "")

        # XXX Temporary hack for model dir
        model_dir = os.getenv("AIWORKER_CACHE_HOME", "")
        if not model_dir:
            os.environ["HORDE_MODEL_DIR_CHECKPOINTS"] = self._this_dir("../")
        else:
            os.environ["HORDE_MODEL_DIR_CHECKPOINTS"] = os.path.join(
                model_dir, "nataili", "compvis"
            )

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
        files = glob.glob(self._this_dir("node_*.py", subdir="nodes"))
        for file in files:
            self._load_node(os.path.basename(file))

    def _fix_pipeline_types(self, data: dict) -> dict:
        # We have a list of nodes and each node has a class type, which we may want to change
        for nodename, node in data.items():
            if ("class_type" in node) and (
                node["class_type"] in Comfy_Horde.NODE_REPLACEMENTS
            ):
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

    # Execute the named pipeline and pass the pipeline the parameter provided.
    # For the horde we assume the pipeline returns an array of images.
    def run_pipeline(self, pipeline_name: str, params: dict) -> dict | None:
        # Sanity
        if pipeline_name not in self.pipelines:
            logger.error(f"Unknown inference pipeline: {pipeline_name}")
            return None

        # Grab a copy of the pipeline
        pipeline = copy.copy(self.pipelines[pipeline_name])

        # Inject our model manager if required
        from hordelib.shared_model_manager import SharedModelManager
        if "model_loader.model_manager" not in params:
            params["model_loader.model_manager"] = SharedModelManager

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
        inference.execute(pipeline)

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
