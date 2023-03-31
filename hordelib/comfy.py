# comfy.py
# Wrapper around comfy to allow usage by the horde worker.
import copy
import glob
import json
import os
import re

from loguru import logger

from hordelib.ComfyUI import execution


class Comfy:
    def __init__(self):
        self.pipelines = {}

        # FIXME Temporary hack for model dir
        os.environ["HORDE_MODEL_DIR_CHECKPOINTS"] = self._this_dir("../")

        # Load our pipelines
        self._load_pipelines()

    def _this_dir(self, filename):
        return os.path.join(os.path.dirname(os.path.realpath(__file__)), filename)

    def _load_node(self, filename):
        try:
            execution.nodes.load_custom_node(self._this_dir(filename))
        except Exception:
            logger.error(f"Failed to load custom pipeline node: {filename}")
            return
        logger.debug(f"Loaded custom pipeline node: {filename}")

    def _load_custom_nodes(self):
        files = glob.glob(self._this_dir("node_*.py"))
        for file in files:
            self._load_node(os.path.basename(file))

    def _load_pipeline(self, filename):
        if not os.path.exists(filename):
            logger.error(f"No such inference pipeline file: {filename}")
            return

        try:
            with open(filename) as jsonfile:
                pipeline_name = re.match(r".*pipeline_(.*)\.json", filename)[1]
                data = json.loads(jsonfile.read())
                self.pipelines[pipeline_name] = data
                logger.debug(f"Loaded inference pipeline: {pipeline_name}")
                return True
        except (OSError, ValueError):
            logger.error(f"Invalid inference pipeline file: {filename}")

    def _load_pipelines(self):
        files = glob.glob(self._this_dir("pipeline_*.json"))
        loaded_count = 0
        for file in files:
            if self._load_pipeline(file):
                loaded_count += 1
        return loaded_count

    # Inject parameters into a pre-configured pipeline
    # We allow "inputs" to be missing from the key name, if it is we insert it.
    def _set(self, dct, **kwargs):
        for key, value in kwargs.items():
            keys = key.split(".")
            if "inputs" not in keys:
                keys.insert(1, "inputs")
            current = dct

            for k in keys[:-1]:
                if k not in current:
                    logger.error(f"Attempt to set unknown pipeline parameter {key}")
                    break
                else:
                    current = current[k]

            current[keys[-1]] = value

    # Execute the named pipeline and pass the pipeline the parameter provided.
    # For the horde we assume the pipeline returns an array of images.
    def run_pipeline(self, pipeline_name, params):

        # Sanity
        if pipeline_name not in self.pipelines:
            logger.error(f"Unknown inference pipeline: {pipeline_name}")
            return

        # Grab a copy of the pipeline
        pipeline = copy.copy(self.pipelines[pipeline_name])
        # Set the pipeline parameters
        self._set(pipeline, **params)
        # Run it!
        inference = execution.PromptExecutor(self)
        # Load our custom nodes
        self._load_custom_nodes()
        inference.execute(pipeline)

        return inference.outputs

    # Run a pipeline that returns an image in pixel space
    def run_image_pipeline(self, pipeline_name, params):
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
        return result["output_image"]["images"]
