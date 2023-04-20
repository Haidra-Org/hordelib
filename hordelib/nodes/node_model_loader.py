# node_model_loader.py
# Simple proof of concept custom node to load models.

import contextlib
import os
import pickle

from loguru import logger


class HordeCheckpointLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_manager": ("<model manager instance>",),
                "model_name": ("<model name>",),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load_checkpoint"

    CATEGORY = "loaders"

    def load_checkpoint(
        self,
        model_manager,
        model_name,
    ):
        logger.debug(f"Loading model {model_name} through our custom node")

        if model_manager.manager is None:
            logger.error("horde_model_manager appears to be missing!")
            raise RuntimeError  # XXX better guarantees need to be made

        loaded_models = model_manager.manager.loaded_models
        if model_name not in loaded_models:
            logger.error(f"Model {model_name} is not loaded")
            raise RuntimeError  # XXX better guarantees need to be made

        model = loaded_models[model_name]["model"]
        clip = loaded_models[model_name]["clip"]
        vae = loaded_models[model_name]["vae"]

        # If we got strings, not objects, it's a cache reference, load the cache
        if type(model) is str:
            logger.info("Loading model data from disk cache")
            model_cache = model
            try:
                with open(model, "rb") as cache:
                    model = pickle.load(cache)
                    vae = pickle.load(cache)
                    clip = pickle.load(cache)
            except (pickle.PickleError, EOFError):
                # Most likely corrupt cache file, remove the file
                with contextlib.suppress(OSError):
                    os.remove(model)  # ... at least try to remove it

                raise Exception(f"Model cache file {model_cache} was corrupt. It has been removed.")

        # XXX # TODO I would like to revisit this dict->tuple conversion at some point soon
        return (model, clip, vae)


NODE_CLASS_MAPPINGS = {"HordeCheckpointLoader": HordeCheckpointLoader}
