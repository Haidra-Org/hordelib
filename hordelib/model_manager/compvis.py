import copy
import os
import pickle
import typing

from loguru import logger
from typing_extensions import override

from hordelib import UserSettings
from hordelib.comfy_horde import horde_load_checkpoint
from hordelib.consts import MODEL_CATEGORY_NAMES
from hordelib.model_manager.base import BaseModelManager


class CompVisModelManager(BaseModelManager):
    def __init__(
        self,
        download_reference=False,
        # custom_path="models/custom",  # XXX Remove this and any others like it?
    ):
        super().__init__(
            model_category_name=MODEL_CATEGORY_NAMES.compvis,
            download_reference=download_reference,
        )

    @override
    def is_local_model(self, model_name):
        parts = os.path.splitext(model_name.lower())
        if parts[-1] in [".safetensors", ".ckpt"]:
            return True
        return False

    @override
    def modelToRam(
        self,
        model_name: str,
        **kwargs,
    ) -> dict[str, typing.Any]:
        embeddings_path = os.path.join(UserSettings.get_model_directory(), "ti")
        if not embeddings_path:
            logger.debug("No embeddings path found, disabling embeddings")

        if not kwargs.get("local", False):
            ckpt_path = self.getFullModelPath(model_name)
        else:
            ckpt_path = os.path.join(self.modelFolderPath, model_name)
        return horde_load_checkpoint(
            ckpt_path=ckpt_path,
            embeddings_path=embeddings_path if embeddings_path else None,
        )

    def can_cache_on_disk(self):
        """Can this of type model be cached on disk?"""
        if UserSettings.disable_disk_cache.active:
            return False
        return True

    def can_auto_unload(self):
        # Allow compvis models to be auto unloaded
        return True

    def can_move_to_vram(self):
        # Allow moving directly to vram to save ram
        return True

    def get_model_cache_filename(self, model_name):
        cache_dir = os.getenv("AIWORKER_TEMP_DIR", "./tmp")
        # Create cache directory if it doesn't already exist
        if not os.path.isdir(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, model_name)
        return f"{cache_file}.hordelib.cache"

    def have_model_cache(self, model_name):
        model_filename = self.getFullModelPath(model_name)
        cache_file = self.get_model_cache_filename(model_name)
        if os.path.exists(cache_file):
            if not self.validate_model(model_name):
                # The model is invalid, so delete the cache file because that's almost certainly no good either
                # This should only happen if the model was updated on disk manually, or the model reference changed
                logger.error(f"The model {model_name} is invalid, deleting the cache file.")
                os.remove(path=cache_file)
                return False
            # We have a cache file but only consider it valid if it's up to date
            model_timestamp = os.path.getmtime(model_filename)
            cache_timestamp = os.path.getmtime(cache_file)
            if model_timestamp <= cache_timestamp:
                return True
        return False

    def load_from_disk_cache(self, model_name):
        filename = self.get_model_cache_filename(model_name)
        logger.info(f"Model {model_name} warm loaded from disk cache")
        return {
            "model": filename,
            "clip": filename,
            "vae": filename,
            "clipVisionModel": None,
        }

    def move_to_disk_cache(self, model_name):
        with self._mutex:
            cache_file = self.get_model_cache_filename(model_name)
            # Serialise our objects
            model_data = copy.copy(self.get_loaded_model(model_name))
            components = ["model", "vae", "clip"]
            if not self.have_model_cache(model_name):
                # Only do one sequential write at a time
                with self._disk_write_mutex:
                    with open(cache_file, "wb") as cache:
                        for component in components:
                            pickle.dump(
                                self.get_loaded_model(model_name)[component],
                                cache,
                                protocol=pickle.HIGHEST_PROTOCOL,
                            )
            for component in components:
                model_data[component] = cache_file
            # Remove from vram/ram
            self.free_model_resources(model_name)
            # Point the model to the cache
            self.add_loaded_model(model_name, model_data)

    def move_from_disk_cache(self, model_name, model, clip, vae):
        self.ensure_ram_available()
        with self._mutex:
            self._loaded_models[model_name]["model"] = model
            self._loaded_models[model_name]["clip"] = clip
            self._loaded_models[model_name]["vae"] = vae
