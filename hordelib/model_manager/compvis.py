import copy
import os
import pickle
import typing

from typing_extensions import override

from hordelib.comfy_horde import horde_load_checkpoint
from hordelib.consts import MODEL_CATEGORY_NAMES, MODEL_DB_NAMES, MODEL_FOLDER_NAMES
from hordelib.model_manager.base import BaseModelManager


class CompVisModelManager(BaseModelManager):
    def __init__(
        self,
        download_reference=False,
        # custom_path="models/custom",  # XXX Remove this and any others like it?
    ):
        super().__init__(
            modelFolder=MODEL_FOLDER_NAMES[MODEL_CATEGORY_NAMES.compvis],
            models_db_name=MODEL_DB_NAMES[MODEL_CATEGORY_NAMES.compvis],
            download_reference=download_reference,
        )

    @override
    def modelToRam(
        self,
        model_name: str,
        **kwargs,
    ) -> dict[str, typing.Any]:

        embeddings_path = os.getenv("HORDE_MODEL_DIR_EMBEDDINGS", "./")

        return horde_load_checkpoint(
            ckpt_path=self.getFullModelPath(model_name),
            embeddings_path=embeddings_path,
        )

    def can_cache_on_disk(self):
        """Can this of type model be cached on disk?"""
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
            # We have a cache file but only consider it valid if it's up to date
            model_timestamp = os.path.getmtime(model_filename)
            cache_timestamp = os.path.getmtime(cache_file)
            if model_timestamp <= cache_timestamp:
                return True
        return False

    def move_to_disk_cache(self, model_name):
        with self._mutex:
            cache_file = self.get_model_cache_filename(model_name)
            # Serialise our objects
            model_data = copy.copy(self.get_loaded_model(model_name))
            components = ["model", "vae", "clip"]
            if not self.have_model_cache(model_name):
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
