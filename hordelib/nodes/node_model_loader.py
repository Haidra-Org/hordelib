# node_model_loader.py
# Simple proof of concept custom node to load models.


import comfy.model_management
import comfy.sd
import folder_paths  # type: ignore

from hordelib import SharedModelManager


class HordeCheckpointLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "will_load_loras": ("<bool>",),
                "horde_model_name": ("<horde model name>",),
                "ckpt_name": ("<ckpt name>",),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load_checkpoint"

    CATEGORY = "loaders"

    def load_checkpoint(
        self,
        will_load_loras: bool,
        horde_model_name: str,
        ckpt_name: str | None = None,
        output_vae=True,
        output_clip=True,
        preloading=False,
    ):
        if SharedModelManager.manager.compvis is None:
            raise ValueError("CompVisModelManager is not initialised.")

        same_loaded_model = SharedModelManager.manager._models_in_ram.get(horde_model_name)

        # Check if the model was previously loaded and if so, not loaded with Loras
        if same_loaded_model and not same_loaded_model[1]:
            return same_loaded_model[0]

        if not ckpt_name:
            if not SharedModelManager.manager.compvis.is_model_available(horde_model_name):
                raise ValueError(f"Model {horde_model_name} is not available.")

            ckpt_name = SharedModelManager.manager.compvis.get_model_filename(horde_model_name).name

        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        result = comfy.sd.load_checkpoint_guess_config(
            ckpt_path,
            output_vae=output_vae,
            output_clip=output_clip,
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
        )

        SharedModelManager.manager._models_in_ram[horde_model_name] = result, will_load_loras

        return result


NODE_CLASS_MAPPINGS = {"HordeCheckpointLoader": HordeCheckpointLoader}
