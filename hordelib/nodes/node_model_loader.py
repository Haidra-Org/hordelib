# node_model_loader.py
# Simple proof of concept custom node to load models.

from types import FunctionType
import comfy.model_management
import comfy.sd
import folder_paths  # type: ignore

from hordelib.shared_model_manager import SharedModelManager


# Code from https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/b06d7b365c28729681016fdfedc42d965ac8a838/modules/sd_hijack.py#L321
# Function `add_circular_option_to_conv_2d` used under AGPL-3.0 License
# Copyright (c) 2023 AUTOMATIC1111
conv2d_constructor: FunctionType | None = None


def add_circular_option_to_conv_2d():
    import comfy.ops

    global conv2d_constructor
    conv2d_constructor = comfy.ops.Conv2d.__init__

    def conv2d_constructor_circular(self, *args, **kwargs):
        return conv2d_constructor(self, *args, padding_mode="circular", **kwargs)

    comfy.ops.Conv2d.__init__ = conv2d_constructor_circular


def remove_circular_option_from_conv_2d():
    global conv2d_constructor
    if conv2d_constructor is None:
        return

    import comfy.ops

    comfy.ops.Conv2d.__init__ = conv2d_constructor


class HordeCheckpointLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "will_load_loras": ("<bool>",),
                "seamless_tiling_enabled": ("<bool>",),
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
        seamless_tiling_enabled: bool,
        horde_model_name: str,
        ckpt_name: str | None = None,
        output_vae=True,
        output_clip=True,
        preloading=False,
    ):
        if SharedModelManager.manager.compvis is None:
            raise ValueError("CompVisModelManager is not initialised.")

        # same_loaded_model = SharedModelManager.manager._models_in_ram.get(horde_model_name)

        # if same_loaded_model and seamless_tiling_enabled:
        #     make_circular(same_loaded_model[0].model)
        # elif same_loaded_model and not seamless_tiling_enabled:
        #     make_regular(same_loaded_model[0].model)

        # Check if the model was previously loaded and if so, not loaded with Loras
        # if same_loaded_model and not same_loaded_model[1]:
        #     return same_loaded_model[0]

        if not ckpt_name:
            if not SharedModelManager.manager.compvis.is_model_available(horde_model_name):
                raise ValueError(f"Model {horde_model_name} is not available.")

            ckpt_name = SharedModelManager.manager.compvis.get_model_filename(horde_model_name).name

        if seamless_tiling_enabled:
            add_circular_option_to_conv_2d()
        else:
            remove_circular_option_from_conv_2d()

        SharedModelManager.manager._models_in_ram = {}
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        result = comfy.sd.load_checkpoint_guess_config(
            ckpt_path,
            output_vae=output_vae,
            output_clip=output_clip,
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
        )

        # SharedModelManager.manager._models_in_ram[horde_model_name] = result, will_load_loras

        return result


def make_circular(m):
    for child in m.children():
        if "Conv2d" in str(type(child)):
            child.padding_mode = "circular"
        make_circular(child)


def make_regular(m):
    for child in m.children():
        if "Conv2d" in str(type(child)):
            child.padding_mode = "zeros"
        make_regular(child)


NODE_CLASS_MAPPINGS = {"HordeCheckpointLoader": HordeCheckpointLoader}
