# node_image_output.py
# Simple proof of concept to return the similarities of text to the image to the worker
import json

from PIL import Image
from PIL.PngImagePlugin import PngInfo
from hordelib import horde_model_manager
from hordelib.clip.interrogate import Interrogator


class HordeClipSimilarities:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip_name": ("STRING",),
                "images": ("IMAGE",),
                "string_list": ("STRINGS",),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "get_similarities"

    OUTPUT_NODE = True

    CATEGORY = "image"

    def get_similarities(self, clip_name, images, string_list=None):

        results = []
        for image in images:
            model = horde_model_manager.clip.loaded_models[clip_name]
            interrogator = Interrogator(model)
            similarity_results = interrogator(
                image=image, text_array=string_list, similarity=True
            )["default"]
            results.append({"similarity_results": similarity_results, "type": "ARRAY"})

        return {"similarities": results}


NODE_CLASS_MAPPINGS = {"HordeClipSimilarities": HordeClipSimilarities}
