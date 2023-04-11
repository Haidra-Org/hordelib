"""Simple proof of concept to return the similarities of text to the image to the worker"""
# node_image_output.py


from hordelib.clip.interrogate import Interrogator
from hordelib.horde import SharedModelManager


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
            model_info = SharedModelManager.manager.clip.loaded_models[clip_name]
            # XXX better guarantees need to be made
            interrogator = Interrogator(model_info)
            similarity_results = interrogator(
                image=image,
                text_array=string_list,
                similarity=True,
            )
            if similarity_results is None:
                return None  # XXX better guarantees need to be made

            results.append(
                {"similarity_results": similarity_results["default"], "type": "ARRAY"},
            )

        return {"similarities": results}


NODE_CLASS_MAPPINGS = {"HordeClipSimilarities": HordeClipSimilarities}
