# node_image_output.py
# Simple proof of concept to return an image byte stream to the horde worker.
import json

import logfire
import numpy as np
from loguru import logger
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from hordelib.execution.results import UI_ENTRY_IMAGE_KEY, UI_ENTRY_TYPE_KEY, UNENCODED_IMAGE_TYPE, encode_image_png


class HordeImageOutput:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
            },
            "optional": {
                # False returns each image as a PIL image so a caller that re-encodes or keeps
                # working on the pixels skips the PNG encode and decode.
                "encode_png": ("BOOLEAN", {"default": True}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ()
    FUNCTION = "get_image"

    OUTPUT_NODE = True

    CATEGORY = "image"

    def _json_hack(self, obj):
        if hasattr(obj, "__class__"):
            return f"{obj.__class__.__name__} instance"
        return f"Object of type {type(obj).__name__}"

    @logfire.instrument("image.output_node")
    def get_image(self, images, encode_png=True, prompt=None, extra_pnginfo=None):
        logger.info("image.generating_output: image_count={}", len(images))
        results = []
        for idx, image in enumerate(images):
            i = 255.0 * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            if not encode_png:
                results.append({UI_ENTRY_IMAGE_KEY: img, UI_ENTRY_TYPE_KEY: UNENCODED_IMAGE_TYPE})
                continue
            with logfire.span("image.encode_png", image_index=idx):
                metadata = PngInfo()
                # Save the full pipeline and variables into the PNG metadata
                if prompt is not None:
                    metadata.add_text("prompt", json.dumps(prompt, default=self._json_hack))
                if extra_pnginfo is not None:
                    for x in extra_pnginfo:
                        metadata.add_text(x, json.dumps(extra_pnginfo[x], default=self._json_hack))

                results.append({"imagedata": encode_image_png(img, metadata), "type": "PNG"})

        logger.info("image.output_complete: result_count={}", len(results))
        return {"ui": {"images": results}}


NODE_CLASS_MAPPINGS = {"HordeImageOutput": HordeImageOutput}
