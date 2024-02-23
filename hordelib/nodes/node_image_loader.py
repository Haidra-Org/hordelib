# horde_image_loader.py
# Load images into the pipeline from PIL, not disk
import numpy as np
import PIL.Image
import torch
from loguru import logger


class HordeImageLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"image": ("<PIL Instance>",)},
        }

    CATEGORY = "image"

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "load_image"

    def load_image(self, image):
        if image is None:
            logger.error("Input image is None in HordeImageLoader - this is a bug, please report it!")
            raise ValueError("Input image is None in HordeImageLoader")

        if not isinstance(image, PIL.Image.Image):
            logger.error(f"Input image is not a PIL Image, it is a {type(image)}")
            raise ValueError(f"Input image is not a PIL Image, it is a {type(image)}")

        new_image = image.convert("RGB")
        new_image = np.array(new_image).astype(np.float32) / 255.0
        new_image = torch.from_numpy(new_image)[None,]
        if "A" in image.getbands():
            mask = np.array(image.getchannel("A")).astype(np.float32) / 255.0
            mask = 1.0 - torch.from_numpy(mask)
        else:
            mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
        return (new_image, mask)


NODE_CLASS_MAPPINGS = {"HordeImageLoader": HordeImageLoader}
