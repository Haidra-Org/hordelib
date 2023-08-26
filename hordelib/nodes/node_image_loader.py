# horde_image_loader.py
# Load images into the pipeline from PIL, not disk
import numpy as np
import torch


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
