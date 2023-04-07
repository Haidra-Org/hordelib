import numpy as np
from transformers import CLIPFeatureExtractor

from hordelib.shared_model_manager import SharedModelManager


def is_image_nsfw(image):
    if "safety_checker" not in SharedModelManager.manager.loaded_models:
        SharedModelManager.manager.load("safety_checker")
    safety_checker = SharedModelManager.manager.loaded_models["safety_checker"]["model"]
    feature_extractor = CLIPFeatureExtractor()
    image_features = feature_extractor(image, return_tensors="pt").to("cpu")
    _, has_nsfw_concept = safety_checker(
        clip_input=image_features.pixel_values,
        images=[np.asarray(image)],
    )
    return has_nsfw_concept is not None and True in has_nsfw_concept
