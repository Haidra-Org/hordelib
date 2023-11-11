# Produce a showcase of all models. Assumes you have them all cached already locally.
import os

from loguru import logger

import hordelib

hordelib.initialise(setup_logging=False)

from hordelib.consts import MODEL_CATEGORY_NAMES
from hordelib.horde import HordeLib
from hordelib.shared_model_manager import SharedModelManager

# Set this to where your hordelib tmp model cache is
os.environ["AIWORKER_TEMP_DIR"] = "d:/temp/ray"

# Do inference with all cached models
VALIDATE_ALL_CACHED_MODELS = True


def add_model(model_name):
    logger.warning(f"Loading model {model_name}")
    SharedModelManager.manager.load(model_name)
    model_count = len(SharedModelManager.manager.compvis.get_loaded_models_names())
    logger.warning(f"{model_count} models now loaded")


def do_inference(model_name, iterations=1):
    """Do some work on the GPU"""
    horde = HordeLib()
    for _ in range(iterations):
        data = {
            "sampler_name": "k_euler",
            "cfg_scale": 6.5,
            "denoising_strength": 1.0,
            "seed": 3688490319,
            "height": 512,
            "width": 512,
            "karras": True,
            "tiling": False,
            "hires_fix": False,
            "clip_skip": 1,
            "control_type": None,
            "image_is_control": False,
            "return_control_map": False,
            "prompt": (
                "a woman closeup made out of metal, (cyborg:1.1), realistic skin, (detailed wire:1.3), "
                "(intricate details), hdr, (intricate details, hyperdetailed:1.2), cinematic shot, "
                "vignette, centered"
            ),
            "ddim_steps": 30,
            "n_iter": 1,
            "model": model_name,
        }
        pil_image = horde.basic_inference_single_image(data).image
        if not pil_image:
            logger.error("Inference is failing to generate images")
        else:
            pil_image.save(f"images/all_models/{model_name}.webp", quality=90)


def main():
    HordeLib()
    SharedModelManager.load_model_managers([MODEL_CATEGORY_NAMES.compvis])

    os.makedirs("images/all_models/", exist_ok=True)

    # We may have just fast-loaded a bunch of cached models, do some inference with each of them
    if VALIDATE_ALL_CACHED_MODELS:
        logger.warning("Validating cached model files")
        for model in SharedModelManager.manager.get_loaded_models_names():
            do_inference(model)
        logger.warning("Model cache files validation completed.")
        exit(0)


if __name__ == "__main__":
    main()
