# This tests running hordelib standalone, as an external caller would use it.
# Call with: python -m test.run_memory_test
# You need all the deps in whatever environment you are running this.
import os
import random
import threading
import time

import psutil
from loguru import logger

import hordelib
from hordelib.consts import MODEL_CATEGORY_NAMES

hordelib.initialise(setup_logging=False)

from hordelib.horde import HordeLib
from hordelib.settings import UserSettings
from hordelib.shared_model_manager import SharedModelManager
from hordelib.utils.gpuinfo import GPUInfo

# Set this to where you want the model cache to go
os.environ["AIWORKER_TEMP_DIR"] = "d:/temp/ray"
# Disable the disk cache
UserSettings.disable_disk_cache.activate()

# Do inference with all cached models
BACKGROUND_THREAD = False


def get_ram():
    virtual_memory = psutil.virtual_memory()
    total_ram_mb = virtual_memory.total / (1024 * 1024)
    used_ram = virtual_memory.used / (1024 * 1024)
    free_ram = virtual_memory.available / (1024 * 1024)
    return (int(total_ram_mb), int(used_ram), int(free_ram))


def get_free_ram():
    _, _, free = get_ram()
    return free


def get_free_vram():
    gpu = GPUInfo()
    return gpu.get_free_vram_mb()


def report_ram():
    logger.warning(f"Free RAM {get_free_ram()} MB")
    logger.warning(f"Free VRAM {get_free_vram()} MB")


def add_model(model_name):
    logger.warning(f"Loading model {model_name}")
    SharedModelManager.manager.load(model_name)
    report_ram()
    model_count = len(SharedModelManager.manager.compvis.get_loaded_models_names())
    logger.warning(f"{model_count} models now loaded")


def get_available_models():
    return SharedModelManager.manager.get_available_models()


def do_inference(model_name, iterations=1):
    """Do some work on the GPU"""
    horde = HordeLib()
    for _ in range(iterations):
        data = {
            "sampler_name": "k_euler",
            "cfg_scale": 7.5,
            "denoising_strength": 1.0,
            "seed": 123456789,
            "height": 512,
            "width": 512,
            "karras": True,
            "tiling": False,
            "hires_fix": False,
            "clip_skip": 1,
            "control_type": None,
            "image_is_control": False,
            "return_control_map": False,
            "prompt": "an ancient llamia monster",
            "ddim_steps": 50,
            "n_iter": 1,
            "model": model_name,
        }
        pil_image = horde.basic_inference_single_image(data).image
        if not pil_image:
            logger.error("Inference is failing to generate images")
        else:
            pil_image.save(f"images/stresstest/{model_name}.webp", quality=90)


def do_background_inference():
    """Keep doing inference using random loaded models. To be run in a background thread."""
    count = 1
    random.seed()
    while True:
        models = SharedModelManager.manager.get_loaded_models_names()
        model = random.choice(models)
        logger.info(f"Doing inference iteration {count} with model {model} ({len(models)} models loaded)")
        do_inference(model, 3)
        count += 1


def main():
    HordeLib()
    GPUInfo()
    SharedModelManager.load_model_managers([MODEL_CATEGORY_NAMES.compvis])

    report_ram()

    # Reserve 50% of our ram
    UserSettings.set_ram_to_leave_free_mb("50%")
    logger.warning(f"Keep {UserSettings.get_ram_to_leave_free_mb()} MB RAM free")

    # Reserve 50% of our vram
    UserSettings.set_vram_to_leave_free_mb("50%")
    logger.warning(f"Keep {UserSettings.get_vram_to_leave_free_mb()} MB VRAM free")

    # Get to our limits by loading models
    models = get_available_models()
    model_index = 0
    logger.info(f"Found {len(models)} available models")
    while model_index < len(models):
        # First we fill ram and vram
        logger.warning("Filling available memory")
        if model_index < len(models):
            add_model(models[model_index])
            model_index += 1
        else:
            logger.error("Exceeded available models")
            break

        if (
            get_free_vram() < UserSettings.get_vram_to_leave_free_mb()
            and get_free_ram() < UserSettings.get_ram_to_leave_free_mb()
        ):
            break

    # From this point, any model loading will push us past our configured resource limits

    # Start doing background inference
    if BACKGROUND_THREAD:
        thread = threading.Thread(daemon=True, target=do_background_inference)
        thread.start()

    # Push us past our limits
    how_far = 10
    if model_index < len(models) and how_far:
        add_model(models[model_index])
        model_index += 1
        how_far -= 1
    # That would have pushed something to disk, force a memory cleanup
    # cleanup()
    report_ram()

    logger.warning("Loaded all models")

    random.seed()
    models = SharedModelManager.manager.get_loaded_models_names()
    model = random.choice(models)
    count = 1
    while True:
        # Keeping doing inference
        if BACKGROUND_THREAD:
            time.sleep(60)
            break

        model = random.choice(models)
        logger.info(f"Doing inference with model {model} ({len(models)} models loaded)")
        do_inference(model)
        count += 1
        if count > 20000:
            break


if __name__ == "__main__":
    main()
