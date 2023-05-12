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

hordelib.initialise(setup_logging=True, clear_logs=True)

from hordelib.comfy_horde import cleanup
from hordelib.horde import HordeLib
from hordelib.settings import UserSettings
from hordelib.shared_model_manager import SharedModelManager
from hordelib.utils.gpuinfo import GPUInfo

# Set this to where you want the model cache to go
os.environ["AIWORKER_TEMP_DIR"] = "d:/temp/ray"
# Disable the disk cache
# UserSettings.disable_disk_cache.activate()

# Do inference with all cached models
VALIDATE_ALL_CACHED_MODELS = False


def get_ram():
    virtual_memory = psutil.virtual_memory()
    total_ram_mb = virtual_memory.total / (1024 * 1024)
    used_ram = virtual_memory.used / (1024 * 1024)
    free_ram = total_ram_mb - used_ram
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
    models = SharedModelManager.manager.get_available_models()
    return models


def do_inference(model_name, iterations=1):
    """Do some work on the GPU"""
    horde = HordeLib()
    for i in range(iterations):
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
            "ddim_steps": 15,
            "n_iter": 1,
            "model": model_name,
        }
        pil_image = horde.basic_inference(data)
        if not pil_image:
            logger.error("Inference is failing to generate images")
        else:
            pil_image.save(f"images/stresstest/{model_name}.webp", quality=90)


def do_background_inference(seed):
    """Keep doing inference using random loaded models. To be run in a background thread."""
    random.seed(seed)
    logger.warning(f"Thread random seed is {seed}")
    count = 1
    while True:
        models = SharedModelManager.manager.get_loaded_models_names()
        model = random.choice(models)
        # model = models[seed % 20]
        logger.info(f"Doing inference iteration {count} with model {model} ({len(models)} models loaded)")
        do_inference(model, random.randint(10, 15))
        count += 1
        if count > 10:
            return


def main():
    HordeLib()
    GPUInfo()
    SharedModelManager.loadModelManagers(compvis=True)

    report_ram()

    # Reserve 50% of our ram
    UserSettings.set_ram_to_leave_free_mb("50%")
    logger.warning(f"Keep {UserSettings.get_ram_to_leave_free_mb()} MB RAM free")

    # Reserve 50% of our vram
    UserSettings.set_vram_to_leave_free_mb("50%")
    logger.warning(f"Keep {UserSettings.get_vram_to_leave_free_mb()} MB VRAM free")

    # Start doing background inference
    start_time = time.time()
    random.seed(123456789)
    thread1 = threading.Thread(daemon=True, target=do_background_inference, args=[random.randint(0, 100000000)])
    thread2 = threading.Thread(daemon=True, target=do_background_inference, args=[random.randint(0, 100000000)])
    thread3 = threading.Thread(daemon=True, target=do_background_inference, args=[random.randint(0, 100000000)])
    thread1.start()
    thread2.start()
    thread3.start()

    while thread1.is_alive() or thread2.is_alive() or thread3.is_alive():
        # Keeping doing inference
        time.sleep(2)
        report_ram()

    logger.warning(f"Test took {round((time.time()-start_time),2)} seconds")


if __name__ == "__main__":
    main()

# 1158 seconds no mutexes, 3 threads
# 1110 seconds no mutexes, 3 threads, serialised cache read/write
# 1030 seconds no mutexes, 3 threads, serialised cache read/write, same three models
# 1044 seconds no mutexes, 3 threads, serialised cache read/write, same three models, 10 iterations, vae mutex separate from sampler mutex
#  986 seconds no mutexes, 3 threads, serialised cache read/write, same three models, 10 iterations, no property or vae mutex
# 1114 seconds no mutexes, 3 threads, serialised cache read/write, 10 iterations, no property or vae mutex
