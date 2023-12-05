# This tests running hordelib standalone, as an external caller would use it.
# Call with: python -m test.run_memory_test
# You need all the deps in whatever environment you are running this.
import time

from loguru import logger
from PIL import Image

import hordelib

hordelib.initialise(setup_logging=True, clear_logs=True)

from hordelib.consts import MODEL_CATEGORY_NAMES
from hordelib.horde import HordeLib
from hordelib.settings import UserSettings
from hordelib.shared_model_manager import SharedModelManager

UserSettings.disable_disk_cache.activate()


BASE_KUDOS = 10


def add_model(model_name):
    logger.warning(f"Loading model {model_name}")
    SharedModelManager.manager.load(model_name)
    model_count = len(SharedModelManager.manager.compvis.get_loaded_models_names())
    logger.warning(f"{model_count} models now loaded")


def get_base_data(width=1024, height=1024, karras=False, steps=50, model_name="SDXL 1.0"):
    return {
        "sampler_name": "k_euler",
        "cfg_scale": 7.5,
        "denoising_strength": 1.0,
        "seed": 123456789,
        "width": width,
        "height": height,
        "karras": karras,
        "tiling": False,
        "hires_fix": False,
        "clip_skip": 1,
        "control_type": None,
        "image_is_control": False,
        "return_control_map": False,
        "prompt": "an ancient llamia monster",
        "ddim_steps": steps,
        "n_iter": 1,
        "model": model_name,
    }


def do_inference(data):
    """Do some work on the GPU"""
    ITERATIONS = 1
    horde = HordeLib()
    start_time = time.time()
    for _ in range(ITERATIONS):  # do n times and average the time
        pil_image = horde.basic_inference_single_image(data).image
        if not pil_image:
            logger.error("Inference is failing to generate images")
        else:
            pil_image.save(f"images/stresstest/{data['model']}.webp", quality=90)
    return round((time.time() - start_time) / ITERATIONS, 2)


def calculate_kudos_cost(base_time, job_data) -> tuple[float, float]:
    """Calculate the kudos cost of a job, and return the kudos cost and time taken"""
    job_time = do_inference(job_data)
    return round(BASE_KUDOS * (job_time / base_time), 2), job_time


def main():
    horde = HordeLib()
    SharedModelManager.load_model_managers(
        [
            MODEL_CATEGORY_NAMES.compvis,
            MODEL_CATEGORY_NAMES.controlnet,
            MODEL_CATEGORY_NAMES.esrgan,
            MODEL_CATEGORY_NAMES.gfpgan,
            MODEL_CATEGORY_NAMES.codeformer,
        ],
    )

    add_model("stable_diffusion")

    # Do some inference to warm up
    base_time = do_inference(get_base_data())
    base_time = do_inference(get_base_data())

    # base time worth 10 kudos
    logger.info(f"Calculating time for base of {BASE_KUDOS} kudos")
    base_time = do_inference(get_base_data())

    logger.info("Calculating time for steps 10")
    base_steps_10_kudos = calculate_kudos_cost(base_time, get_base_data(steps=10))
    logger.info("Calculating time for steps 100")
    base_steps_100 = calculate_kudos_cost(base_time, get_base_data(steps=100))

    logger.info("Calculating kudos for karras")
    base_karras = calculate_kudos_cost(base_time, get_base_data(karras=True))

    logger.info("Calculating kudos for weights")
    tmpdata = get_base_data()
    tmpdata["prompt"] = "(dog:1.5) and (cat:1.5) and (mouse:1.2) and (bird:1.6) and (owl:1.9)"
    base_weights = calculate_kudos_cost(base_time, tmpdata)
    tmpdata = get_base_data()
    tmpdata["prompt"] = "dog and cat and mouse and bird and owl"
    base_no_weights = calculate_kudos_cost(base_time, tmpdata)

    logger.info("Calculating kudos for 1024x1024")
    base_1024 = calculate_kudos_cost(base_time, get_base_data(1024, 1024))

    logger.info("Calculating kudos for 2048x2048")
    base_2048 = calculate_kudos_cost(base_time, get_base_data(2048, 2048))

    logger.info("Calculating kudos for 1024x1024")
    tmpdata = get_base_data(1024, 1024)
    tmpdata["hires_fix"] = True
    base_hires_fix = calculate_kudos_cost(base_time, tmpdata)

    # Benchmark all samplers
    samplers = {}
    for sampler in horde.SAMPLERS_MAP.keys():
        logger.info(f"Calculating kudos for sampler {sampler}")
        data = get_base_data()
        data["sampler_name"] = sampler
        kudos = calculate_kudos_cost(base_time, data)
        samplers[sampler] = kudos

    # Benchmark all controlnet types
    controltypes = {}
    for controltype in horde.CONTROLNET_IMAGE_PREPROCESSOR_MAP.keys():
        break
        logger.info(f"Calculating kudos for controlnet {controltype}")
        data = get_base_data()
        data["control_type"] = controltype
        data["source_processing"] = "img2img"
        data["source_image"] = Image.open("images/test_db0.jpg")
        kudos = calculate_kudos_cost(base_time, data)
        controltypes[controltype] = kudos

    # Results
    logger.info(f"Base time {base_time} == 10 kudos")
    logger.info("Results: (kudos, time)")
    logger.info(f"10 steps: {base_steps_10_kudos}")
    logger.info(f"100 steps: {base_steps_100}")
    logger.info(f"Karras: {base_karras}")
    logger.info(f"No weights: {base_no_weights}")
    logger.info(f"Weights: {base_weights}")
    logger.info(f"Kudos 1024x1024: {base_1024}")
    logger.info(f"Hires fix 1024x1024: {base_hires_fix}")
    logger.info(f"Kudos 2048x2048: {base_2048}")

    for sampler, kudos in samplers.items():
        logger.info(f"{sampler}: {kudos}")

    for controltype, kudos in controltypes.items():
        logger.info(f"{controltype}: {kudos}")


if __name__ == "__main__":
    main()
