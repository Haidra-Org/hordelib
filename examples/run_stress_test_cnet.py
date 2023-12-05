# This tests running hordelib standalone, as an external caller would use it.
# Call with: python -m test.run_stress_test_cnet
# You need all the deps in whatever environment you are running this.
import os
import random
import sys
import threading
import time

from loguru import logger
from PIL import Image

from hordelib.consts import MODEL_CATEGORY_NAMES

if __name__ != "__main__":
    exit(0)

import hordelib

hordelib.initialise(setup_logging=False)
from hordelib.horde import HordeLib
from hordelib.shared_model_manager import SharedModelManager

random.seed(999)

if len(sys.argv) > 2:
    print(f"Usage: {sys.argv[0]} [<iterations>]")
    sys.exit(1)
if len(sys.argv) == 2:
    try:
        ITERATIONS = int(sys.argv[1])
    except ValueError:
        print("Please provide an integer as the argument.")
        sys.exit(1)
else:
    ITERATIONS = 50

out_dir = f"images/stresstest/{os.path.splitext(os.path.basename(sys.argv[0]))[0]}"
os.makedirs(out_dir, exist_ok=True)

generate = HordeLib()
SharedModelManager.load_model_managers([MODEL_CATEGORY_NAMES.compvis, MODEL_CATEGORY_NAMES.controlnet])

models = ["Deliberate", "Anything Diffusion", "Realistic Vision", "URPM"]
cnets = ["canny", "hed", "fakescribbles", "depth", "normal", "hough"]
SAMPLERS_MAP = {
    "k_euler": "euler",
    "k_euler_a": "euler_ancestral",
    "k_heun": "heun",
    "k_dpm_2": "dpm_2",
    "k_dpm_2_a": "dpm_2_ancestral",
    "k_lms": "lms",
    "k_dpm_fast": "dpm_fast",
    "k_dpm_adaptive": "dpm_adaptive",
    "k_dpmpp_2s_a": "dpmpp_2s_ancestral",
    "k_dpmpp_sde": "dpmpp_sde",
    "k_dpmpp_2m": "dpmpp_2m",
    "ddim": "ddim",
    "uni_pc": "uni_pc",
    "uni_pc_bh2": "uni_pc_bh2",
    "plms": "euler",
}

for model in models:
    SharedModelManager.manager.load(model)

start_time = time.time()

mutex = threading.Lock()
count = 0


def inc():
    with mutex:
        global count
        count += 1
        return count


def generate_images_cnet():
    i = inc()
    logger.info(f"Thread {threading.current_thread().ident} starting iteration {i}")
    cnet_type = random.choice(cnets)
    model = random.choice(models)
    sampler = random.choice(list(SAMPLERS_MAP.keys()))
    data = {
        "sampler_name": sampler,
        "cfg_scale": 7.5,
        "denoising_strength": 1.0,
        "seed": 123456789,  # random.randint(1, 1000000000),
        "height": 512,
        "width": 512,
        "karras": True,
        "tiling": False,
        "hires_fix": False,
        "clip_skip": 1,
        "control_type": cnet_type,
        "image_is_control": False,
        "return_control_map": False,
        "prompt": "a man walking in the snow",
        "ddim_steps": 25,
        "n_iter": 1,
        "model": model,
        "source_image": Image.open("images/test_db0.jpg"),
        "source_processing": "img2img",
    }
    horde = HordeLib()
    pil_image = horde.basic_inference_single_image(data).image
    pil_image.save(
        f"{out_dir}/cnet_{model}_{sampler}_{cnet_type}_{threading.current_thread().ident}_{i}.webp",
        quality=80,
    )


def run_iterations():
    for _ in range(ITERATIONS):
        generate_images_cnet()


def main():
    global count
    count = 0
    threads = [
        threading.Thread(daemon=True, target=run_iterations),
        threading.Thread(daemon=True, target=run_iterations),
        threading.Thread(daemon=True, target=run_iterations),
        threading.Thread(daemon=True, target=run_iterations),
        threading.Thread(daemon=True, target=run_iterations),
    ]
    [x.start() for x in threads]
    [x.join() for x in threads if x]

    logger.warning(f"Test took {round(time.time() - start_time)} seconds ({count} generations)")


main()
