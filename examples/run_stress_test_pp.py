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

if __name__ != "__main__":
    exit(0)

import hordelib

hordelib.initialise(setup_logging=False)
from hordelib.consts import MODEL_CATEGORY_NAMES
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
SharedModelManager.load_model_managers(
    [
        MODEL_CATEGORY_NAMES.compvis,
        MODEL_CATEGORY_NAMES.esrgan,
        MODEL_CATEGORY_NAMES.gfpgan,
    ],
)
models = [
    "CodeFormers",
    "GFPGAN",
]
models_upscale = [
    "RealESRGAN_x4plus",
    "RealESRGAN_x2plus",
    "NMKD_Siax",
    "RealESRGAN_x4plus_anime_6B",
    "4x_AnimeSharp",
]
for model in models:
    SharedModelManager.manager.load(model)
for model in models_upscale:
    SharedModelManager.manager.load(model)

start_time = time.time()

mutex = threading.Lock()
count = 0


def inc():
    with mutex:
        global count
        count += 1
        return count


def generate_images_pp():
    i = inc()
    logger.info(f"Thread {threading.current_thread().ident} starting iteration {i}")
    model = random.choice(models)
    data = {
        "model": model,
        "source_image": Image.open("images/test_facefix.png"),
    }
    pil_image = generate.image_facefix(data).image
    pil_image.save(
        f"{out_dir}/pp_{model}_{threading.current_thread().ident}_{i}.webp",
        quality=80,
    )


def generate_images_pp_upscale():
    i = inc()
    logger.info(f"Thread {threading.current_thread().ident} starting iteration {i}")
    model = random.choice(models_upscale)
    data = {
        "model": model,
        "source_image": Image.open("images/test_db0.jpg"),
    }
    pil_image = generate.image_upscale(data).image
    pil_image.save(
        f"{out_dir}/pp_{model}_{threading.current_thread().ident}_{i}.webp",
        quality=80,
    )


def run_iterations():
    for _ in range(ITERATIONS):
        if random.random() < 0.5:
            generate_images_pp()
        else:
            generate_images_pp_upscale()


def main():
    global count
    count = 0
    threads = [
        threading.Thread(daemon=True, target=run_iterations),
        threading.Thread(daemon=True, target=run_iterations),
        threading.Thread(daemon=True, target=run_iterations),
    ]
    [x.start() for x in threads]
    [x.join() for x in threads if x]

    logger.warning(f"Test took {round(time.time() - start_time)} seconds ({count} generations)")


main()
