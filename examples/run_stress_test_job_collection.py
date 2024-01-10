# Call with: python -m test.run_stress_test_job_collection
# You need all the deps in whatever environment you are running this.
import copy
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
from hordelib.settings import UserSettings
from hordelib.shared_model_manager import SharedModelManager

DOWNLOAD_LORAS = True

FILTER_JOBS = None  # ["txt2img", "img2img"]  # filter on "desc" attribute

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

logger.warning(f"Running for {ITERATIONS} iterations: {sys.argv}")

# Reserve 50% of our ram
UserSettings.set_ram_to_leave_free_mb("30%")
logger.warning(f"Keep {UserSettings.get_ram_to_leave_free_mb()} MB RAM free")

# Reserve 50% of our vram
UserSettings.set_vram_to_leave_free_mb("20%")
logger.warning(f"Keep {UserSettings.get_vram_to_leave_free_mb()} MB VRAM free")

out_dir = f"images/stresstest/{os.path.splitext(os.path.basename(sys.argv[0]))[0]}"
os.makedirs(out_dir, exist_ok=True)

generate = HordeLib()
SharedModelManager.load_model_managers(
    [
        MODEL_CATEGORY_NAMES.compvis,
        MODEL_CATEGORY_NAMES.controlnet,
        MODEL_CATEGORY_NAMES.codeformer,
        MODEL_CATEGORY_NAMES.esrgan,
        MODEL_CATEGORY_NAMES.gfpgan,
        MODEL_CATEGORY_NAMES.lora,
    ],
)
if DOWNLOAD_LORAS:
    SharedModelManager.manager.lora.download_default_loras()
    SharedModelManager.manager.lora.wait_for_downloads()

models = [
    "Deliberate",
    "Anything Diffusion",
    "Realistic Vision",
    "URPM",
    "Abyss OrangeMix",
    "Counterfeit",
    "ChilloutMix",
    "Epic Diffusion",
    "Babes",
    "ICBINP - I Can't Believe It's Not Photography",
]
pp_models = [
    "CodeFormers",
    "GFPGAN",
]
pp_models_upscale = [
    "RealESRGAN_x4plus",
    "RealESRGAN_x2plus",
    "NMKD_Siax",
    "RealESRGAN_x4plus_anime_6B",
    "4x_AnimeSharp",
]

for model in models:
    SharedModelManager.manager.load(model)
for model in pp_models:
    SharedModelManager.manager.load(model)
for model in pp_models_upscale:
    SharedModelManager.manager.load(model)


# We run a series of fixed jobs, across many threads. All the images in each group should be
# the same or else jobs are bleeding into one another. These jobs are carefully chosen to ensure
# model locking doesn't prevent multi-threading as much as possible
THE_JOBS = [
    {
        # Simple canny control net
        "desc": "canny",
        "sampler_name": "k_euler",
        "cfg_scale": 7.5,
        "denoising_strength": 1.0,
        "seed": 1234897333,
        "height": 512,
        "width": 512,
        "karras": True,
        "tiling": False,
        "hires_fix": False,
        "clip_skip": 1,
        "control_type": "canny",
        "image_is_control": False,
        "return_control_map": False,
        "prompt": "a man walking in heaven",
        "ddim_steps": 25,
        "n_iter": 1,
        "model": "Deliberate",
        "source_image": Image.open("images/test_db0.jpg"),
        "source_processing": "img2img",
    },
    {
        # Simple txt2img
        "desc": "txt2img",
        "sampler_name": "k_euler",
        "cfg_scale": 7.5,
        "denoising_strength": 1.0,
        "seed": 346666,
        "height": 512,
        "width": 512,
        "karras": True,
        "tiling": False,
        "hires_fix": False,
        "clip_skip": 1,
        "control_type": None,
        "image_is_control": False,
        "return_control_map": False,
        "prompt": "a dog",
        "ddim_steps": 25,
        "n_iter": 1,
        "model": "Realistic Vision",
        "source_image": None,
        "source_processing": "txt2img",
    },
    {
        # Simple img2img
        "desc": "img2img",
        "sampler_name": "k_euler",
        "cfg_scale": 7.5,
        "denoising_strength": 0.4,
        "seed": 250636385744582,
        "height": 512,
        "width": 512,
        "karras": True,
        "tiling": False,
        "hires_fix": False,
        "clip_skip": 1,
        "control_type": None,
        "image_is_control": False,
        "return_control_map": False,
        "prompt": "a dinosaur",
        "ddim_steps": 25,
        "n_iter": 1,
        "model": "URPM",
        "source_image": Image.open("images/test_db0.jpg"),
    },
    {
        # Simple LORA 1
        "desc": "lora1",
        "sampler_name": "k_euler",
        "cfg_scale": 8.0,
        "denoising_strength": 1.0,
        "seed": 304886399544324,
        "height": 512,
        "width": 512,
        "karras": True,
        "tiling": False,
        "hires_fix": False,
        "clip_skip": 1,
        "control_type": None,
        "image_is_control": False,
        "return_control_map": False,
        "prompt": "a dark magical crystal, GlowingRunesAI_paleblue",
        "loras": [{"name": "GlowingRunesAI - konyconi", "model": 1.0, "clip": 1.0}],
        "ddim_steps": 20,
        "n_iter": 1,
        "model": "Anything Diffusion",
    },
    {
        # Simple LORA 2
        "desc": "lora2",
        "sampler_name": "k_euler",
        "cfg_scale": 8.0,
        "denoising_strength": 1.0,
        "seed": 304886399545324,
        "height": 512,
        "width": 512,
        "karras": True,
        "tiling": False,
        "hires_fix": False,
        "clip_skip": 1,
        "control_type": None,
        "image_is_control": False,
        "return_control_map": False,
        "prompt": "a dark magical crystal, GlowingRunesAI_green, Dr490nSc4leAI",
        "loras": [
            {"name": "GlowingRunesAI - konyconi", "model": 1.0, "clip": 1.0},
            {"name": "dra9onscaleai", "model": 1.0, "clip": 1.0},
        ],
        "ddim_steps": 20,
        "n_iter": 1,
        "model": "Abyss OrangeMix",
    },
    # Differenet sizes
    {
        # Simple canny control net
        "sampler_name": "k_euler",
        "desc": "canny",
        "cfg_scale": 7.5,
        "denoising_strength": 1.0,
        "seed": 1234897333,
        "height": 768,
        "width": 512,
        "karras": True,
        "tiling": False,
        "hires_fix": False,
        "clip_skip": 1,
        "control_type": "canny",
        "image_is_control": False,
        "return_control_map": False,
        "prompt": "a man walking in heaven",
        "ddim_steps": 25,
        "n_iter": 1,
        "model": "Counterfeit",
        "source_image": Image.open("images/test_db0.jpg"),
        "source_processing": "img2img",
    },
    {
        # Simple txt2img
        "desc": "txt2img",
        "sampler_name": "k_euler",
        "cfg_scale": 7.5,
        "denoising_strength": 1.0,
        "seed": 346666,
        "height": 768,
        "width": 512,
        "karras": True,
        "tiling": False,
        "hires_fix": False,
        "clip_skip": 1,
        "control_type": None,
        "image_is_control": False,
        "return_control_map": False,
        "prompt": "a dog",
        "ddim_steps": 25,
        "n_iter": 1,
        "model": "ChilloutMix",
        "source_image": None,
        "source_processing": "txt2img",
    },
    {
        # Simple img2img
        "desc": "img2img",
        "sampler_name": "k_euler",
        "cfg_scale": 7.5,
        "denoising_strength": 0.4,
        "seed": 250636385744582,
        "height": 768,
        "width": 512,
        "karras": True,
        "tiling": False,
        "hires_fix": False,
        "clip_skip": 1,
        "control_type": None,
        "image_is_control": False,
        "return_control_map": False,
        "prompt": "a dinosaur",
        "ddim_steps": 25,
        "n_iter": 1,
        "model": "Epic Diffusion",
        "source_image": Image.open("images/test_db0.jpg"),
    },
    {
        # Simple LORA 1
        "desc": "lora1",
        "sampler_name": "k_euler",
        "cfg_scale": 8.0,
        "denoising_strength": 1.0,
        "seed": 304886399544324,
        "height": 768,
        "width": 512,
        "karras": True,
        "tiling": False,
        "hires_fix": False,
        "clip_skip": 1,
        "control_type": None,
        "image_is_control": False,
        "return_control_map": False,
        "prompt": "a dark magical crystal, arcane style",
        "loras": [{"name": "arcane style lora", "model": 1.0, "clip": 1.0}],
        "ddim_steps": 20,
        "n_iter": 1,
        "model": "Babes",
    },
    {
        # Simple LORA 2
        "desc": "lora2",
        "sampler_name": "k_euler",
        "cfg_scale": 8.0,
        "denoising_strength": 1.0,
        "seed": 304886399545324,
        "height": 768,
        "width": 512,
        "karras": True,
        "tiling": False,
        "hires_fix": False,
        "clip_skip": 1,
        "control_type": None,
        "image_is_control": False,
        "return_control_map": False,
        "prompt": "a closeup portrait photo of a womans face, ahegao, ahri",
        "loras": [
            {"name": "ahegao", "model": 1.0, "clip": 1.0},
            {"name": "ahri (league of legends) lora", "model": 1.0, "clip": 1.0},
        ],
        "ddim_steps": 20,
        "n_iter": 1,
        "model": "ICBINP - I Can't Believe It's Not Photography",
    },
]

if FILTER_JOBS:
    ACTIVE_JOBS = [x for x in THE_JOBS if x["desc"] in FILTER_JOBS]
else:
    ACTIVE_JOBS = THE_JOBS

start_time = time.time()

mutex = threading.Lock()
count = 0


def inc():
    with mutex:
        global count
        count += 1
        return count


def run_iterations():
    random.seed()
    for i in range(ITERATIONS):
        next_job = inc()
        job_num = random.randint(0, len(ACTIVE_JOBS) - 1)
        data = copy.deepcopy(ACTIVE_JOBS[job_num])
        logger.info(f"Starting job {next_job}")
        pil_image = generate.basic_inference_single_image(data).image
        logger.info(f"Ended job {next_job}")
        if pil_image:
            pil_image.save(
                f"{out_dir}/{data['desc']}-group_{job_num}-it_{i}_total_{next_job}-{threading.current_thread().ident}.webp",
                quality=90,
            )
        else:
            with open(
                f"{out_dir}/{data['desc']}-group_{job_num}-it_{i}_total_{next_job}-{threading.current_thread().ident}.txt",
                "w",
            ) as f:
                f.write("failed")


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
    [x.join() for x in threads]

    logger.warning(f"Test took {round(time.time() - start_time)} seconds ({count} generations)")


main()
