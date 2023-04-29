import os
import platform
import time
from io import BytesIO

import requests
from loguru import logger
from PIL import Image

from hordelib.utils.gpuinfo import GPUInfo

FACE_FIX_TEST_IMAGE = "https://github.com/jug-dev/hordelib/blob/main/images/test_facefix.png?raw=true"
CONTROLNET_TEST_IMAGE = "https://github.com/jug-dev/hordelib/blob/main/images/test_db0.jpg?raw=true"


def download_image(url):
    response = requests.get(url)
    if response.status_code == 200:
        image_data = BytesIO(response.content)
        pil_image = Image.open(image_data)
        return pil_image
    else:
        raise Exception(f"Failed to download image. Status code: {response.status_code}")


timings = {}


def delta(desc):
    global timings
    if desc not in timings:
        timings[desc] = {"start": time.time(), "end": None}
        return 0
    else:
        timings[desc]["end"] = time.time()
        return timings[desc]["end"] - timings[desc]["start"]


def get_os():
    os_name = platform.system()
    if os_name == "Linux":
        import distro

        distro_name = distro.name()
        distro_version = distro.version()
        os_name = f"{os_name} ({distro_name} {distro_version})"
    elif os_name == "Windows":
        win_version = "-".join(platform.win32_ver())
        os_name = f"{os_name} ({win_version})"
    elif os_name == "Darwin":
        mac_ver, _, _ = platform.mac_ver()
        os_name = f"{os_name} ({mac_ver})"
    else:
        os_name = "Unknown OS"
    return os_name


def get_software_stack():
    try:
        import torch
        import xformers
        from packaging import version

        torch_v = version.parse(torch.__version__)
        xformers_v = version.parse(xformers.__version__)
        return f"Torch {torch_v}, xFormers {xformers_v}"
    except Exception:
        return "Unknown torch and xformers version"


def main():
    if not os.getenv("AIWORKER_CACHE_HOME"):
        print("No model directory found. Environmental variable AIWORKER_CACHE_HOME is not set.")
        exit(1)

    try:
        gpu = GPUInfo().get_info()
        gpu_name = gpu["product"]
        gpu_vram = gpu["vram_total"]
        gpu_vram_mb = GPUInfo().get_total_vram_mb()
    except Exception:
        gpu = "unknown"
        gpu_name = ""
        gpu_vram = ""
        gpu_vram_mb = 0

    disable_controlnet = False
    if gpu_vram_mb < 80000:
        disable_controlnet = True

    delta("initialisation")
    import hordelib

    hordelib.initialise(setup_logging=True)
    delta("initialisation")

    from hordelib.horde import HordeLib
    from hordelib.shared_model_manager import SharedModelManager

    try:
        from hordelib._version import __version__
    except Exception:
        __version__ = "dev branch"

    generate = HordeLib()
    delta("model-manager-load")
    SharedModelManager.loadModelManagers(compvis=True, controlnet=not disable_controlnet)
    delta("model-manager-load")
    SharedModelManager.manager.load("stable_diffusion")

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
        "control_strength": 1.0,
        "clip_skip": 1,
        "control_type": None,
        "image_is_control": False,
        "return_control_map": False,
        "prompt": "an ancient llamia monster ### sea, lake, cartoon",
        "ddim_steps": 100,
        "n_iter": 1,
        "model": "stable_diffusion",
        "source_processing": "txt2img",
    }

    logger.warning("Benchmarking inference")
    # overhead
    delta("basic-inference-overhead")
    data["ddim_steps"] = 1
    pil_image = generate.basic_inference(data)
    if not pil_image:
        raise Exception("Image generation failed")
    last = delta("basic-inference-overhead")
    inference_overhead = last
    # inference
    attempts = [10, 20, 30, 40, 50, 100, 200, 300]
    max_iterations = 0
    for attempt in attempts:
        delta(f"basic-inference-{attempt}-steps")
        data["ddim_steps"] = attempt
        pil_image = generate.basic_inference(data)
        if not pil_image:
            raise Exception("Image generation failed")
        last = delta(f"basic-inference-{attempt}-steps")
        if last > 30:
            break
        max_iterations = attempt
    its = round(attempt / last, 1)
    its_raw = round(attempt / (last - inference_overhead), 1)

    logger.warning("Benchmarking model load/unload")
    delta("stable-diffusion-model-load-x3")
    SharedModelManager.manager.unload_model("stable_diffusion")
    SharedModelManager.manager.load("stable_diffusion")
    SharedModelManager.manager.unload_model("stable_diffusion")
    SharedModelManager.manager.load("stable_diffusion")
    SharedModelManager.manager.unload_model("stable_diffusion")
    SharedModelManager.manager.load("stable_diffusion")
    last = delta("stable-diffusion-model-load-x3")
    model_load = round(last / 3, 1)

    if not disable_controlnet:
        logger.warning("Benchmarking controlnet")
        # approximate pipeline overhead
        data["ddim_steps"] = 1
        data["source_image"] = download_image(CONTROLNET_TEST_IMAGE)
        data["control_type"] = "hed"
        data["source_processing"] = "img2img"
        delta("controlnet-overhead")
        pil_image = generate.basic_inference(data)
        cnet_overhead = round(delta("controlnet-overhead"), 2)
        # inference
        data["ddim_steps"] = max_iterations
        data["source_image"] = download_image(CONTROLNET_TEST_IMAGE)
        data["control_type"] = "hed"
        data["source_processing"] = "img2img"
        delta("controlnet-inference")
        pil_image = generate.basic_inference(data)
        last = delta("controlnet-inference")
        cnet_its = round(max_iterations / last, 1)
        cnet_raw_its = round(max_iterations / (last - cnet_overhead), 1)
        if not pil_image:
            raise Exception("Image generation failed")

    # Display Results
    print()
    print()
    print(f"--- hordelib {__version__} Benchmark ---")
    print(f"    GPU: {gpu_name} {gpu_vram}")
    print(f"     OS: {get_os()}")
    print(f"  Torch: {get_software_stack()}")
    print()
    print("    Iterations per second @ 512x512:")
    print(f"{its:>{9}} basic inference (empirical)")
    print(f"{its_raw:>{9}} basic inference (theoretical)")
    if not disable_controlnet:
        print(f"{cnet_its:>{9}} controlnet inference (empirical)")
        print(f"{cnet_raw_its:>{9}} controlnet inference (theoretical)")
    print()
    print(f"{model_load:>{9}}s model load speed")
    print()
    print("    Details:")
    for name, data in timings.items():
        secs = round(data["end"] - data["start"], 2)
        print(f"{secs:>{9}}s {name}")
    print()


if __name__ == "__main__":
    main()
