# This tests running hordelib standalone, as an external caller would use it.
# Call with: python -m test.run_performance_test
# You need all the deps in whatever environment you are running this.
import os
import time

import hordelib


def main():
    hordelib.initialise()

    import threading

    from loguru import logger
    from PIL import Image

    from hordelib.horde import HordeLib
    from hordelib.shared_model_manager import SharedModelManager

    generate = HordeLib()
    SharedModelManager.loadModelManagers(compvis=True)
    SharedModelManager.manager.load("Deliberate")
    SharedModelManager.manager.load("Anything Diffusion")
    SharedModelManager.manager.load("Realistic Vision")
    SharedModelManager.manager.load("Papercutcraft")

    def generate_images(model, threadid, count):
        data = {
            "sampler_name": "k_dpmpp_2m",
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
            "ddim_steps": 25,
            "n_iter": 1,
            "model": model,
        }
        horde = HordeLib()
        for i in range(count):
            pil_image = horde.basic_inference(data)
            pil_image.save(f"images/perftest/image_{threadid}_iter_{i}.webp", quality=90)

    def generate_images_portrait(model, threadid, count):
        data = {
            "sampler_name": "k_dpmpp_2m",
            "cfg_scale": 7.5,
            "denoising_strength": 1.0,
            "seed": 123456789,
            "height": 512,
            "width": 768,
            "karras": True,
            "tiling": False,
            "hires_fix": False,
            "clip_skip": 1,
            "control_type": None,
            "image_is_control": False,
            "return_control_map": False,
            "prompt": "an ancient llamia monster",
            "ddim_steps": 25,
            "n_iter": 1,
            "model": model,
        }
        horde = HordeLib()
        for i in range(count):
            pil_image = horde.basic_inference(data)
            pil_image.save(f"images/perftest/image_{threadid}_iter_{i}.webp", quality=90)

    def generate_images_cnet(model, threadid, count):
        data = {
            "sampler_name": "k_dpmpp_2m",
            "cfg_scale": 7.5,
            "denoising_strength": 1.0,
            "seed": 123456789,
            "height": 512,
            "width": 768,
            "karras": True,
            "tiling": False,
            "hires_fix": False,
            "clip_skip": 1,
            "control_type": "",
            "image_is_control": False,
            "return_control_map": False,
            "prompt": "a man walking in the snow",
            "ddim_steps": 25,
            "n_iter": 1,
            "model": model,
            "source_image": Image.open("images/test_db0.jpg"),
            "source_processing": "img2img",
        }
        for i in range(count):
            horde = HordeLib()
            pil_image = horde.basic_inference(data)
            pil_image.save(f"images/perftest/image_{threadid}_iter_{i}.webp", quality=90)

    def generate_images_img2img(model, threadid, count):
        data = {
            "sampler_name": "k_dpmpp_2m",
            "cfg_scale": 7.5,
            "denoising_strength": 0.4,
            "seed": 666,
            "height": 768,
            "width": 512,
            "karras": False,
            "tiling": False,
            "hires_fix": False,
            "clip_skip": 1,
            "control_type": None,
            "image_is_control": False,
            "return_control_map": False,
            "prompt": "a dinosaur",
            "ddim_steps": 25,
            "n_iter": 1,
            "model": "Deliberate",
            "source_image": Image.open("images/test_db0.jpg"),
            "source_processing": "img2img",
        }
        for i in range(count):
            horde = HordeLib()
            pil_image = horde.basic_inference(data)
            pil_image.save(f"images/perftest/image_{threadid}_iter_{i}.webp", quality=90)

    if not os.path.exists("images/perftest"):
        os.makedirs("images/perftest")

    start_time = time.time()
    threads = []
    threads.append(
        threading.Thread(
            daemon=True,
            target=generate_images,
            kwargs={"model": "Deliberate", "threadid": 1, "count": 10},
        ),
    )
    threads.append(
        threading.Thread(
            daemon=True,
            target=generate_images_portrait,
            kwargs={"model": "Anything Diffusion", "threadid": 2, "count": 10},
        ),
    )
    threads.append(
        threading.Thread(
            daemon=True,
            target=generate_images_cnet,
            kwargs={"model": "Realistic Vision", "threadid": 3, "count": 10},
        ),
    )
    threads.append(
        threading.Thread(
            daemon=True,
            target=generate_images_img2img,
            kwargs={"model": "Papercutcraft", "threadid": 4, "count": 10},
        ),
    )
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    logger.warning(f"Testing completed in {round(time.time()-start_time)} seconds")


if __name__ == "__main__":
    main()
