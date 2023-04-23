# This tests running hordelib standalone, as an external caller would use it.
# Call with: python -m test.run_txt2img
# You need all the deps in whatever environment you are running this.
import os
import time

from loguru import logger

import hordelib


def main():
    hordelib.initialise(setup_logging=False)

    from hordelib.horde import HordeLib
    from hordelib.shared_model_manager import SharedModelManager

    generate = HordeLib()
    SharedModelManager.loadModelManagers(compvis=True)
    SharedModelManager.manager.load("stable_diffusion")

    ITS = 3000
    data = {
        "sampler_name": "euler",
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
        "ddim_steps": ITS,
        "n_iter": 1,
        "model": "stable_diffusion",
    }
    starttime = time.time()
    pil_image = generate.basic_inference(data)
    endtime = time.time()
    if not pil_image:
        raise Exception("Image generation failed")

    total = round(endtime - starttime, 1)
    itsec = round(ITS / total, 2)

    logger.warning(f"Total time was {total} seconds, empirical overall it/s of {itsec}")


if __name__ == "__main__":
    main()
