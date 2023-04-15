# This tests running hordelib standalone, as an external caller would use it.
# Call with: python -m test.run_facefix
# You need all the deps in whatever environment you are running this.
import os

import hordelib


def main():

    hordelib.initialise()

    from PIL import Image

    from hordelib.horde import HordeLib
    from hordelib.shared_model_manager import SharedModelManager

    generate = HordeLib()
    modeltypes = {
        # aitemplate
        "blip": True,
        "clip": True,
        "codeformer": True,
        "compvis": True,
        "controlnet": True,
        "diffusers": True,
        "esrgan": True,
        "gfpgan": True,
        "safety_checker": True,
    }

    SharedModelManager.loadModelManagers(**modeltypes)
    SharedModelManager.manager.load("CodeFormers")

    data = {
        "model": "CodeFormers",
        "source_image": Image.open("images/test_facefix.png"),
    }
    pil_image = generate.image_facefix(data)
    pil_image.save("images/run_facefix.webp", quality=90)


if __name__ == "__main__":
    main()
