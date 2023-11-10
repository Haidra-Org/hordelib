# This tests running hordelib standalone, as an external caller would use it.
# Call with: python -m test.run_facefix
# You need all the deps in whatever environment you are running this.

import hordelib


def main():
    hordelib.initialise()

    from PIL import Image

    from hordelib.consts import MODEL_CATEGORY_NAMES
    from hordelib.horde import HordeLib
    from hordelib.shared_model_manager import SharedModelManager

    generate = HordeLib()

    SharedModelManager.load_model_managers(
        [
            MODEL_CATEGORY_NAMES.codeformer,
            MODEL_CATEGORY_NAMES.compvis,
            MODEL_CATEGORY_NAMES.controlnet,
            MODEL_CATEGORY_NAMES.esrgan,
            MODEL_CATEGORY_NAMES.gfpgan,
            MODEL_CATEGORY_NAMES.safety_checker,
        ],
    )
    SharedModelManager.manager.load("CodeFormers")

    data = {
        "model": "CodeFormers",
        "source_image": Image.open("images/test_facefix.png"),
    }
    pil_image = generate.image_facefix(data).image
    pil_image.save("images/run_facefix.webp", quality=90)


if __name__ == "__main__":
    main()
