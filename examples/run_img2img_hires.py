# This tests running hordelib standalone, as an external caller would use it.
# Call with: python -m test.run_img2img_hires
# You need all the deps in whatever environment you are running this.

import hordelib
from hordelib.consts import MODEL_CATEGORY_NAMES


def main():
    hordelib.initialise()

    from PIL import Image

    from hordelib.shared_model_manager import SharedModelManager
    from hordelib.horde import HordeLib

    generate = HordeLib()
    SharedModelManager.load_model_managers([MODEL_CATEGORY_NAMES.compvis])
    SharedModelManager.manager.load("Deliberate")

    data = {
        "sampler_name": "k_dpmpp_2m",
        "cfg_scale": 7.5,
        "denoising_strength": 0.3,
        "seed": 250636385744582,
        "height": 1024,
        "width": 1024,
        "karras": False,
        "tiling": False,
        "hires_fix": True,
        "clip_skip": 1,
        "control_type": None,
        "image_is_control": False,
        "return_control_map": False,
        "prompt": "a dinosaur",
        "ddim_steps": 25,
        "n_iter": 1,
        "model": "Deliberate",
        "source_image": Image.open("images/test_db0.jpg"),
    }
    pil_image = generate.basic_inference_single_image(data)
    pil_image.save("images/run_img2img_hires.webp", quality=90)


if __name__ == "__main__":
    main()
