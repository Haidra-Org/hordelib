# This tests running hordelib standalone, as an external caller would use it.
# Call with: python -m test.run_controlnet
# You need all the deps in whatever environment you are running this.

import hordelib
from hordelib.consts import MODEL_CATEGORY_NAMES


def main():
    hordelib.initialise()

    from PIL import Image

    from hordelib.horde import HordeLib
    from hordelib.shared_model_manager import SharedModelManager

    generate = HordeLib()
    SharedModelManager.load_model_managers([MODEL_CATEGORY_NAMES.compvis, MODEL_CATEGORY_NAMES.controlnet])
    SharedModelManager.manager.load("Deliberate")

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
        "control_type": "",
        "image_is_control": False,
        "return_control_map": False,
        "prompt": "a man walking in the snow",
        "ddim_steps": 25,
        "n_iter": 1,
        "model": "Deliberate",
        "source_image": Image.open("images/test_db0.jpg"),
        "source_processing": "img2img",
    }
    for preproc in HordeLib.CONTROLNET_IMAGE_PREPROCESSOR_MAP.keys():
        if preproc == "scribble":
            # Not valid for normal image input test
            continue
        data["control_type"] = preproc
        pil_image = generate.basic_inference_single_image(data).image
        pil_image.save(f"images/run_controlnet_{preproc}.webp", quality=90)


if __name__ == "__main__":
    main()
