# This tests running hordelib standalone, as an external caller would use it.
# Call with: python -m test.run_txt2img_local_model
# You need all the deps in whatever environment you are running this.

import hordelib
from hordelib.consts import MODEL_CATEGORY_NAMES


def main():
    hordelib.initialise()

    from hordelib.horde import HordeLib
    from hordelib.shared_model_manager import SharedModelManager

    generate = HordeLib()
    SharedModelManager.load_model_managers([MODEL_CATEGORY_NAMES.compvis])
    localfile = "cyberrealistic_v13.safetensors"
    SharedModelManager.manager.load(localfile, local=True)

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
        "prompt": (
            "(masterpiece, photorealistic, raw,:1.4), (extremely intricate:1.2), "
            "close up, cinematic light, sidelighting, ultra high res, best shadow, "
            "RAW, upper body, old man, wearing pullover"
        ),
        "ddim_steps": 25,
        "n_iter": 1,
        "model": localfile,
    }
    pil_image = generate.basic_inference_single_image(data).image
    pil_image.save("images/run_txt2img_local.webp", quality=90)


if __name__ == "__main__":
    main()
