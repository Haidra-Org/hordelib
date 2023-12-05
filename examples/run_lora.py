# This tests running hordelib standalone, as an external caller would use it.
# Call with: python -m test.run_txt2img
# You need all the deps in whatever environment you are running this.

import hordelib


def main():
    hordelib.initialise(setup_logging=False)

    from hordelib.consts import MODEL_CATEGORY_NAMES
    from hordelib.horde import HordeLib
    from hordelib.shared_model_manager import SharedModelManager

    generate = HordeLib()
    SharedModelManager.load_model_managers(
        [
            MODEL_CATEGORY_NAMES.compvis,
            MODEL_CATEGORY_NAMES.lora,
        ],
    )
    SharedModelManager.manager.lora.download_default_loras()
    SharedModelManager.manager.lora.wait_for_downloads()
    SharedModelManager.manager.load("Deliberate")

    data = {
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
        "prompt": "a dark magical crystal, GlowingRunesAIV2_red, Dr490nSc4leAI",
        "loras": [
            {"name": "GlowingRunesAIV6", "model": 1.0, "clip": 1.0},
            {"name": "Dra9onScaleAI", "model": 1.0, "clip": 1.0},
        ],
        "ddim_steps": 20,
        "n_iter": 1,
        "model": "Deliberate",
    }
    pil_image = generate.basic_inference_single_image(data).image
    pil_image.save("images/run_lora.webp", quality=90)


if __name__ == "__main__":
    main()
