# Here we try to measure the impact, if any, on inference time with long prompts
import time

import hordelib
from hordelib.consts import MODEL_CATEGORY_NAMES

metrics = {}
last_label = ""


def start_clock(label):
    global metrics
    global last_label
    metrics[label] = time.time()
    last_label = label


def stop_clock():
    global metrics
    start_time = metrics[last_label]
    metrics[last_label] = f"{round(time.time() - start_time, 2)} seconds"


def main():
    hordelib.initialise(setup_logging=False)

    from hordelib.horde import HordeLib
    from hordelib.settings import UserSettings
    from hordelib.shared_model_manager import SharedModelManager

    generate = HordeLib()
    SharedModelManager.load_model_managers([MODEL_CATEGORY_NAMES.compvis])
    SharedModelManager.manager.load("Deliberate")

    # As basic as we can get data
    basic_data = {
        "sampler_name": "k_euler",
        "cfg_scale": 7.5,
        "denoising_strength": 1.0,
        "seed": 123456789,
        "height": 512,
        "width": 512,
        "karras": False,
        "tiling": False,
        "hires_fix": False,
        "clip_skip": 1,
        "control_type": None,
        "image_is_control": False,
        "return_control_map": False,
        "prompt": "a dog",
        "ddim_steps": 50,
        "n_iter": 1,
        "model": "Deliberate",
    }

    # Same but with a long prompt
    long_prompt_data = {
        "sampler_name": "k_euler",
        "cfg_scale": 7.5,
        "denoising_strength": 1.0,
        "seed": 123456789,
        "height": 512,
        "width": 512,
        "karras": False,
        "tiling": False,
        "hires_fix": False,
        "clip_skip": 1,
        "control_type": None,
        "image_is_control": False,
        "return_control_map": False,
        "prompt": (
            "a dog in a field with a burning bird flying over a blue river with green fish swimming under the surface "
            "all part of an oil painting in the style of Picasso hanging on the wall in an abandoned museum "
            "with old wooden floor boards and a broken window in the background. Outside the window the sun "
            "is setting and off in the distance can be seen a dog in a field with a burning bird flying over "
            "a blue river"
        ),
        "ddim_steps": 50,
        "n_iter": 1,
        "model": "Deliberate",
    }

    # Warmup
    generate.basic_inference_single_image(basic_data)

    i = 1

    # Do this a few times to be sure
    for _ in range(3):
        # let us enable comfyui's default behaviour of batch optimisations
        UserSettings.enable_batch_optimisations.activate()
        start_clock(f"{i}a. Inference with default comfyui")
        generate.basic_inference_single_image(basic_data)
        stop_clock()

        # no batch optimisations
        UserSettings.enable_batch_optimisations.disable()
        start_clock(f"{i}b. Inference with comfyui batch optimisations disabled")
        generate.basic_inference_single_image(basic_data)
        stop_clock()

        i += 1

    # Try the same thing with a really long prompt
    for _ in range(3):
        # let us enable comfyui's default behaviour of batch optimisations
        UserSettings.enable_batch_optimisations.activate()
        start_clock(f"{i}a. Long Prompt Inference with default comfyui")
        generate.basic_inference_single_image(long_prompt_data)
        stop_clock()

        # no batch optimisations
        UserSettings.enable_batch_optimisations.disable()
        start_clock(f"{i}b. Long Prompt Inference with comfyui batch optimisations disabled")
        generate.basic_inference_single_image(long_prompt_data)
        stop_clock()

        i += 1

    # Dump the results
    for k, v in metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
