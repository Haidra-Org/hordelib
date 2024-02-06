from hordelib.horde import HordeLib
from hordelib.initialisation import initialise


def main():
    initialise()

    hordelib_instance = HordeLib()

    data = {
        "sampler_name": "k_dpmpp_2m",
        "cfg_scale": 7.5,
        "denoising_strength": 1.0,
        "seed": 123456789,
        "height": 512.1,  # test param fix
        "width": 512.1,  # test param fix
        "karras": False,
        "tiling": False,
        "hires_fix": False,
        "clip_skip": 1,
        "control_type": None,
        "image_is_control": False,
        "return_control_map": False,
        "prompt": "an ancient llamia monster",
        "ddim_steps": 25,
        "n_iter": 1,
        "model": "Deliberate",
    }
    pil_image = hordelib_instance.basic_inference_single_image(data).image

    img_filename = "text_to_image.png"
    pil_image.save(f"images/{img_filename}", quality=100)


if __name__ == "__main__":
    main()
