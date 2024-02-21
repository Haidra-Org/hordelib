# test_horde.py

from PIL import Image

from hordelib.horde import HordeLib


class TestHordeInference:
    def test_sdxl_text_to_image(
        self,
        hordelib_instance: HordeLib,
        stable_cascade_base_model_name: str,
    ):
        data = {
            "sampler_name": "k_euler",
            "cfg_scale": 4,
            "denoising_strength": 1.0,
            "seed": 1313,
            "height": 1024,
            "width": 1024,
            "karras": False,
            "tiling": False,
            "hires_fix": False,
            "clip_skip": 1,
            "control_type": None,
            "image_is_control": False,
            "return_control_map": False,
            "prompt": (
                "A magic glowing long sword lying flat on a medieval shop rack, in the style of Dungeons and Dragons. "
                "Splash art, Digital Painting, ornate handle with gems"
            ),
            "ddim_steps": 20,
            "n_iter": 1,
            "model": stable_cascade_base_model_name,
        }

        pil_image = hordelib_instance.basic_inference_single_image(data).image
        assert pil_image is not None
        assert isinstance(pil_image, Image.Image)

        img_filename = "stable_cascade_text_to_image.png"
        pil_image.save(f"images/{img_filename}", quality=100)

        # assert check_single_inference_image_similarity(
        #     f"images_expected/{img_filename}",
        #     pil_image,
        # )
