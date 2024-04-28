# test_horde.py

from PIL import Image

from hordelib.horde import HordeLib

from .testing_shared_functions import check_single_inference_image_similarity


class TestHordeInference:
    def test_custom_model_text_to_image(
        self,
        hordelib_instance: HordeLib,
        custom_model_info_for_testing: tuple[str, str, str, str],
    ):
        model_name, _, _, _ = custom_model_info_for_testing
        data = {
            "sampler_name": "k_euler_a",
            "cfg_scale": 7.5,
            "denoising_strength": 1.0,
            "seed": 1312,
            "height": 1024,
            "width": 1024,
            "karras": False,
            "tiling": False,
            "hires_fix": False,
            "clip_skip": 2,
            "control_type": None,
            "image_is_control": False,
            "return_control_map": False,
            "prompt": (
                "surreal,amazing quality,masterpiece,best quality,awesome,inspiring,cinematic composition"
                ",soft shadows,Film grain,shallow depth of field,highly detailed,high budget,cinemascope,epic,"
                "OverallDetail,color graded cinematic,atmospheric lighting,imperfections,natural,shallow dof,"
                "1girl,solo,looking at viewer,kurumi_ebisuzawa,twin tails,hair ribbon,leather jacket,leather pants,"
                "black jacket,tight pants,black chocker,zipper,fingerless gloves,biker clothes,spikes,unzipped,"
                "shoulder spikes,multiple belts,shiny clothes,(graffiti:1.2),brick wall,dutch angle,crossed arms,"
                "arms under breasts,anarchist mask,v-shaped eyebrows"
            ),
            "ddim_steps": 30,
            "n_iter": 1,
            "model": model_name,
        }
        pil_image = hordelib_instance.basic_inference_single_image(data).image
        assert pil_image is not None
        assert isinstance(pil_image, Image.Image)

        img_filename = "custom_model_text_to_image.png"
        pil_image.save(f"images/{img_filename}", quality=100)

        assert check_single_inference_image_similarity(
            f"images_expected/{img_filename}",
            pil_image,
        )
