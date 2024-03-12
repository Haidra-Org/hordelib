# test_horde.py

import pytest
from PIL import Image

from hordelib.horde import HordeLib

from .testing_shared_functions import check_list_inference_images_similarity, check_single_inference_image_similarity


class TestHordeInference:
    def test_text_to_image(
        self,
        hordelib_instance: HordeLib,
        stable_diffusion_model_name_for_testing: str,
    ):
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
            "model": stable_diffusion_model_name_for_testing,
        }
        pil_image = hordelib_instance.basic_inference_single_image(data).image
        assert pil_image is not None
        assert isinstance(pil_image, Image.Image)

        img_filename = "text_to_image.png"
        pil_image.save(f"images/{img_filename}", quality=100)

        assert check_single_inference_image_similarity(
            f"images_expected/{img_filename}",
            pil_image,
        )

    def test_text_to_image_n_iter(
        self,
        hordelib_instance: HordeLib,
        stable_diffusion_model_name_for_testing: str,
    ):
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
            "n_iter": 4,
            "model": stable_diffusion_model_name_for_testing,
        }
        image_results = hordelib_instance.basic_inference(data)

        assert len(image_results) == 4

        img_pairs_to_check = []

        img_filename_base = "text_to_image_n_iter_{0}.png"

        for i, image_result in enumerate(image_results):
            assert image_result.image is not None
            assert isinstance(image_result.image, Image.Image)

            img_filename = img_filename_base.format(i)

            image_result.image.save(f"images/{img_filename}", quality=100)
            img_pairs_to_check.append((f"images_expected/{img_filename}", image_result.image))

        assert check_single_inference_image_similarity(
            "images_expected/text_to_image.png",
            "images/text_to_image_n_iter_0.png",
        )

        assert check_list_inference_images_similarity(img_pairs_to_check)

    def test_sdxl_text_to_image(
        self,
        hordelib_instance: HordeLib,
        sdxl_1_0_base_model_name: str,
    ):
        data = {
            "sampler_name": "k_euler",
            "cfg_scale": 4,
            "denoising_strength": 1.0,
            "seed": 123456789,
            "height": 1024,
            "width": 1024,
            "karras": False,
            "tiling": False,
            "hires_fix": False,
            "clip_skip": 1,
            "control_type": None,
            "image_is_control": False,
            "return_control_map": False,
            "prompt": "painting of an cat in a fancy hat in a fancy room",
            "ddim_steps": 25,
            "n_iter": 1,
            "model": sdxl_1_0_base_model_name,
        }

        pil_image = hordelib_instance.basic_inference_single_image(data).image
        assert pil_image is not None
        assert isinstance(pil_image, Image.Image)

        img_filename = "sdxl_text_to_image.png"
        pil_image.save(f"images/{img_filename}", quality=100)

        assert check_single_inference_image_similarity(
            f"images_expected/{img_filename}",
            pil_image,
        )

    @pytest.mark.skip(reason="This test is too slow to run on every test run")
    def test_sdxl_text_to_image_recommended_resolutions(
        self,
        hordelib_instance: HordeLib,
        sdxl_1_0_base_model_name: str,
    ):
        data = {
            "sampler_name": "k_euler",
            "cfg_scale": 4,
            "denoising_strength": 1.0,
            "seed": 123456789,
            "height": 1024,
            "width": 1024,
            "karras": False,
            "tiling": False,
            "hires_fix": False,
            "clip_skip": 1,
            "control_type": None,
            "image_is_control": False,
            "return_control_map": False,
            "prompt": "painting of an cat in a fancy hat in a fancy room",
            "ddim_steps": 15,
            "n_iter": 1,
            "model": sdxl_1_0_base_model_name,
        }

        recommended_resolutions: list[tuple[int, int]] = [
            # (1024, 1024), covered by previous test
            (1152, 896),
            (896, 1152),
            (1216, 832),
            (832, 1216),
            (1344, 768),
            (768, 1344),
            (1536, 640),
            (640, 1536),
        ]
        completed_images: list[tuple[str, Image.Image]] = []
        for resolution in recommended_resolutions:
            data["width"] = resolution[0]
            data["height"] = resolution[1]
            pil_image = hordelib_instance.basic_inference_single_image(data).image
            assert pil_image is not None
            assert isinstance(pil_image, Image.Image)

            img_filename = f"sdxl_text_to_image_{resolution[0]}_{resolution[1]}.png"
            pil_image.save(f"images/{img_filename}", quality=100)

            completed_images.append((img_filename, pil_image))

        for img_filename, pil_image in completed_images:
            assert check_single_inference_image_similarity(
                f"images_expected/{img_filename}",
                pil_image,
            )

    def test_text_to_image_small(
        self,
        hordelib_instance: HordeLib,
        stable_diffusion_model_name_for_testing: str,
    ):
        data = {
            "sampler_name": "k_dpmpp_2m",
            "cfg_scale": 7.5,
            "denoising_strength": 1.0,
            "seed": 32323,
            "height": 320,
            "width": 320,
            "karras": False,
            "tiling": False,
            "hires_fix": False,
            "clip_skip": 1,
            "control_type": None,
            "image_is_control": False,
            "return_control_map": False,
            "prompt": "a photo of cute dinosaur ### painting, drawing, artwork, red border",
            "ddim_steps": 20,
            "n_iter": 1,
            "model": stable_diffusion_model_name_for_testing,
        }
        pil_image = hordelib_instance.basic_inference_single_image(data).image
        assert pil_image is not None
        assert isinstance(pil_image, Image.Image)

        img_filename = "text_to_image_small.png"
        pil_image.save(f"images/{img_filename}", quality=100)

        assert check_single_inference_image_similarity(
            f"images_expected/{img_filename}",
            pil_image,
        )

    def test_text_to_image_clip_skip_2(
        self,
        hordelib_instance: HordeLib,
        stable_diffusion_model_name_for_testing: str,
    ):
        data = {
            "sampler_name": "k_dpmpp_2m",
            "cfg_scale": 7.5,
            "denoising_strength": 1.0,
            "seed": 123456789,
            "height": 512,
            "width": 512,
            "karras": False,
            "tiling": False,
            "hires_fix": False,
            "clip_skip": 2,
            "control_type": None,
            "image_is_control": False,
            "return_control_map": False,
            "prompt": "an ancient llamia monster",
            "ddim_steps": 25,
            "n_iter": 1,
            "model": stable_diffusion_model_name_for_testing,
        }
        pil_image = hordelib_instance.basic_inference_single_image(data).image
        assert pil_image is not None
        assert isinstance(pil_image, Image.Image)

        img_filename = "text_to_image_clip_skip_2.png"
        pil_image.save(f"images/{img_filename}", quality=100)

        assert check_single_inference_image_similarity(
            f"images_expected/{img_filename}",
            pil_image,
        )

    def test_text_to_image_hires_fix(
        self,
        hordelib_instance: HordeLib,
        stable_diffusion_model_name_for_testing: str,
    ):
        data = {
            "sampler_name": "k_dpmpp_2m",
            "cfg_scale": 7.5,
            "denoising_strength": 1.0,
            "seed": 123456789,
            "height": 768,
            "width": 768,
            "karras": False,
            "tiling": False,
            "hires_fix": True,
            "clip_skip": 1,
            "control_type": None,
            "image_is_control": False,
            "return_control_map": False,
            "prompt": "an ancient llamia monster",
            "ddim_steps": 25,
            "n_iter": 1,
            "model": stable_diffusion_model_name_for_testing,
        }
        pil_image = hordelib_instance.basic_inference_single_image(data).image
        assert pil_image is not None
        assert isinstance(pil_image, Image.Image)

        img_filename = "text_to_image_hires_fix.png"
        pil_image.save(f"images/{img_filename}", quality=100)

        assert check_single_inference_image_similarity(
            f"images_expected/{img_filename}",
            pil_image,
        )

    def test_text_to_image_tiling(
        self,
        hordelib_instance: HordeLib,
        stable_diffusion_model_name_for_testing: str,
    ):
        data = {
            "sampler_name": "k_dpmpp_2m",
            "cfg_scale": 7.5,
            "denoising_strength": 1.0,
            "seed": 123456789,
            "height": 512,
            "width": 512,
            "karras": False,
            "tiling": True,
            "hires_fix": True,
            "clip_skip": 1,
            "control_type": None,
            "image_is_control": False,
            "return_control_map": False,
            "prompt": "closeup of rocks, texture, pattern, black and white",
            "ddim_steps": 25,
            "n_iter": 1,
            "model": stable_diffusion_model_name_for_testing,
        }
        pil_image = hordelib_instance.basic_inference_single_image(data).image
        assert pil_image is not None
        assert isinstance(pil_image, Image.Image)

        img_filename = "text_to_image_tiling.png"
        pil_image.save(f"images/{img_filename}", quality=100)

        assert check_single_inference_image_similarity(
            f"images_expected/{img_filename}",
            pil_image,
        )

        data["tiling"] = False
        pil_image_no_tiling = hordelib_instance.basic_inference_single_image(data).image
        assert pil_image_no_tiling is not None
        assert isinstance(pil_image_no_tiling, Image.Image)

        img_no_tiling_filename = "text_to_image_no_tiling.png"
        pil_image_no_tiling.save(f"images/{img_no_tiling_filename}", quality=100)

        assert check_single_inference_image_similarity(
            f"images_expected/{img_no_tiling_filename}",
            pil_image_no_tiling,
        )

    def test_text_to_image_hires_fix_n_iter(
        self,
        hordelib_instance: HordeLib,
        stable_diffusion_model_name_for_testing: str,
    ):
        data = {
            "sampler_name": "k_euler",
            "cfg_scale": 7.5,
            "denoising_strength": 1.0,
            "seed": 1312,
            "height": 768,
            "width": 768,
            "karras": False,
            "tiling": False,
            "hires_fix": True,
            "clip_skip": 1,
            "control_type": None,
            "image_is_control": False,
            "return_control_map": False,
            "prompt": "an ancient charybdis monster",
            "ddim_steps": 25,
            "n_iter": 2,
            "model": stable_diffusion_model_name_for_testing,
        }
        image_results = hordelib_instance.basic_inference(data)

        assert len(image_results) == 2

        img_pairs_to_check = []

        img_filename_base = "text_to_image_hires_fix_n_iter_{0}.png"

        for i, image_result in enumerate(image_results):
            assert image_result.image is not None
            assert isinstance(image_result.image, Image.Image)

            img_filename = img_filename_base.format(i)

            image_result.image.save(f"images/{img_filename}", quality=100)
            img_pairs_to_check.append((f"images_expected/{img_filename}", image_result.image))

        assert check_list_inference_images_similarity(img_pairs_to_check)

    def test_callback_with_post_processors(
        self,
        hordelib_instance: HordeLib,
        stable_diffusion_model_name_for_testing: str,
    ):

        data = {
            "sampler_name": "k_euler",
            "cfg_scale": 7.5,
            "denoising_strength": 1.0,
            "seed": 1312,
            "height": 512,
            "width": 512,
            "karras": False,
            "tiling": False,
            "hires_fix": True,
            "clip_skip": 1,
            "control_type": None,
            "image_is_control": False,
            "return_control_map": False,
            "prompt": "a portrait of an intense woman looking at the camera",
            "ddim_steps": 25,
            "n_iter": 2,
            "post_processing": ["4x_AnimeSharp", "CodeFormers"],
            "model": stable_diffusion_model_name_for_testing,
        }

        from hordelib.horde import ProgressReport, ProgressState

        starting_messages = 0
        post_processing_messages = 0
        finished_messages = 0

        def callback_function(progress_report: ProgressReport):
            assert progress_report is not None
            if progress_report.hordelib_progress_state == ProgressState.started:
                nonlocal starting_messages
                starting_messages += 1
            elif progress_report.hordelib_progress_state == ProgressState.post_processing:
                nonlocal post_processing_messages
                post_processing_messages += 1
            elif progress_report.hordelib_progress_state == ProgressState.finished:
                nonlocal finished_messages
                finished_messages += 1

            if progress_report.comfyui_progress is not None:
                assert progress_report.comfyui_progress.rate == -1 or progress_report.comfyui_progress.rate > 0
                assert progress_report.hordelib_progress_state == ProgressState.progress
                assert progress_report.comfyui_progress.current_step >= 0
                assert progress_report.comfyui_progress.total_steps > 0

        image_results = hordelib_instance.basic_inference(data, progress_callback=callback_function)

        assert len(image_results) == 2
        assert starting_messages == 1
        assert post_processing_messages == 1
        assert finished_messages == 1

        image_filename_base = "text_to_image_callback_{0}.png"

        for i, image_result in enumerate(image_results):
            assert image_result.image is not None
            assert isinstance(image_result.image, Image.Image)

            image_filename = image_filename_base.format(i)
            image_result.image.save(f"images/{image_filename}", quality=100)

            assert check_single_inference_image_similarity(
                f"images_expected/{image_filename}",
                image_result.image,
            )
