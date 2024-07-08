# test_horde.py

from PIL import Image

from hordelib.horde import HordeLib

from .testing_shared_functions import check_list_inference_images_similarity, check_single_inference_image_similarity


class TestHordeInferenceImg2Img:
    def test_image_to_image(
        self,
        stable_diffusion_model_name_for_testing: str,
        hordelib_instance: HordeLib,
    ):
        data = {
            "sampler_name": "k_dpmpp_2m",
            "cfg_scale": 7.5,
            "denoising_strength": 0.4,
            "seed": 666,
            "height": 512,
            "width": 512,
            "karras": False,
            "tiling": False,
            "hires_fix": False,
            "clip_skip": 1,
            "control_type": None,
            "image_is_control": False,
            "return_control_map": False,
            "prompt": "a dinosaur",
            "ddim_steps": 25,
            "n_iter": 1,
            "model": stable_diffusion_model_name_for_testing,
            "source_image": Image.open("images/test_db0.jpg"),
            "source_processing": "img2img",
        }
        pil_image = hordelib_instance.basic_inference_single_image(data).image
        assert pil_image is not None
        assert isinstance(pil_image, Image.Image)
        assert pil_image.size == (512, 512)

        img_filename = "image_to_image.png"
        pil_image.save(f"images/{img_filename}", quality=100)

        assert check_single_inference_image_similarity(
            f"images_expected/{img_filename}",
            pil_image,
        )

    def test_image_to_image_tiling(
        self,
        stable_diffusion_model_name_for_testing: str,
        hordelib_instance: HordeLib,
    ):
        data = {
            "sampler_name": "k_dpmpp_2m",
            "cfg_scale": 7.5,
            "denoising_strength": 1,
            "seed": 666,
            "height": 512,
            "width": 512,
            "karras": False,
            "tiling": True,
            "hires_fix": False,
            "clip_skip": 1,
            "control_type": None,
            "image_is_control": False,
            "return_control_map": False,
            "prompt": "a dinosaur",
            "ddim_steps": 25,
            "n_iter": 1,
            "model": stable_diffusion_model_name_for_testing,
            "source_image": Image.open("images/test_db0.jpg"),
            "source_processing": "img2img",
        }
        pil_image = hordelib_instance.basic_inference_single_image(data).image
        assert pil_image is not None
        assert isinstance(pil_image, Image.Image)
        assert pil_image.size == (512, 512)

        img_filename = "image_to_image_tiling.png"
        pil_image.save(f"images/{img_filename}", quality=100)

        assert check_single_inference_image_similarity(
            f"images_expected/{img_filename}",
            pil_image,
        )

    def test_sdxl_image_to_image(
        self,
        hordelib_instance: HordeLib,
        sdxl_1_0_base_model_name: str,
    ):
        data = {
            "sampler_name": "k_dpmpp_2m",
            "cfg_scale": 7.5,
            "denoising_strength": 0.7,
            "seed": 666,
            "height": 1024,
            "width": 1024,
            "karras": False,
            "tiling": False,
            "hires_fix": False,
            "clip_skip": 1,
            "control_type": None,
            "image_is_control": False,
            "return_control_map": False,
            "prompt": "a dinosaur man",
            "ddim_steps": 25,
            "n_iter": 1,
            "model": sdxl_1_0_base_model_name,
            "source_image": Image.open("images/test_db0.jpg"),
            "source_processing": "img2img",
        }
        pil_image = hordelib_instance.basic_inference_single_image(data).image
        assert pil_image is not None
        assert isinstance(pil_image, Image.Image)

        img_filename = "sdxl_image_to_image.png"
        pil_image.save(f"images/{img_filename}", quality=100)

        assert pil_image.size == (1024, 1024)

        assert check_single_inference_image_similarity(
            f"images_expected/{img_filename}",
            pil_image,
        )

    def test_image_to_image_hires_fix_small(
        self,
        stable_diffusion_model_name_for_testing: str,
        hordelib_instance: HordeLib,
    ):
        data = {
            "sampler_name": "k_dpmpp_2m",
            "cfg_scale": 7.5,
            "denoising_strength": 0.4,
            "seed": 666,
            "height": 512,
            "width": 512,
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
            "model": stable_diffusion_model_name_for_testing,
            "source_image": Image.open("images/test_db0.jpg"),
            "source_processing": "img2img",
        }
        pil_image = hordelib_instance.basic_inference_single_image(data).image
        assert pil_image is not None
        assert isinstance(pil_image, Image.Image)
        assert pil_image.size == (512, 512)

        img_filename = "image_to_image_hires_fix_small.png"
        pil_image.save(f"images/{img_filename}", quality=100)

        assert check_single_inference_image_similarity(
            f"images_expected/{img_filename}",
            pil_image,
        )

    def test_image_to_image_hires_fix_large(
        self,
        stable_diffusion_model_name_for_testing: str,
        hordelib_instance: HordeLib,
    ):
        data = {
            "sampler_name": "k_dpmpp_2m",
            "cfg_scale": 7.5,
            "denoising_strength": 0.4,
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
            "prompt": "a dinosaur",
            "ddim_steps": 25,
            "n_iter": 1,
            "model": stable_diffusion_model_name_for_testing,
            "source_image": Image.open("images/test_db0.jpg"),
            "source_processing": "img2img",
        }
        assert hordelib_instance is not None
        pil_image = hordelib_instance.basic_inference_single_image(data).image
        assert pil_image is not None
        assert isinstance(pil_image, Image.Image)
        assert pil_image.size == (768, 768)
        img_filename = "image_to_image_hires_fix_large.png"
        pil_image.save(f"images/{img_filename}", quality=100)
        assert check_single_inference_image_similarity(
            f"images_expected/{img_filename}",
            pil_image,
        )

    def test_img2img_masked_denoise_1(
        self,
        stable_diffusion_model_name_for_testing: str,
        sdxl_1_0_base_model_name: str,
        hordelib_instance: HordeLib,
    ):
        data = {
            "sampler_name": "k_dpmpp_2m",
            "cfg_scale": 7.5,
            "denoising_strength": 1,
            "seed": 24636666,
            "height": 512,
            "width": 512,
            "karras": False,
            "clip_skip": 1,
            "prompt": "a mecha robot sitting on a bench",
            "ddim_steps": 20,
            "n_iter": 1,
            "model": stable_diffusion_model_name_for_testing,
            "source_image": Image.open("images/test_img2img_alpha.png"),
            "source_processing": "img2img",
        }

        pil_image = hordelib_instance.basic_inference_single_image(data).image
        assert pil_image is not None
        assert isinstance(pil_image, Image.Image)
        assert pil_image.size == (512, 512)

        img_filename = "img2img_to_masked_denoise_1.png"
        pil_image.save(f"images/{img_filename}", quality=100)

        assert check_single_inference_image_similarity(
            f"images_expected/{img_filename}",
            pil_image,
        )

    def test_img2img_masked_denoise_high(
        self,
        stable_diffusion_model_name_for_testing: str,
        hordelib_instance: HordeLib,
    ):
        data = {
            "sampler_name": "k_dpmpp_2m",
            "cfg_scale": 7.5,
            "denoising_strength": 0.6,
            "seed": 3,
            "height": 512,
            "width": 512,
            "karras": False,
            "clip_skip": 1,
            "prompt": "a mecha robot sitting on a bench",
            "ddim_steps": 20,
            "n_iter": 1,
            "model": stable_diffusion_model_name_for_testing,
            "source_image": Image.open("images/test_img2img_alpha.png"),
            "source_processing": "img2img",
        }
        pil_image = hordelib_instance.basic_inference_single_image(data).image
        assert pil_image is not None
        assert isinstance(pil_image, Image.Image)
        assert pil_image.size == (512, 512)

        img_filename = "img2img_to_masked_denoise_0.6.png"
        pil_image.save(f"images/{img_filename}", quality=100)

        assert check_single_inference_image_similarity(
            f"images_expected/{img_filename}",
            pil_image,
        )

    def test_img2img_masked_denoise_mid(
        self,
        stable_diffusion_model_name_for_testing: str,
        hordelib_instance: HordeLib,
    ):
        data = {
            "sampler_name": "k_dpmpp_2m",
            "cfg_scale": 7.5,
            "denoising_strength": 0.4,
            "seed": 3,
            "height": 512,
            "width": 512,
            "karras": False,
            "clip_skip": 1,
            "prompt": "a mecha robot sitting on a bench",
            "ddim_steps": 20,
            "n_iter": 1,
            "model": stable_diffusion_model_name_for_testing,
            "source_image": Image.open("images/test_img2img_alpha.png"),
            "source_processing": "img2img",
        }
        pil_image = hordelib_instance.basic_inference_single_image(data).image
        assert pil_image is not None
        assert isinstance(pil_image, Image.Image)
        assert pil_image.size == (512, 512)

        img_filename = "img2img_to_masked_denoise_0.4.png"
        pil_image.save(f"images/{img_filename}", quality=100)

        assert check_single_inference_image_similarity(
            f"images_expected/{img_filename}",
            pil_image,
        )

    def test_img2img_masked_denoise_low(
        self,
        stable_diffusion_model_name_for_testing: str,
        hordelib_instance: HordeLib,
    ):
        data = {
            "sampler_name": "k_dpmpp_2m",
            "cfg_scale": 7.5,
            "denoising_strength": 0.2,
            "seed": 3,
            "height": 512,
            "width": 512,
            "karras": False,
            "clip_skip": 1,
            "prompt": "a mecha robot sitting on a bench",
            "ddim_steps": 20,
            "n_iter": 1,
            "model": stable_diffusion_model_name_for_testing,
            "source_image": Image.open("images/test_img2img_alpha.png"),
            "source_processing": "img2img",
        }
        pil_image = hordelib_instance.basic_inference_single_image(data).image
        assert pil_image is not None
        assert isinstance(pil_image, Image.Image)
        assert pil_image.size == (512, 512)

        img_filename = "img2img_to_masked_denoise_0.2.png"
        pil_image.save(f"images/{img_filename}", quality=100)

        assert check_single_inference_image_similarity(
            f"images_expected/{img_filename}",
            pil_image,
        )

    def test_sdxl_img2img_masked_denoise_95(
        self,
        sdxl_1_0_base_model_name: str,
        hordelib_instance: HordeLib,
    ):
        data = {
            "sampler_name": "k_dpmpp_2m",
            "cfg_scale": 7.5,
            "denoising_strength": 0.95,
            "seed": 3,
            "height": 1024,
            "width": 1024,
            "karras": False,
            "clip_skip": 1,
            "prompt": "a mecha robot sitting on a bench",
            "ddim_steps": 20,
            "n_iter": 1,
            "model": sdxl_1_0_base_model_name,
            "source_image": Image.open("images/test_img2img_alpha.png"),
            "source_processing": "img2img",
        }
        pil_image = hordelib_instance.basic_inference_single_image(data).image
        assert pil_image is not None
        assert isinstance(pil_image, Image.Image)
        assert pil_image.size == (1024, 1024)

        img_filename = "sdxl_img2img_to_masked_denoise_0.95.png"
        pil_image.save(f"images/{img_filename}", quality=100)

        assert check_single_inference_image_similarity(
            f"images_expected/{img_filename}",
            pil_image,
        )

    def test_image_to_faulty_source_image(
        self,
        stable_diffusion_model_name_for_testing: str,
        hordelib_instance: HordeLib,
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
            "source_image": "THIS SHOULD FAILOVER TO TEXT2IMG",
            "source_processing": "img2img",
        }
        pil_image = hordelib_instance.basic_inference_single_image(data).image
        assert pil_image is not None
        assert isinstance(pil_image, Image.Image)

        img_filename = "img2img_fallback_to_txt2img.png"
        pil_image.save(f"images/{img_filename}", quality=100)

        assert check_single_inference_image_similarity(
            f"images_expected/{img_filename}",
            pil_image,
        )

    def test_image_to_image_n_iter(
        self,
        hordelib_instance: HordeLib,
        stable_diffusion_model_name_for_testing: str,
    ):
        data = {
            "sampler_name": "k_euler",
            "cfg_scale": 7.5,
            "denoising_strength": 0.6,
            "seed": 1312,
            "height": 512,
            "width": 512,
            "karras": False,
            "tiling": False,
            "hires_fix": False,
            "clip_skip": 1,
            "control_type": None,
            "image_is_control": False,
            "return_control_map": False,
            "prompt": "a weird west goth outlaw walking away into the sunset",
            "ddim_steps": 25,
            "n_iter": 4,
            "model": stable_diffusion_model_name_for_testing,
            "source_image": Image.open("images/test_db0.jpg"),
            "source_processing": "img2img",
        }
        image_results = hordelib_instance.basic_inference(data)

        assert len(image_results) == 4

        img_pairs_to_check = []

        img_filename_base = "image_to_image_n_iter_{0}.png"

        for i, image_result in enumerate(image_results):
            assert image_result.image is not None
            assert isinstance(image_result.image, Image.Image)

            img_filename = img_filename_base.format(i)

            image_result.image.save(f"images/{img_filename}", quality=100)
            img_pairs_to_check.append((f"images_expected/{img_filename}", image_result.image))

        assert check_list_inference_images_similarity(img_pairs_to_check)

    def test_img2img_masked_n_iter(
        self,
        stable_diffusion_model_name_for_testing: str,
        hordelib_instance: HordeLib,
    ):
        data = {
            "sampler_name": "k_dpmpp_2m",
            "cfg_scale": 7.5,
            "denoising_strength": 0.75,
            "seed": 1312,
            "height": 512,
            "width": 512,
            "karras": False,
            "clip_skip": 1,
            "prompt": "a mecha robot sitting on a bench",
            "ddim_steps": 20,
            "n_iter": 2,
            "model": stable_diffusion_model_name_for_testing,
            "source_image": Image.open("images/test_img2img_alpha.png"),
            "source_processing": "img2img",
        }
        image_results = hordelib_instance.basic_inference(data)

        assert len(image_results) == 2

        img_pairs_to_check = []

        img_filename_base = "img2img_masked_n_iter_{0}.png"

        for i, image_result in enumerate(image_results):
            assert image_result.image is not None
            assert isinstance(image_result.image, Image.Image)

            img_filename = img_filename_base.format(i)

            image_result.image.save(f"images/{img_filename}", quality=100)
            img_pairs_to_check.append((f"images_expected/{img_filename}", image_result.image))

        assert check_list_inference_images_similarity(img_pairs_to_check)

    def test_image_to_image_hires_fix_n_iter(
        self,
        stable_diffusion_model_name_for_testing: str,
        hordelib_instance: HordeLib,
    ):
        data = {
            "sampler_name": "k_dpmpp_2m",
            "cfg_scale": 7.5,
            "denoising_strength": 0.5,
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
            "source_image": Image.open("images/test_db0.jpg"),
            "source_processing": "img2img",
        }
        image_results = hordelib_instance.basic_inference(data)

        assert len(image_results) == 2

        img_pairs_to_check = []

        img_filename_base = "img2img_hires_fix_n_iter_{0}.png"

        for i, image_result in enumerate(image_results):
            assert image_result.image is not None
            assert isinstance(image_result.image, Image.Image)

            img_filename = img_filename_base.format(i)

            image_result.image.save(f"images/{img_filename}", quality=100)
            img_pairs_to_check.append((f"images_expected/{img_filename}", image_result.image))

        assert check_list_inference_images_similarity(img_pairs_to_check)
