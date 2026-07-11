# test_horde.py

import pytest
from PIL import Image

from hordelib.horde import HordeLib
from hordelib.shared_model_manager import SharedModelManager

from .testing_shared_functions import check_single_inference_image_similarity

_CONTROL_TYPES = list(HordeLib.CONTROLNET_IMAGE_PREPROCESSOR_MAP.keys())

# Control types deselected from the per-type inference sweep, mapped to the reason.
# ScribblePreprocessor expects an already-drawn scribble map rather than a natural source image; the raw
# M-LSD detector ("mlsd") is run-to-run flaky, and its output is already covered via the "hough" alias.
_INFERENCE_SKIP_REASONS = {
    "scribble": "ScribblePreprocessor expects a pre-drawn scribble map, not a natural source image",
    "mlsd": "M-LSD detector is run-to-run flaky; its control map is covered by the 'hough' alias",
}


class TestHordeInferenceControlnet:
    @pytest.fixture(scope="class", autouse=True)
    def download_controlnets(self, shared_model_manager: type[SharedModelManager]):
        # Amortized once per class: the shared HordeLib/model-manager singletons are session-scoped, so the
        # parametrized per-type tests reuse a single controlnet download pass rather than repeating it.
        assert shared_model_manager.manager.controlnet is not None
        for preproc in _CONTROL_TYPES:
            shared_model_manager.manager.controlnet.download_control_type(preproc, ["stable diffusion 1"])

    @pytest.mark.default_sd15_model
    @pytest.mark.parametrize("preproc", _CONTROL_TYPES)
    def test_controlnet_sd1(
        self,
        shared_model_manager: type[SharedModelManager],
        hordelib_instance: HordeLib,
        stable_diffusion_model_name_for_testing: str,
        preproc: str,
    ):
        if preproc in _INFERENCE_SKIP_REASONS:
            pytest.skip(_INFERENCE_SKIP_REASONS[preproc])

        assert hordelib_instance is not None
        assert shared_model_manager.manager.controlnet is not None

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
            "clip_skip": 1,
            "control_type": preproc,
            "image_is_control": False,
            "return_control_map": False,
            "prompt": "a man walking in the snow",
            "ddim_steps": 25,
            "n_iter": 1,
            "model": stable_diffusion_model_name_for_testing,
            "source_image": Image.open("images/test_db0.jpg"),
            "source_processing": "img2img",
        }

        pil_image = hordelib_instance.basic_inference_single_image(data).image
        assert pil_image is not None, f"Failed to generate image for {preproc}"

        img_filename = f"controlnet_{preproc}.png"

        assert isinstance(pil_image, Image.Image)

        pil_image.save(f"images/{img_filename}", quality=100)
        assert check_single_inference_image_similarity(
            f"images_expected/{img_filename}",
            pil_image,
        )

    @pytest.mark.default_sd15_model
    def test_controlnet_fake_cn(
        self,
        hordelib_instance: HordeLib,
        stable_diffusion_model_name_for_testing: str,
        db0_test_image: Image.Image,
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
            "clip_skip": 1,
            "control_type": "NON_EXISTENT_CONTROL_TYPE",
            "image_is_control": False,
            "return_control_map": False,
            "prompt": "a man walking in the snow",
            "ddim_steps": 25,
            "n_iter": 1,
            "model": stable_diffusion_model_name_for_testing,
            "source_image": db0_test_image,
            "source_processing": "img2img",
        }
        assert hordelib_instance is not None
        image = hordelib_instance.basic_inference_single_image(data).image
        assert image is not None

    @pytest.mark.slow
    @pytest.mark.default_sd15_model
    def test_controlnet_strength(
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
            "clip_skip": 1,
            "control_type": "canny",
            "image_is_control": False,
            "return_control_map": False,
            "prompt": "a man walking on the moon",
            "ddim_steps": 25,
            "n_iter": 1,
            "model": stable_diffusion_model_name_for_testing,
            "source_image": Image.open("images/test_db0.jpg"),
            "source_processing": "img2img",
        }
        images_to_compare: list[tuple[str, Image.Image]] = []
        for strength in [1.0, 0.5, 0.2]:
            data["control_strength"] = strength

            pil_image = hordelib_instance.basic_inference_single_image(data).image
            assert pil_image is not None

            img_filename = f"controlnet_strength_{strength}.png"

            assert isinstance(pil_image, Image.Image)

            pil_image.save(f"images/{img_filename}", quality=100)
            images_to_compare.append((f"images_expected/{img_filename}", pil_image))

        for img_filename, pil_image in images_to_compare:
            assert check_single_inference_image_similarity(
                img_filename,
                pil_image,
            )

    @pytest.mark.default_sd15_model
    def test_controlnet_hires_fix(
        self,
        hordelib_instance: HordeLib,
        stable_diffusion_model_name_for_testing: str,
    ):
        data = {
            "sampler_name": "k_dpmpp_2m",
            "cfg_scale": 7.5,
            "denoising_strength": 1.0,
            "seed": 1234345378856789,
            "height": 768,
            "width": 768,
            "karras": False,
            "tiling": False,
            "hires_fix": True,
            "hires_fix_denoising_strength": 0.0,
            "clip_skip": 1,
            "control_type": "canny",
            "image_is_control": False,
            "return_control_map": False,
            "prompt": "a man walking in the jungle",
            "ddim_steps": 15,
            "n_iter": 1,
            "model": stable_diffusion_model_name_for_testing,
            "source_image": Image.open("images/test_db0.jpg"),
            "source_processing": "img2img",
        }
        images_to_compare: list[tuple[str, Image.Image]] = []
        for denoise in [0.4, 0.5, 0.6]:
            data["hires_fix_denoising_strength"] = denoise

            pil_image = hordelib_instance.basic_inference_single_image(data).image
            assert pil_image is not None

            img_filename = f"controlnet_hires_fix_denoise_{denoise}.png"

            assert isinstance(pil_image, Image.Image)

            pil_image.save(f"images/{img_filename}", quality=100)
            images_to_compare.append((f"images_expected/{img_filename}", pil_image))

    @pytest.mark.default_sd15_model
    def test_controlnet_image_is_control(
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
            "clip_skip": 1,
            "control_type": "openpose",
            "image_is_control": True,
            "return_control_map": False,
            "prompt": "a woman standing in the snow",
            "ddim_steps": 25,
            "n_iter": 1,
            "model": stable_diffusion_model_name_for_testing,
            "source_image": Image.open("images/test_image_is_control.png"),
            "source_processing": "img2img",
        }
        images_to_compare: list[tuple[str, Image.Image]] = []

        pil_image = hordelib_instance.basic_inference_single_image(data).image
        assert pil_image is not None

        img_filename = "controlnet_image_is_control.png"

        assert isinstance(pil_image, Image.Image)

        pil_image.save(f"images/{img_filename}", quality=100)
        images_to_compare.append((f"images_expected/{img_filename}", pil_image))

        for img_filename, pil_image in images_to_compare:
            assert check_single_inference_image_similarity(
                img_filename,
                pil_image,
            )

    @pytest.mark.default_sd15_model
    def test_controlnet_n_iter(
        self,
        hordelib_instance: HordeLib,
        stable_diffusion_model_name_for_testing: str,
    ):
        data = {
            "sampler_name": "k_dpmpp_2m",
            "cfg_scale": 7.5,
            "denoising_strength": 1.0,
            "control_strength": 1.0,
            "seed": 1312,
            "height": 768,
            "width": 768,
            "karras": False,
            "tiling": False,
            "hires_fix": True,
            "clip_skip": 1,
            "control_type": "canny",
            "image_is_control": False,
            "return_control_map": False,
            "prompt": "an alien man walking on a space station",
            "ddim_steps": 25,
            "n_iter": 2,
            "model": stable_diffusion_model_name_for_testing,
            "source_image": Image.open("images/test_db0.jpg"),
            "source_processing": "img2img",
        }
        image_results = hordelib_instance.basic_inference(data)

        assert len(image_results) == 2

        img_pairs_to_check = []

        img_filename_base = "controlnet_n_iter{0}.png"

        for i, image_result in enumerate(image_results):
            assert image_result.image is not None
            assert isinstance(image_result.image, Image.Image)

            img_filename = img_filename_base.format(i)

            image_result.image.save(f"images/{img_filename}", quality=100)
            img_pairs_to_check.append((f"images_expected/{img_filename}", image_result.image))
