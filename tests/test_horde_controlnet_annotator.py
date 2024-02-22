# test_horde.py

import PIL.Image
import pytest

from hordelib.horde import HordeLib
from hordelib.shared_model_manager import SharedModelManager

from .testing_shared_functions import check_single_inference_image_similarity


class TestControlnetAnnotator:
    @pytest.fixture(autouse=True)
    def setup_and_teardown(self, shared_model_manager: type[SharedModelManager]):
        assert shared_model_manager.manager.controlnet
        for preproc in HordeLib.CONTROLNET_IMAGE_PREPROCESSOR_MAP.keys():
            shared_model_manager.manager.controlnet.download_control_type(preproc, ["stable diffusion 1"])
        assert shared_model_manager.preload_annotators()

    def test_controlnet_annotator(
        self,
        hordelib_instance: HordeLib,
        shared_model_manager: type[SharedModelManager],
    ):
        image = PIL.Image.open("images/test_annotator.jpg")
        width, height = image.size
        data = {
            "sampler_name": "k_dpmpp_2m",
            "cfg_scale": 7.5,
            "denoising_strength": 1.0,
            "seed": 123456789,
            "height": height,
            "width": width,
            "karras": False,
            "tiling": False,
            "hires_fix": False,
            "clip_skip": 1,
            "control_type": "",
            "image_is_control": False,
            "return_control_map": True,
            "prompt": "this is not used here",
            "ddim_steps": 25,
            "n_iter": 1,
            "model": "Deliberate",
            "source_image": image,
            "source_processing": "img2img",
        }

        for preproc in HordeLib.CONTROLNET_IMAGE_PREPROCESSOR_MAP.keys():
            if preproc == "scribble":
                # Not valid for normal image input test
                continue
            assert shared_model_manager.manager.controlnet
            data["control_type"] = preproc
            pil_image = hordelib_instance.basic_inference_single_image(data).image
            assert pil_image is not None
            assert isinstance(pil_image, PIL.Image.Image)
            img_filename = f"annotator_{preproc}.png"
            pil_image.save(f"images/{img_filename}", quality=100)
            assert check_single_inference_image_similarity(
                f"images_expected/{img_filename}",
                pil_image,
            )
