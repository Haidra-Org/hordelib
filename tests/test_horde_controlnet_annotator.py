# test_horde.py
import os

import pytest
from PIL import Image

from hordelib.horde import HordeLib
from hordelib.shared_model_manager import SharedModelManager
from hordelib.utils.distance import are_images_identical


class TestHordeInference:
    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        self.horde = HordeLib()

        self.default_model_manager_args = {
            # aitemplate
            # "blip": True,
            # "clip": True,
            # "codeformer": True,
            # "compvis": True,
            "controlnet": True,
            # "diffusers": True,
            # "esrgan": True,
            # "gfpgan": True,
            # "safety_checker": True,
        }
        SharedModelManager.loadModelManagers(**self.default_model_manager_args)
        assert SharedModelManager.manager is not None
        for preproc in HordeLib.CONTROLNET_IMAGE_PREPROCESSOR_MAP.keys():
            SharedModelManager.manager.controlnet.download_control_type(preproc, ["stable diffusion 1"])
        assert SharedModelManager.preloadAnnotators()
        self.image = Image.open("images/test_annotator.jpg")
        self.width, self.height = self.image.size
        TestHordeInference.distance_threshold = int(os.getenv("IMAGE_DISTANCE_THRESHOLD", "100000"))
        yield
        del self.horde
        SharedModelManager._instance = None
        SharedModelManager.manager = None

    def test_controlnet_annotator(self):
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
            "control_type": "",
            "image_is_control": False,
            "return_control_map": True,
            "prompt": "this is not used here",
            "ddim_steps": 25,
            "n_iter": 1,
            "model": "Deliberate",
            "source_image": Image.open("images/test_annotator.jpg"),
            "source_processing": "img2img",
        }
        assert self.horde is not None

        for preproc in HordeLib.CONTROLNET_IMAGE_PREPROCESSOR_MAP.keys():
            if preproc == "scribble":
                # Not valid for normal image input test
                continue
            assert (
                SharedModelManager.manager.controlnet.check_control_type_available(
                    preproc,
                    "stable diffusion 1",
                )
                is True
            )
            data["control_type"] = preproc
            pil_image = self.horde.basic_inference(data)
            assert pil_image is not None
            img_filename = f"annotator_{preproc}.png"
            pil_image.save(f"images/{img_filename}", quality=100)
            assert are_images_identical(f"images_expected/{img_filename}", pil_image, self.distance_threshold)
