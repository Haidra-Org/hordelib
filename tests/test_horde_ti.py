# test_horde_ti.py
import os
from datetime import datetime, timedelta

import pytest
from PIL import Image

from hordelib.horde import HordeLib
from hordelib.shared_model_manager import SharedModelManager

from .testing_shared_functions import check_single_lora_image_similarity


class TestHordeTI:
    @pytest.fixture(autouse=True, scope="class")
    def setup_and_teardown(self, shared_model_manager: type[SharedModelManager]):
        # shared_model_manager.manager.lora.download_default_loras()
        # shared_model_manager.manager.lora.wait_for_downloads()
        yield
        shared_model_manager.manager.lora.stop_all()

    def test_basic_ti(
        self,
        hordelib_instance: HordeLib,
        stable_diffusion_modelname_for_testing: str,
    ):

        data = {
            "sampler_name": "k_euler",
            "cfg_scale": 8.0,
            "denoising_strength": 1.0,
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
            "prompt": "Closeup portrait of a Lesotho teenage girl wearing a Seanamarena blanket, walking in a field of flowers, holding a bundle of flowers, detailed background, light rays, atmospheric lighting, embedding:style-sylvamagic###(embedding:easynegative:0.5), embedding:bhands-neg",
            "ddim_steps": 20,
            "n_iter": 1,
            "model": stable_diffusion_modelname_for_testing,
        }

        pil_image = hordelib_instance.basic_inference(data)
        assert pil_image is not None

        img_filename = "ti_basic.png"
        pil_image.save(f"images/{img_filename}", quality=100)

        # assert check_single_lora_image_similarity(
        #     f"images_expected/{img_filename}",
        #     pil_image,
        # )
