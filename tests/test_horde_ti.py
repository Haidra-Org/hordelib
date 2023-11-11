# test_horde_ti.py
import os
from pathlib import Path

import pytest

from hordelib.horde import HordeLib
from hordelib.shared_model_manager import SharedModelManager

from .testing_shared_functions import check_single_lora_image_similarity


class TestHordeTI:
    @pytest.fixture(scope="class")
    def basic_ti_payload_data(
        self,
        stable_diffusion_model_name_for_testing: str,
    ) -> dict:
        return {
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
            "prompt": "(embedding:7523:1.0),Closeup portrait of a Lesotho teenage girl wearing a Seanamarena blanket, "
            "walking in a field of flowers, (holding a bundle of flowers:1.2), detailed background, light rays, "
            "atmospheric lighting###(embedding:7808:0.5),(embedding:64870:1.0)",
            "tis": [
                {"name": 7523},
                {"name": 7808},
                {"name": 64870},
            ],
            "ddim_steps": 20,
            "n_iter": 1,
            "model": stable_diffusion_model_name_for_testing,
        }

    def test_basic_ti(
        self,
        shared_model_manager: type[SharedModelManager],
        hordelib_instance: HordeLib,
        basic_ti_payload_data,
    ):
        assert shared_model_manager.manager.ti

        pil_image = hordelib_instance.basic_inference_single_image(basic_ti_payload_data).image
        assert pil_image is not None
        assert (
            Path(os.path.join(shared_model_manager.manager.ti.model_folder_path, "64870.safetensors")).exists() is True
        )

        img_filename = "ti_basic.png"
        pil_image.save(f"images/{img_filename}", quality=100)

    def test_inject_ti(
        self,
        hordelib_instance: HordeLib,
        stable_diffusion_model_name_for_testing: str,
        basic_ti_payload_data: dict,
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
            "prompt": "Closeup portrait of a Lesotho teenage girl wearing a Seanamarena blanket, "
            "walking in a field of flowers, (holding a bundle of flowers:1.2), detailed background, light rays, "
            "atmospheric lighting",
            "tis": [
                {"name": 7523, "inject_ti": "prompt", "strength": 1.0},
                {"name": 7808, "inject_ti": "negprompt", "strength": 0.5},
                {"name": 64870, "inject_ti": "negprompt", "strength": 1.0},
            ],
            "ddim_steps": 20,
            "n_iter": 1,
            "model": stable_diffusion_model_name_for_testing,
        }

        payload, _, _ = hordelib_instance._get_validated_payload_and_pipeline_data(data)

        basic_payload, _, _ = hordelib_instance._get_validated_payload_and_pipeline_data(
            basic_ti_payload_data,
        )

        assert payload["prompt.text"] == basic_payload["prompt.text"]
        assert payload["negative_prompt.text"] == basic_payload["negative_prompt.text"]

        assert "(embedding:7523:1.0)" in payload["prompt.text"]
        assert "(embedding:7808:0.5)" in payload["negative_prompt.text"]
        assert "(embedding:64870:1.0)" in payload["negative_prompt.text"]

        pil_image = hordelib_instance.basic_inference_single_image(data).image
        assert pil_image is not None

        img_filename = "ti_inject.png"
        pil_image.save(f"images/{img_filename}", quality=100)

        assert check_single_lora_image_similarity(
            f"images_expected/{img_filename}",
            pil_image,
        )

    def test_bad_inject_ti(
        self,
        hordelib_instance: HordeLib,
        stable_diffusion_model_name_for_testing: str,
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
            "prompt": "Closeup portrait of a Lesotho teenage girl wearing a Seanamarena blanket, "
            "walking in a field of flowers, (holding a bundle of flowers:1.2), detailed background, light rays, "
            "atmospheric lighting",
            "tis": [
                {"name": 7523, "inject_ti": "prompt", "strength": None},
                {"name": 7808, "inject_ti": "negprompt", "strength": "0.5"},
                {"name": 64870, "inject_ti": "negprompt", "strength": "1.0"},
                {"name": 4629, "inject_ti": "YOLO", "strength": "YOLO"},
            ],
            "ddim_steps": 20,
            "n_iter": 1,
            "model": stable_diffusion_model_name_for_testing,
        }

        payload, _, _ = hordelib_instance._get_validated_payload_and_pipeline_data(data)

        assert "(embedding:7523:1.0)" in payload["prompt.text"]
        assert "(embedding:7808:0.5)" in payload["negative_prompt.text"]
        assert "(embedding:64870:1.0)" in payload["negative_prompt.text"]

        pil_image = hordelib_instance.basic_inference_single_image(data).image
        assert pil_image is not None

        img_filename = "ti_bad_inject.png"
        pil_image.save(f"images/{img_filename}", quality=100)

        assert check_single_lora_image_similarity(
            f"images_expected/{img_filename}",
            pil_image,
        )
