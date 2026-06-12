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

    @pytest.mark.default_sd15_model
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

    @pytest.mark.default_sd15_model
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

        graph, _, _, _ = hordelib_instance._materialize_image_graph(dict(data))
        basic_graph, _, _, _ = hordelib_instance._materialize_image_graph(dict(basic_ti_payload_data))

        prompt_text = graph.node("prompt")["inputs"]["text"]
        negative_prompt_text = graph.node("negative_prompt")["inputs"]["text"]

        assert prompt_text == basic_graph.node("prompt")["inputs"]["text"]
        assert negative_prompt_text == basic_graph.node("negative_prompt")["inputs"]["text"]

        assert "(embedding:7523:1.0)" in prompt_text
        assert "(embedding:7808:0.5)" in negative_prompt_text
        assert "(embedding:64870:1.0)" in negative_prompt_text

        pil_image = hordelib_instance.basic_inference_single_image(data).image
        assert pil_image is not None

        img_filename = "ti_inject.png"
        pil_image.save(f"images/{img_filename}", quality=100)

        assert check_single_lora_image_similarity(
            f"images_expected/{img_filename}",
            pil_image,
        )

    @pytest.mark.default_sd15_model
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

        graph, _, _, _ = hordelib_instance._materialize_image_graph(dict(data))

        assert "(embedding:7523:1.0)" in graph.node("prompt")["inputs"]["text"]
        assert "(embedding:7808:0.5)" in graph.node("negative_prompt")["inputs"]["text"]
        assert "(embedding:64870:1.0)" in graph.node("negative_prompt")["inputs"]["text"]

        pil_image = hordelib_instance.basic_inference_single_image(data).image
        assert pil_image is not None

        img_filename = "ti_bad_inject.png"
        pil_image.save(f"images/{img_filename}", quality=100)

        assert check_single_lora_image_similarity(
            f"images_expected/{img_filename}",
            pil_image,
        )
