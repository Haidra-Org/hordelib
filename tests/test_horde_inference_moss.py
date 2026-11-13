# test_horde_inference_moss.py

import threading
from collections.abc import Iterable

import pytest
import torch
from PIL import Image

from hordelib.horde import HordeLib

from .testing_shared_functions import check_single_inference_image_similarity


class TestHordeInferenceMossSFT:
    @pytest.mark.default_moss_sft_model
    def test_moss_sft_text_to_image(
        self,
        hordelib_instance: HordeLib,
        moss_sft_base_model_name: str,
    ):
        data = {
            "sampler_name": "er_sde",
            "cfg_scale": 2.5,
            "denoising_strength": 1.0,
            "seed": 1413,
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
                'a solarpunk text that says "MOSS-Image-SFT Horde Engine" in vibrant solar lights, vibrant colors, '
                "futuristic mossy cityscape background, high detail, digital art"
            ),
            "ddim_steps": 25,
            "n_iter": 1,
            "model": moss_sft_base_model_name,
        }
        pil_image = hordelib_instance.basic_inference_single_image(data).image
        assert pil_image is not None
        assert isinstance(pil_image, Image.Image)

        img_filename = "moss_sft_text_to_image.png"
        pil_image.save(f"images/{img_filename}", quality=100)

        assert check_single_inference_image_similarity(
            f"images_expected/{img_filename}",
            pil_image,
        )

    @pytest.mark.default_moss_sft_model
    def test_a_moss_sft_job_finishes_while_the_text_encoder_prefetch_is_mid_read(
        self,
        hordelib_instance: HordeLib,
        moss_sft_base_model_name: str,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """A job finishes with an image while the prefetch of the quantized text encoder is held mid-read.

        Loading the text encoder to the device swaps the encoder's parameters in place. The prefetch thread is
        held at the first tensor for the whole job, so the load runs while the read is still in progress.
        """
        from hordelib.execution import weight_prefetch

        monkeypatch.delenv("HORDELIB_DISABLE_WEIGHT_PREFETCH", raising=False)
        text_encoder_read_started = threading.Event()
        release_text_encoder_read = threading.Event()
        real_touch = weight_prefetch.touch_cpu_weights

        def touch_parked_for_the_text_encoder(tensors: Iterable[torch.Tensor]) -> int:
            if threading.current_thread().name.endswith(":text_encoder"):
                text_encoder_read_started.set()
                release_text_encoder_read.wait(timeout=600)
            return real_touch(tensors)

        monkeypatch.setattr(weight_prefetch, "touch_cpu_weights", touch_parked_for_the_text_encoder)
        data = {
            "sampler_name": "er_sde",
            "cfg_scale": 2.5,
            "denoising_strength": 1.0,
            "seed": 1413,
            "height": 1024,
            "width": 1024,
            "karras": False,
            "tiling": False,
            "hires_fix": False,
            "clip_skip": 1,
            "control_type": None,
            "image_is_control": False,
            "return_control_map": False,
            "prompt": "a mossy stone bridge over a stream, morning light, high detail",
            "ddim_steps": 8,
            "n_iter": 1,
            "model": moss_sft_base_model_name,
        }
        try:
            pil_image = hordelib_instance.basic_inference_single_image(data).image
        finally:
            release_text_encoder_read.set()

        assert text_encoder_read_started.is_set(), "the text encoder's prefetch did not start during the job"
        assert isinstance(pil_image, Image.Image)
