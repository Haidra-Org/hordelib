"""The schedules ComfyUI carries as nodes, rendered end to end.

``tests/pipeline/test_sigma_schedules.py`` proves the computed sigmas equal the upstream node's.
These render through the whole pipeline instead, which is what proves the schedule actually reaches
the sampler: the graph's scheduler input carries a placeholder, so a schedule that failed to reach the
run state would still produce a perfectly good image, on the wrong schedule and with nothing raised.
Align Your Steps is covered on both families it has published noise levels for, because the levels are
the part that is per-family.
"""

import pytest
from PIL import Image

from hordelib.horde import HordeLib

from .testing_shared_functions import check_single_inference_image_similarity

SIGMA_GENERATOR_SCHEDULES = ["align_your_steps", "gits"]

_PROMPT = (
    "a woman closeup made out of metal, (cyborg:1.1), realistic skin, (detailed wire:1.3), "
    "(intricate details), hdr, (intricate details, hyperdetailed:1.2), cinematic shot, "
    "vignette, centered"
)


def _payload(model_name: str, scheduler: str, resolution: int) -> dict:
    return {
        "sampler_name": "k_euler",
        "cfg_scale": 6.5,
        "denoising_strength": 1.0,
        "seed": 3688490319,
        "height": resolution,
        "width": resolution,
        "karras": False,
        "scheduler": scheduler,
        "tiling": False,
        "hires_fix": False,
        "clip_skip": 1,
        "control_type": None,
        "image_is_control": False,
        "return_control_map": False,
        "prompt": _PROMPT,
        "ddim_steps": 20,
        "n_iter": 1,
        "model": model_name,
    }


class TestSigmaGeneratorSchedules:
    @pytest.mark.default_sd15_model
    @pytest.mark.parametrize("scheduler", SIGMA_GENERATOR_SCHEDULES)
    def test_sd15(
        self,
        stable_diffusion_model_name_for_testing: str,
        hordelib_instance: HordeLib,
        scheduler: str,
    ):
        pil_image = hordelib_instance.basic_inference_single_image(
            _payload(stable_diffusion_model_name_for_testing, scheduler, 512),
        ).image
        assert pil_image is not None
        assert isinstance(pil_image, Image.Image)

        img_filename = f"schedule_20_steps_sd15_{scheduler}.png"
        pil_image.save(f"images/{img_filename}", quality=100)

        assert check_single_inference_image_similarity(
            f"images_expected/{img_filename}",
            pil_image,
        )

    @pytest.mark.default_sdxl_model
    @pytest.mark.parametrize("scheduler", SIGMA_GENERATOR_SCHEDULES)
    def test_sdxl(
        self,
        sdxl_1_0_base_model_name: str,
        hordelib_instance: HordeLib,
        scheduler: str,
    ):
        pil_image = hordelib_instance.basic_inference_single_image(
            _payload(sdxl_1_0_base_model_name, scheduler, 1024),
        ).image
        assert pil_image is not None
        assert isinstance(pil_image, Image.Image)

        img_filename = f"schedule_20_steps_sdxl_{scheduler}.png"
        pil_image.save(f"images/{img_filename}", quality=100)

        assert check_single_inference_image_similarity(
            f"images_expected/{img_filename}",
            pil_image,
        )
