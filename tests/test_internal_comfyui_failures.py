import pytest

from hordelib.horde import HordeLib
from hordelib.shared_model_manager import SharedModelManager


def test_lora_failure(
    shared_model_manager: type[SharedModelManager],
    hordelib_instance: HordeLib,
    stable_diffusion_model_name_for_testing: str,
    lora_GlowingRunesAI: str,
):
    # If this test fails (does NOT raise an exception from `basic_inference_single_image`)
    # then we're not detecting exceptions originating from within comfyui. This can (and has)
    # led to duplicate images being returned to the user.

    # See PRs #144, #143, #136 for more info.

    import os

    os.environ["FAILURE_TEST"] = "1"

    try:
        assert shared_model_manager.manager.lora

        # Blue, fuzzy search on version
        trigger = shared_model_manager.manager.lora.find_lora_trigger(lora_GlowingRunesAI, "blue")
        data = {
            "sampler_name": "k_euler",
            "cfg_scale": 8.0,
            "denoising_strength": 1.0,
            "seed": 304886399544324,
            "height": 512,
            "width": 512,
            "karras": False,
            "tiling": False,
            "hires_fix": False,
            "clip_skip": 1,
            "control_type": None,
            "image_is_control": False,
            "return_control_map": False,
            "prompt": f"a dark magical crystal, {trigger}, 8K resolution###blurry, out of focus",
            "loras": [{"name": lora_GlowingRunesAI, "model": 1.0, "clip": 1.0}],
            "ddim_steps": 20,
            "n_iter": 1,
            "model": stable_diffusion_model_name_for_testing,
        }

        with pytest.raises(RuntimeError, match="Pipeline failed to run"):
            hordelib_instance.basic_inference_single_image(data)
    finally:
        del os.environ["FAILURE_TEST"]
        assert os.getenv("FAILURE_TEST", False) is False
