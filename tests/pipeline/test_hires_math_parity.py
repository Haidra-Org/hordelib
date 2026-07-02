"""Cross-library hires-fix math parity: hordelib's ImageUtils vs horde_sdk's image_utils.

The two-pass hires-fix values can be computed in two places: the SDK's AI-Horde converter
computes them when building generic parameters (which flow through the typed path as payload
overrides), and hordelib recomputes them during materialization for payloads without
overrides (the legacy dict path). Both implementations must agree, or the same job would
render differently depending on which path carried it. This grid pins that agreement.

hordelib twin: hordelib.utils.image_utils.ImageUtils
horde_sdk twin: horde_sdk.utils.image_utils
"""

import pytest
from horde_model_reference.meta_consts import KNOWN_IMAGE_GENERATION_BASELINE, get_baseline_native_resolution
from horde_sdk.utils import image_utils as sdk_image_utils

from hordelib.utils.image_utils import ImageUtils

RESOLUTION_GRID = [
    (512, 512),
    (768, 512),
    (1024, 1024),
    (1216, 832),
    (1536, 1024),
    (2048, 1152),
    (3072, 2048),
]

BASELINE_GRID = [
    KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_1,
    KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_xl,
    KNOWN_IMAGE_GENERATION_BASELINE.stable_cascade,
]


@pytest.mark.parametrize(("width", "height"), RESOLUTION_GRID)
@pytest.mark.parametrize("baseline", BASELINE_GRID)
def test_first_pass_resolution_parity(baseline, width, height) -> None:
    """Both libraries derive the same first-pass resolution for every baseline strategy."""
    sdk_result = sdk_image_utils.get_first_pass_image_resolution_by_baseline(
        width=width,
        height=height,
        baseline=baseline,
    )

    if baseline is KNOWN_IMAGE_GENERATION_BASELINE.stable_cascade:
        hordelib_result = ImageUtils.get_first_pass_image_resolution_max(width, height)
    elif baseline is KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_xl:
        hordelib_result = ImageUtils.get_first_pass_image_resolution_sdxl(width, height)
    else:
        hordelib_result = ImageUtils.get_first_pass_image_resolution_min(width, height)

    assert sdk_result == hordelib_result


@pytest.mark.parametrize(("width", "height"), RESOLUTION_GRID)
@pytest.mark.parametrize("baseline", BASELINE_GRID)
@pytest.mark.parametrize("denoise", [0.3, 0.65, 1.0])
@pytest.mark.parametrize("steps", [10, 30, 100])
def test_second_pass_steps_parity(baseline, width, height, denoise, steps) -> None:
    """Both libraries derive the same second-pass step count over the parameter grid."""
    native_resolution = get_baseline_native_resolution(baseline)

    sdk_steps = sdk_image_utils.calc_upscale_sampler_steps(
        model_native_resolution=native_resolution,
        width=width,
        height=height,
        hires_fix_denoising_strength=denoise,
        ddim_steps=steps,
    )
    hordelib_steps = ImageUtils.calc_upscale_sampler_steps(
        model_native_resolution=native_resolution,
        width=width,
        height=height,
        hires_fix_denoising_strength=denoise,
        ddim_steps=steps,
    )

    assert sdk_steps == hordelib_steps
