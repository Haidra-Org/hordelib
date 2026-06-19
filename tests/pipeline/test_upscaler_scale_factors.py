"""Tests for the upscaler scale-factor ROM table (GPU-free)."""

from hordelib.pipeline.constants import (
    UPSCALER_SCALE_FACTORS,
    max_upscale_factor,
    upscaler_scale_factor,
)


def test_known_upscaler_returns_its_factor() -> None:
    assert upscaler_scale_factor("RealESRGAN_x4plus") == 4
    assert upscaler_scale_factor("RealESRGAN_x2plus") == 2


def test_none_is_no_enlargement() -> None:
    assert upscaler_scale_factor(None) == 1


def test_unknown_upscaler_errs_high() -> None:
    """A new upscaler the ROM has not learned over-reserves (factor 4), never under-reserves (1)."""
    assert upscaler_scale_factor("BrandNewUpscaler_x4") == 4


def test_max_picks_the_largest() -> None:
    assert max_upscale_factor(["RealESRGAN_x2plus", "RealESRGAN_x4plus"]) == 4


def test_max_empty_is_one() -> None:
    assert max_upscale_factor([]) == 1
    assert max_upscale_factor([None]) == 1


def test_all_known_upscalers_have_a_factor() -> None:
    """Every KNOWN_UPSCALERS value should be in the ROM so none silently takes the unknown default."""
    from horde_sdk.generation_parameters import KNOWN_UPSCALERS

    missing = [u.value for u in KNOWN_UPSCALERS if u.value not in UPSCALER_SCALE_FACTORS]
    assert not missing, f"Upscalers missing a scale factor: {missing}"
