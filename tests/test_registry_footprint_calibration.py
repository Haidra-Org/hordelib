"""Empirical CI guard: verify each baseline's resident-footprint seed against what ComfyUI actually loads.

The burden registry (:mod:`hordelib.feature_impact`) seeds a resident weight footprint per baseline, and a
consumer scheduler charges :meth:`BaselineBurden.resident_footprint_estimate_mb` when deciding whether a
card can host a model beside anything else. A seed that reads *smaller* than the weights the backend
force-loads grants co-residency room that does not exist: on a full card the model is then admitted beside a
sibling and runs out of memory mid-sample. That is a data drift the unit tests cannot catch, because the
truth lives in the checkpoints, so it is measured here on real hardware.

The measurement records every component ComfyUI loads to the device during one real job
(:func:`record_resident_footprint`), so a text encoder evicted before VAE decode is still counted, and sums
their weights. The guard is deliberately on the *total* footprint, not the core/support split: a baseline may
model its encoder either as a separate ``vram_support_weights_mb`` (Flux) or folded into ``vram_weights_mb``
(Z-Image); both are correct as long as the total reflects the full resident set. Under-counting the total is
the failure that drove the incident and is the hard assertion; a conservative over-count is tolerated with
headroom.

These run on the GPU CI runner alongside the per-baseline inference tests. A new heavy multi-component
baseline added to the registry without accounting for its text encoder fails here, which is the drift the
Qwen-Image seed (DiT weights only, no text-encoder mass) is the first instance of.
"""

from __future__ import annotations

import pytest

from hordelib.feature_impact import get_baseline_burden
from hordelib.horde import HordeLib
from hordelib.utils.torch_memory import ResidentFootprint, record_resident_footprint

# A seed may sit at or above the measured resident weights (conservative is safe); it must not read below
# them, which is the co-residency-room over-grant that causes the out-of-memory sample. A small band absorbs
# measurement/dtype noise. The upper bound catches a gross over-count that would reserve the card needlessly.
_UNDER_COUNT_TOLERANCE = 0.10
_OVER_COUNT_TOLERANCE = 0.50


def _assert_footprint_seed_matches(baseline: str, measured: ResidentFootprint) -> None:
    """Assert the registry footprint for ``baseline`` reflects the measured resident weight set."""
    burden = get_baseline_burden(baseline)
    assert burden is not None, f"no burden seed for baseline {baseline!r}"
    seed_total = float(burden.resident_footprint_estimate_mb())

    assert measured.total_mb > 0, "no model components were recorded during the job"
    assert seed_total >= measured.total_mb * (1 - _UNDER_COUNT_TOLERANCE), (
        f"{baseline}: footprint seed {seed_total:.0f} MB under-counts the measured resident weights "
        f"{measured.total_mb:.0f} MB (core {measured.core_mb:.0f} + support {measured.support_mb:.0f}). "
        f"A footprint below the force-loaded weights grants co-residency room that does not exist. "
        f"Raise vram_weights_mb and/or vram_support_weights_mb for this baseline."
    )
    assert seed_total <= measured.total_mb * (1 + _OVER_COUNT_TOLERANCE), (
        f"{baseline}: footprint seed {seed_total:.0f} MB far exceeds the measured resident weights "
        f"{measured.total_mb:.0f} MB; a large over-count reserves the card for a model that need not."
    )


def _measure_text_to_image_footprint(instance: HordeLib, model_name: str, *, steps: int) -> ResidentFootprint:
    """Run one minimal text-to-image job and return the measured resident weight footprint.

    Uses few sampling steps because the measurement is of the weights the backend force-loads, not of image
    quality; loading the checkpoint components is what the recorder observes.
    """
    data = {
        "sampler_name": "k_euler",
        "cfg_scale": 1,
        "denoising_strength": 1.0,
        "seed": 1886,
        "height": 1024,
        "width": 1024,
        "karras": False,
        "tiling": False,
        "hires_fix": False,
        "clip_skip": 1,
        "control_type": None,
        "image_is_control": False,
        "return_control_map": False,
        "prompt": "a calibration probe image",
        "ddim_steps": steps,
        "n_iter": 1,
        "model": model_name,
    }
    with record_resident_footprint() as recorder:
        result = instance.basic_inference_single_image(data)
    assert result.image is not None, "inference produced no image; cannot trust the footprint measurement"
    return recorder.resident_footprint()


class TestRegistryFootprintCalibration:
    """Each heavy multi-component baseline's seed must not under-count its real resident weight set."""

    @pytest.mark.default_flux1_model
    def test_flux_schnell_footprint_matches_registry(
        self,
        hordelib_instance: HordeLib,
        flux1_schnell_fp8_base_model_name: str,
    ) -> None:
        """Flux Schnell fp8: DiT plus its T5/CLIP encoders and VAE, seeded with a support figure."""
        measured = _measure_text_to_image_footprint(hordelib_instance, flux1_schnell_fp8_base_model_name, steps=4)
        _assert_footprint_seed_matches("flux_1", measured)

    @pytest.mark.default_z_image_turbo_model
    def test_z_image_turbo_footprint_matches_registry(
        self,
        hordelib_instance: HordeLib,
        z_image_turbo_base_model_name: str,
    ) -> None:
        """Z-Image Turbo: DiT, text encoder and VAE, folded into the core weight seed (no split)."""
        measured = _measure_text_to_image_footprint(hordelib_instance, z_image_turbo_base_model_name, steps=4)
        _assert_footprint_seed_matches("z_image_turbo", measured)

    @pytest.mark.xfail(
        strict=True,
        reason="qwen_image seed counts only the DiT weights; its Qwen2.5-VL text encoder is unseeded. "
        "Remove this marker after setting vram_support_weights_mb from this test's measured support.",
    )
    @pytest.mark.default_qwen_model
    def test_qwen_image_footprint_matches_registry(
        self,
        hordelib_instance: HordeLib,
        qwen_image_fp8_base_model_name: str,
    ) -> None:
        """Qwen-Image fp8: the reproduced under-count. The measured support here is what the seed must carry."""
        measured = _measure_text_to_image_footprint(hordelib_instance, qwen_image_fp8_base_model_name, steps=8)
        _assert_footprint_seed_matches("qwen_image", measured)
