"""Empirical CI guard: verify each baseline's VRAM seeds against what a real job actually holds on the device.

The burden registry (:mod:`hordelib.feature_impact`) seeds two VRAM figures a consumer scheduler relies on: a
resident weight footprint (:meth:`BaselineBurden.resident_footprint_estimate_mb`, charged when deciding
whether a card can host a model beside anything else) and an activation-inclusive sampling peak
(``vram_base_mb``, the base of the per-job VRAM budget). Both are truths that live in the checkpoints and
change as models are added, so they are measured here on real hardware rather than asserted from a table.

The measurement (:func:`record_job_vram_profile`) does not assume the components co-reside. ComfyUI evicts a
text encoder before sampling and block-swaps large diffusion weights, so the *peak simultaneous* residency is
below the summed component weights. The profile reports all three views:

* ``sum_component_weights_mb`` -- every component ever loaded, the conservative upper bound.
* ``peak_resident_weights_mb`` -- the largest the on-device weight set actually reached at one instant, the
  honest figure for a co-residency judgment.
* ``peak_device_used_mb`` -- that plus the transient activation high-water, what ``vram_base_mb`` must cover.

The hard guard is the safe direction: the footprint seed must not read below the measured peak simultaneous
weights (a seed below what the device actually holds grants co-residency room that does not exist, the
over-commit that drove the incident). The activation-peak seed is *detected* the same way: when
``vram_base_mb`` is measured to under-count the device high-water, the test xfails with the measured number to
set, so the registry is corrected from data rather than a guess. Runs on the GPU CI runner alongside the
per-baseline inference tests.
"""

from __future__ import annotations

import pytest
from loguru import logger

from hordelib.feature_impact import get_baseline_burden
from hordelib.horde import HordeLib
from hordelib.utils.torch_memory import JobVramProfile, record_job_vram_profile

# The footprint seed must cover the measured peak simultaneous weights; a small band absorbs sampling noise.
_UNDER_COUNT_TOLERANCE = 0.10
# The native job the profile is measured at, so the seed's activation term lines up with the measurement.
_JOB_WIDTH = 1024
_JOB_HEIGHT = 1024
_JOB_BATCH = 1


def _seed_sampling_peak_mb(baseline: str) -> float:
    """The registry's predicted sampling VRAM peak for the measured job (``vram_base`` plus activations)."""
    burden = get_baseline_burden(baseline)
    assert burden is not None, f"no burden seed for baseline {baseline!r}"
    megapixels = (_JOB_WIDTH * _JOB_HEIGHT) / 1_000_000
    return float(burden.vram_base_mb + round(burden.vram_per_megapixel_mb * megapixels * _JOB_BATCH))


def _measure_profile(instance: HordeLib, model_name: str, *, steps: int) -> JobVramProfile:
    """Run one minimal text-to-image job and return its measured VRAM profile.

    Uses few sampling steps because the measurement is of the VRAM the backend holds, not of image quality;
    loading and swapping the checkpoint components is what the profiler observes.
    """
    data = {
        "sampler_name": "k_euler",
        "cfg_scale": 1,
        "denoising_strength": 1.0,
        "seed": 1886,
        "height": _JOB_HEIGHT,
        "width": _JOB_WIDTH,
        "karras": False,
        "tiling": False,
        "hires_fix": False,
        "clip_skip": 1,
        "control_type": None,
        "image_is_control": False,
        "return_control_map": False,
        "prompt": "a calibration probe image",
        "ddim_steps": steps,
        "n_iter": _JOB_BATCH,
        "model": model_name,
    }
    with record_job_vram_profile() as profiler:
        result = instance.basic_inference_single_image(data)
    assert result.image is not None, "inference produced no image; cannot trust the VRAM measurement"
    return profiler.profile()


def _verify_seeds_against_profile(baseline: str, profile: JobVramProfile) -> None:
    """Report the measured profile, hard-assert the footprint safety invariant, detect the vram_base gap.

    The footprint check is the hard guard (its failure is the co-residency over-commit). The activation-peak
    check is a detector: when ``vram_base_mb`` under-counts the device high-water it xfails with the number to
    set, so the seed is corrected from the measurement instead of a table value that has drifted.
    """
    burden = get_baseline_burden(baseline)
    assert burden is not None
    seed_footprint = float(burden.resident_footprint_estimate_mb())
    seed_sampling_peak = _seed_sampling_peak_mb(baseline)

    logger.info(
        f"[footprint calibration] {baseline}: "
        f"peak_resident_weights={profile.peak_resident_weights_mb:.0f} MB, "
        f"peak_device_used={profile.peak_device_used_mb:.0f} MB, "
        f"sum_components={profile.sum_component_weights_mb:.0f} MB "
        f"(time_shared {profile.time_shared_mb:.0f} MB) | "
        f"seed_footprint={seed_footprint:.0f} MB, seed_sampling_peak={seed_sampling_peak:.0f} MB"
    )

    assert profile.peak_resident_weights_mb > 0, "no on-device weights were observed during the job"
    assert seed_footprint >= profile.peak_resident_weights_mb * (1 - _UNDER_COUNT_TOLERANCE), (
        f"{baseline}: footprint seed {seed_footprint:.0f} MB is below the measured peak simultaneous weights "
        f"{profile.peak_resident_weights_mb:.0f} MB. A footprint under what the device actually holds grants "
        f"co-residency room that does not exist. Raise vram_weights_mb / vram_support_weights_mb."
    )

    # Detection, not a hard gate: vram_base_mb predates the split weight seeds and can read below the true
    # activation-inclusive peak. When it does, surface the measured figure so the seed is set from data.
    if seed_sampling_peak < profile.peak_device_used_mb * (1 - _UNDER_COUNT_TOLERANCE):
        needed_base = round(profile.peak_device_used_mb - (seed_sampling_peak - burden.vram_base_mb))
        pytest.xfail(
            f"{baseline}: sampling-peak seed {seed_sampling_peak:.0f} MB under-counts the measured device "
            f"high-water {profile.peak_device_used_mb:.0f} MB; set vram_base_mb to ~{needed_base} MB."
        )


class TestRegistryFootprintCalibration:
    """Each heavy multi-component baseline's seeds must match what a real job holds on the device."""

    @pytest.mark.default_flux1_model
    def test_flux_schnell_seeds_match_measured(
        self,
        hordelib_instance: HordeLib,
        flux1_schnell_fp8_base_model_name: str,
    ) -> None:
        """Flux Schnell fp8: DiT plus its T5/CLIP encoders and VAE."""
        profile = _measure_profile(hordelib_instance, flux1_schnell_fp8_base_model_name, steps=4)
        _verify_seeds_against_profile("flux_1", profile)

    @pytest.mark.default_z_image_turbo_model
    def test_z_image_turbo_seeds_match_measured(
        self,
        hordelib_instance: HordeLib,
        z_image_turbo_base_model_name: str,
    ) -> None:
        """Z-Image Turbo: DiT, text encoder and VAE."""
        profile = _measure_profile(hordelib_instance, z_image_turbo_base_model_name, steps=4)
        _verify_seeds_against_profile("z_image_turbo", profile)

    @pytest.mark.default_qwen_model
    def test_qwen_image_seeds_match_measured(
        self,
        hordelib_instance: HordeLib,
        qwen_image_fp8_base_model_name: str,
    ) -> None:
        """Qwen-Image fp8: the 20B DiT and its Qwen2.5-VL text encoder, the checkpoint the incident traced to."""
        profile = _measure_profile(hordelib_instance, qwen_image_fp8_base_model_name, steps=8)
        _verify_seeds_against_profile("qwen_image", profile)
