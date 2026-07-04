"""The post-processing seed estimates against the measured cost envelope.

``hordelib.feature_impact`` ships seed estimates that schedulers use to admit post-processing work
against free VRAM. These tests pin the *shape* and rough magnitude of the real costs, measured with
:mod:`hordelib.profiling` (one ``post_process`` call per op, models resident, device-wide peak
sampled):

- ESRGAN-family upscaler peaks are flat with respect to input size (the backend tiles the
  activation): roughly 3.2 to 3.9 GB for the 4x family and under 1 GB for 2x, at any generation
  resolution. Wall time, not VRAM, scales with output megapixels.
- Face-fixer peaks scale with *input* megapixels (whole-image face detection) over a substantial
  fixed base. The heavier CodeFormer measures about 2 GB at 0.25 MP rising to about 3.9 GB at 2.4 MP
  (roughly 0.8 GB per megapixel); GFPGAN is lighter and nearly flat. Chained after a 4x upscale the
  input is 16x the generation megapixels, which is what makes large-generation chains expensive.
- ``strip_background`` peaks near 1.1 GB regardless of size.

The seeds are calibrated to this envelope: the upscale peak is charged per ``factor**2`` (flat with
generation size) and the face-fix peak per input megapixel. Re-run ``python -m hordelib.profiling`` on
a target card to regenerate the measured reference before adjusting the seeds.
"""

from __future__ import annotations

from hordelib.feature_impact import FEATURE_KIND, estimate_job_burden


def _post_processing_estimate_mb(
    *,
    width: int,
    features: list[FEATURE_KIND],
    upscale_factor: float = 1.0,
) -> int:
    burden = estimate_job_burden(
        baseline="stable_diffusion_xl",
        width=width,
        height=width,
        features=features,
        post_processing_upscale_factor=upscale_factor,
    )
    return burden.vram_post_processing_mb


class TestUpscalerEnvelope:
    """Upscaler peaks are flat with generation size and land in the measured band."""

    def test_4x_peak_is_flat_across_generation_sizes(self) -> None:
        """A 4x upscale of a 1536-square generation peaks about the same as one of a 512 square."""
        small = _post_processing_estimate_mb(
            width=512,
            features=[FEATURE_KIND.post_processing_upscale],
            upscale_factor=4.0,
        )
        large = _post_processing_estimate_mb(
            width=1536,
            features=[FEATURE_KIND.post_processing_upscale],
            upscale_factor=4.0,
        )
        assert large <= small * 1.5, f"4x estimate grew {large / small:.1f}x from 512 to 1536 input"

    def test_4x_peak_magnitude_at_one_megapixel(self) -> None:
        """The 4x estimate for a 1024-square generation sits inside the measured band plus margin."""
        estimate = _post_processing_estimate_mb(
            width=1024,
            features=[FEATURE_KIND.post_processing_upscale],
            upscale_factor=4.0,
        )
        assert 1500 <= estimate <= 5500, f"4x estimate {estimate} MB is outside the measured band"

    def test_2x_estimate_is_well_below_4x(self) -> None:
        """A 2x upscale is estimated at well under the 4x cost, matching the measured ratio."""
        two_x = _post_processing_estimate_mb(
            width=1024,
            features=[FEATURE_KIND.post_processing_upscale],
            upscale_factor=2.0,
        )
        four_x = _post_processing_estimate_mb(
            width=1024,
            features=[FEATURE_KIND.post_processing_upscale],
            upscale_factor=4.0,
        )
        assert two_x < four_x


class TestFacefixerEnvelope:
    """Face-fixer peaks scale with input megapixels; the flat seed misses both ends."""

    def test_facefix_peak_scales_with_input_megapixels(self) -> None:
        """A face-fix over a 1536-square input is estimated meaningfully above one over 512 square.

        The measured 512->1536 ratio is roughly 1.6-1.8x, not larger: the peak scales with input
        megapixels but over a substantial fixed base, so the growth is real but sub-linear in the ratio.
        """
        small = _post_processing_estimate_mb(width=512, features=[FEATURE_KIND.post_processing_facefix])
        large = _post_processing_estimate_mb(width=1536, features=[FEATURE_KIND.post_processing_facefix])
        assert large >= small * 1.5, f"face-fix estimate is too flat ({small} vs {large} MB)"

    def test_facefix_estimate_is_positive_and_bounded(self) -> None:
        """The face-fix estimate is a positive figure below any whole-card total."""
        estimate = _post_processing_estimate_mb(width=1024, features=[FEATURE_KIND.post_processing_facefix])
        assert 0 < estimate < 24_000


class TestChainAndPhaseAccounting:
    """Composition invariants any reseed must preserve."""

    def test_strip_background_magnitude(self) -> None:
        """The strip-background estimate sits inside the measured band (about 1.1 GB, size-flat)."""
        estimate = _post_processing_estimate_mb(width=1024, features=[FEATURE_KIND.strip_background])
        assert 500 <= estimate <= 2500

    def test_phase_split_sums_to_total(self) -> None:
        """The sampling and post-processing phase figures always sum to the combined total."""
        burden = estimate_job_burden(
            baseline="stable_diffusion_xl",
            width=1024,
            height=1024,
            features=[FEATURE_KIND.post_processing_upscale, FEATURE_KIND.post_processing_facefix],
            post_processing_upscale_factor=4.0,
        )
        assert burden.vram_mb == burden.vram_sampling_mb + burden.vram_post_processing_mb
