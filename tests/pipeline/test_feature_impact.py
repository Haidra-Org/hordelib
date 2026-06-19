"""Table-driven tests for the feature-impact registry (GPU-free)."""

import pytest
from horde_model_reference.model_consts.image import KNOWN_IMAGE_GENERATION_BASELINE
from horde_model_reference.model_reference_records import ImageGenerationModelRecord

from hordelib.feature_impact import (
    FEATURE_KIND,
    FEATURE_PHASE,
    estimate_job_burden,
    get_baseline_burden,
    get_feature_impact_registry,
)


class TestRegistryCoverage:
    def test_every_known_baseline_has_an_entry(self) -> None:
        """Tripwire: a new baseline added to horde-model-reference needs a burden entry here."""
        registry = get_feature_impact_registry()
        missing = [
            baseline.value
            for baseline in KNOWN_IMAGE_GENERATION_BASELINE
            # "infer" is a directive, not a baseline; flux_1 is the canonical flux value.
            if baseline != KNOWN_IMAGE_GENERATION_BASELINE.infer and baseline.value not in registry.baselines
        ]
        assert not missing, f"Baselines without a feature-impact entry: {missing}"

    def test_every_feature_kind_has_an_entry(self) -> None:
        registry = get_feature_impact_registry()
        missing = [kind.value for kind in FEATURE_KIND if kind not in registry.features]
        assert not missing, f"Feature kinds without an impact entry: {missing}"

    def test_baseline_lookup(self) -> None:
        assert get_baseline_burden("stable_diffusion_xl") is not None
        assert get_baseline_burden("not_a_baseline") is None


class TestEstimateJobBurden:
    def test_unknown_baseline_uses_fallback_and_is_flagged(self) -> None:
        estimate = estimate_job_burden(baseline="not_a_baseline", width=512, height=512)
        assert not estimate.baseline_known
        assert estimate.vram_mb > 0
        assert estimate.disk_bytes_needed > 0

    def test_known_baseline_is_flagged_known(self) -> None:
        estimate = estimate_job_burden(baseline="stable_diffusion_1", width=512, height=512)
        assert estimate.baseline_known

    @pytest.mark.parametrize(
        ("smaller", "larger"),
        [
            # More pixels cost more VRAM
            (
                {"baseline": "stable_diffusion_1", "width": 512, "height": 512},
                {"baseline": "stable_diffusion_1", "width": 1024, "height": 1024},
            ),
            # Batching costs more VRAM
            (
                {"baseline": "stable_diffusion_xl", "width": 1024, "height": 1024, "batch": 1},
                {"baseline": "stable_diffusion_xl", "width": 1024, "height": 1024, "batch": 4},
            ),
            # Features cost more VRAM
            (
                {"baseline": "stable_diffusion_1", "width": 512, "height": 512},
                {
                    "baseline": "stable_diffusion_1",
                    "width": 512,
                    "height": 512,
                    "features": [FEATURE_KIND.controlnet],
                },
            ),
        ],
    )
    def test_vram_monotonicity(self, smaller: dict, larger: dict) -> None:
        assert estimate_job_burden(**smaller).vram_mb < estimate_job_burden(**larger).vram_mb

    def test_lora_expects_a_download(self) -> None:
        estimate = estimate_job_burden(
            baseline="stable_diffusion_1",
            width=512,
            height=512,
            features=[FEATURE_KIND.lora],
        )
        assert len(estimate.downloads_expected) == 1
        assert estimate.downloads_expected[0].typical_size_mb

    def test_hires_fix_triggers_no_download(self) -> None:
        estimate = estimate_job_burden(
            baseline="stable_diffusion_1",
            width=512,
            height=512,
            features=[FEATURE_KIND.hires_fix],
        )
        assert estimate.downloads_expected == []

    def test_controlnet_not_applied_to_flux(self) -> None:
        with_controlnet = estimate_job_burden(
            baseline="flux_1",
            width=1024,
            height=1024,
            features=[FEATURE_KIND.controlnet],
        )
        without = estimate_job_burden(baseline="flux_1", width=1024, height=1024)
        assert with_controlnet.vram_mb == without.vram_mb
        assert with_controlnet.downloads_expected == []

    def test_model_record_size_overrides_typical_disk(self) -> None:
        record = ImageGenerationModelRecord(
            name="Deliberate",
            baseline=KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_1,
            nsfw=False,
            description="test",
            size_on_disk_bytes=123_456_789,
        )
        estimate = estimate_job_burden(
            baseline="stable_diffusion_1",
            width=512,
            height=512,
            model_record=record,
        )
        assert estimate.disk_bytes_needed == 123_456_789

    def test_download_sizes_add_to_disk_needed(self) -> None:
        base = estimate_job_burden(baseline="stable_diffusion_1", width=512, height=512)
        with_lora = estimate_job_burden(
            baseline="stable_diffusion_1",
            width=512,
            height=512,
            features=[FEATURE_KIND.lora],
        )
        assert with_lora.disk_bytes_needed > base.disk_bytes_needed


class TestPhaseSplit:
    def test_total_equals_sum_of_phases(self) -> None:
        estimate = estimate_job_burden(
            baseline="stable_diffusion_xl",
            width=1024,
            height=1024,
            features=[FEATURE_KIND.controlnet, FEATURE_KIND.post_processing_upscale],
        )
        assert estimate.vram_mb == estimate.vram_sampling_mb + estimate.vram_post_processing_mb

    def test_baseline_only_has_no_post_processing_burden(self) -> None:
        estimate = estimate_job_burden(baseline="stable_diffusion_xl", width=1024, height=1024)
        assert estimate.vram_post_processing_mb == 0
        assert estimate.vram_sampling_mb == estimate.vram_mb

    def test_sampling_feature_lands_in_sampling_phase(self) -> None:
        """A controlnet (sampling-phase) feature must not leak into the post-processing bucket."""
        base = estimate_job_burden(baseline="stable_diffusion_xl", width=1024, height=1024)
        with_controlnet = estimate_job_burden(
            baseline="stable_diffusion_xl",
            width=1024,
            height=1024,
            features=[FEATURE_KIND.controlnet],
        )
        assert with_controlnet.vram_post_processing_mb == 0
        assert with_controlnet.vram_sampling_mb > base.vram_sampling_mb

    def test_upscaler_lands_in_post_processing_phase(self) -> None:
        base = estimate_job_burden(baseline="stable_diffusion_xl", width=1024, height=1024)
        with_upscale = estimate_job_burden(
            baseline="stable_diffusion_xl",
            width=1024,
            height=1024,
            features=[FEATURE_KIND.post_processing_upscale],
        )
        assert with_upscale.vram_post_processing_mb > 0
        # Sampling-phase cost is unchanged by a post-processing feature.
        assert with_upscale.vram_sampling_mb == base.vram_sampling_mb

    def test_registry_phase_tags(self) -> None:
        features = get_feature_impact_registry().features
        assert features[FEATURE_KIND.post_processing_upscale].phase == FEATURE_PHASE.post_processing
        assert features[FEATURE_KIND.post_processing_facefix].phase == FEATURE_PHASE.post_processing
        assert features[FEATURE_KIND.strip_background].phase == FEATURE_PHASE.post_processing
        assert features[FEATURE_KIND.controlnet].phase == FEATURE_PHASE.sampling
        assert features[FEATURE_KIND.lora].phase == FEATURE_PHASE.sampling


class TestAuxModelWeightOverride:
    def test_override_replaces_flat_weight_but_keeps_activation(self) -> None:
        """A supplied real upscaler weight replaces the flat delta; the per-megapixel term stays."""
        seeded = estimate_job_burden(
            baseline="stable_diffusion_xl",
            width=1024,
            height=1024,
            features=[FEATURE_KIND.post_processing_upscale],
        )
        # The seed flat weight is 2000 MB; a tiny real ESRGAN is ~70 MB, so the override lowers the peak.
        overridden = estimate_job_burden(
            baseline="stable_diffusion_xl",
            width=1024,
            height=1024,
            features=[FEATURE_KIND.post_processing_upscale],
            aux_model_weights_mb={FEATURE_KIND.post_processing_upscale: 70},
        )
        assert overridden.vram_post_processing_mb < seeded.vram_post_processing_mb
        # Activation term (per-megapixel) is still counted, so it is not merely the weight.
        assert overridden.vram_post_processing_mb > 70

    def test_override_for_absent_feature_is_ignored(self) -> None:
        base = estimate_job_burden(
            baseline="stable_diffusion_xl",
            width=1024,
            height=1024,
            features=[FEATURE_KIND.post_processing_upscale],
        )
        with_unused_override = estimate_job_burden(
            baseline="stable_diffusion_xl",
            width=1024,
            height=1024,
            features=[FEATURE_KIND.post_processing_upscale],
            aux_model_weights_mb={FEATURE_KIND.post_processing_facefix: 9999},
        )
        assert with_unused_override.vram_mb == base.vram_mb
