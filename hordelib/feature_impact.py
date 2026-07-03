"""Declarative expected resource burdens for baselines and inference features.

Operators and the worker's benchmark harness need to reason about a job's resource
demands *before* running it: does an SDXL + controlnet job fit this GPU, does enabling
loras imply ad-hoc downloads, how much disk does a flux tier need. This module is the
single registry for those expectations: per-baseline base burdens, per-feature deltas,
and download-trigger flags.

The static seed values below are deliberately conservative estimates, not measurements.
They exist to make pre-flight decisions (skip a tier, warn about VRAM) directionally
correct; benchmark runs emit :class:`CalibrationSample` records in exactly this shape so
measured medians can later override the seeds (merge implementation deferred to the
benchmark calibration phase).

This module is comfy-free and import-safe before ``hordelib.initialise()``; it is part
of the public API surface re-exported from :mod:`hordelib.api`.
"""

from enum import StrEnum, auto

from horde_model_reference.model_reference_records import ImageGenerationModelRecord
from pydantic import BaseModel

_BYTES_PER_GB = 1024**3


class FEATURE_KIND(StrEnum):
    """Inference features with a non-trivial resource or download impact."""

    lora = auto()
    ti = auto()
    controlnet = auto()
    hires_fix = auto()
    img2img = auto()
    post_processing_upscale = auto()
    post_processing_facefix = auto()
    strip_background = auto()


class FEATURE_PHASE(StrEnum):
    """When in a job's lifecycle a feature's VRAM is consumed.

    A consumer that schedules several jobs on one device needs this distinction: the ``sampling`` cost
    is co-resident with the checkpoint while it denoises, whereas a ``post_processing`` cost (an upscaler
    or face-fixer) is claimed *after* sampling completes and the checkpoint may already have been evicted.
    Summing both into a single per-job figure hides that the post-processing peaks of distinct in-flight
    jobs can align in time and over-commit the card, so the estimate is reported per phase.
    """

    sampling = auto()
    post_processing = auto()


class DownloadTrigger(BaseModel):
    """Whether using a feature/model can trigger an ad-hoc network download."""

    triggered: bool
    typical_size_mb: int | None = None
    """Typical single-download size, for bandwidth/disk planning (None when unknown)."""


class FeatureImpact(BaseModel):
    """Expected resource deltas of enabling one feature on top of a baseline."""

    feature: FEATURE_KIND
    vram_delta_mb: int
    vram_delta_per_megapixel_mb: int = 0
    """Activation-scaling component, applied per output megapixel."""
    ram_delta_mb: int
    download: DownloadTrigger
    phase: FEATURE_PHASE = FEATURE_PHASE.sampling
    """The job phase this feature's VRAM is consumed in; drives the per-phase split of the estimate."""
    applies_to_baselines: list[str] | None = None
    """Baseline values this feature is available for (None = all)."""
    notes: str = ""


class BaselineBurden(BaseModel):
    """Expected base resource burden of running one baseline at native resolution."""

    baseline: str
    """A ``KNOWN_IMAGE_GENERATION_BASELINE`` value."""
    vram_base_mb: int
    ram_base_mb: int
    vram_per_megapixel_mb: int
    """Additional VRAM per output megapixel beyond the native resolution, per batch image."""
    native_resolution: tuple[int, int]
    min_recommended_vram_mb: int
    max_recommended_batch: int
    typical_disk_gb: float
    """Typical checkpoint size on disk, used when no model record is available."""
    vram_weights_mb: int = 0
    """Resident weight footprint (MB) at the baseline's typical worker dtype, distinct from activations.

    This is the figure ComfyUI compares against its weight budget when deciding whether to keep weights
    resident or stream them from host RAM, so it (not the activation-inflated ``vram_base_mb``) is what a
    consumer should use to forecast streaming and whole-card residency. Zero means unspecified, in which
    case :meth:`resident_weight_estimate_mb` falls back to ``vram_base_mb``. The seeded figures reflect the
    dtypes the worker typically runs on consumer cards (for example, a Flux fp8 checkpoint)."""

    vram_support_weights_mb: int = 0
    """Resident weight footprint (MB) of the support components loaded alongside the core weights.

    Text encoders and the VAE are force-loaded onto the device over the course of every job, so a job's
    true resident demand is the core weights plus these. They are tracked separately because streaming
    decisions apply to the core (diffusion) weights only, while room and residency verdicts must charge
    the whole set: judging card fit by the core weights alone under-counts a multi-component checkpoint
    by the full text-encoder size. Zero means unmeasured (the footprint falls back to the core figure)."""

    def resident_weight_estimate_mb(self) -> int:
        """Return the core (diffusion) resident weight footprint (MB), falling back when unseeded."""
        return self.vram_weights_mb or self.vram_base_mb

    def resident_footprint_estimate_mb(self) -> int:
        """Return the full per-job resident weight footprint (MB): core weights plus support components.

        This is the figure a scheduler should charge when judging whether a card can host this model
        beside anything else; every component in it is force-loaded to the device at some point in each
        job, so room granted against a smaller number is room that does not exist.
        """
        return self.resident_weight_estimate_mb() + self.vram_support_weights_mb


class BurdenEstimate(BaseModel):
    """The estimated total burden of one job configuration."""

    vram_mb: int
    """Total peak VRAM (``vram_sampling_mb + vram_post_processing_mb``); kept for back-compat callers."""
    vram_sampling_mb: int = 0
    """VRAM resident during sampling: the baseline plus every sampling-phase feature delta.

    Defaults to 0 so a ``BurdenEstimate`` deserialized from an older producer is still constructible; live
    estimates from :func:`estimate_job_burden` always populate it."""
    vram_post_processing_mb: int = 0
    """Marginal VRAM peak of the post-processing phase (upscalers/face-fixers), claimed after sampling.

    This is the figure a scheduler reserves against concurrent dispatch, since it lands while the slot has
    already been released for the next job. Zero when the job has no post-processing features."""
    ram_mb: int
    disk_bytes_needed: int
    downloads_expected: list[DownloadTrigger]
    baseline_known: bool
    """False when the baseline had no registry entry and the fallback seed was used."""


class CalibrationSample(BaseModel):
    """One measured data point, as emitted by benchmark runs, for overriding seeds later."""

    baseline: str
    features: list[FEATURE_KIND]
    width: int
    height: int
    batch: int
    observed_vram_high_water_mb: int
    observed_ram_high_water_mb: int
    observed_its: float
    sample_count: int
    recorded_at: str
    gpu_name: str


class FeatureImpactRegistry(BaseModel):
    """The full set of baseline burdens and feature impacts."""

    baselines: dict[str, BaselineBurden]
    features: dict[FEATURE_KIND, FeatureImpact]
    calibration: list[CalibrationSample] = []
    """Measured samples that future calibration merging will fold over the static seeds."""


# Conservative static seeds; see module docstring. VRAM figures assume fp16 weights and
# include typical activation overhead at native resolution.
_BASELINE_SEEDS: list[BaselineBurden] = [
    BaselineBurden(
        baseline="stable_diffusion_1",
        vram_base_mb=3200,
        ram_base_mb=6000,
        vram_per_megapixel_mb=900,
        native_resolution=(512, 512),
        min_recommended_vram_mb=4000,
        max_recommended_batch=8,
        typical_disk_gb=2.1,
    ),
    BaselineBurden(
        baseline="stable_diffusion_2_512",
        vram_base_mb=3400,
        ram_base_mb=6500,
        vram_per_megapixel_mb=900,
        native_resolution=(512, 512),
        min_recommended_vram_mb=4000,
        max_recommended_batch=8,
        typical_disk_gb=2.5,
    ),
    BaselineBurden(
        baseline="stable_diffusion_2_768",
        vram_base_mb=3800,
        ram_base_mb=7000,
        vram_per_megapixel_mb=900,
        native_resolution=(768, 768),
        min_recommended_vram_mb=5000,
        max_recommended_batch=6,
        typical_disk_gb=2.5,
    ),
    BaselineBurden(
        baseline="stable_diffusion_xl",
        vram_base_mb=7000,
        ram_base_mb=12000,
        vram_per_megapixel_mb=1200,
        native_resolution=(1024, 1024),
        min_recommended_vram_mb=8000,
        max_recommended_batch=4,
        typical_disk_gb=6.9,
        vram_weights_mb=4900,
        vram_support_weights_mb=1700,
    ),
    BaselineBurden(
        baseline="stable_cascade",
        vram_base_mb=9000,
        ram_base_mb=14000,
        vram_per_megapixel_mb=1200,
        native_resolution=(1024, 1024),
        min_recommended_vram_mb=10000,
        max_recommended_batch=2,
        typical_disk_gb=9.0,
        vram_weights_mb=8000,
    ),
    BaselineBurden(
        baseline="flux_1",
        vram_base_mb=13000,
        ram_base_mb=24000,
        vram_per_megapixel_mb=1500,
        native_resolution=(1024, 1024),
        min_recommended_vram_mb=14000,
        max_recommended_batch=1,
        typical_disk_gb=17.0,
        vram_weights_mb=11500,
        vram_support_weights_mb=4900,
    ),
    BaselineBurden(
        baseline="flux_schnell",
        vram_base_mb=13000,
        ram_base_mb=24000,
        vram_per_megapixel_mb=1500,
        native_resolution=(1024, 1024),
        min_recommended_vram_mb=14000,
        max_recommended_batch=1,
        typical_disk_gb=17.0,
        vram_weights_mb=11500,
        vram_support_weights_mb=4900,
    ),
    BaselineBurden(
        baseline="flux_dev",
        vram_base_mb=13000,
        ram_base_mb=24000,
        vram_per_megapixel_mb=1500,
        native_resolution=(1024, 1024),
        min_recommended_vram_mb=14000,
        max_recommended_batch=1,
        typical_disk_gb=17.0,
        vram_weights_mb=11500,
        vram_support_weights_mb=4900,
    ),
    BaselineBurden(
        baseline="qwen_image",
        vram_base_mb=14000,
        ram_base_mb=28000,
        vram_per_megapixel_mb=1500,
        native_resolution=(1024, 1024),
        min_recommended_vram_mb=16000,
        max_recommended_batch=1,
        typical_disk_gb=20.0,
        # The 20B DiT and the Qwen2.5-VL text encoder are both force-loaded every job. A prior estimate of
        # 12000 charged only part of the DiT and none of the encoder, so the whole set read as co-residable on
        # a 24GB card and was admitted beside a sibling that then ran it out of memory mid-sample. These figures
        # are the weights ComfyUI actually loads (measured by test_registry_footprint_calibration): the core DiT
        # alone exceeds a 24GB card's usable VRAM, so the forecast correctly treats it as whole-card/streaming.
        vram_weights_mb=19500,
        vram_support_weights_mb=8200,
    ),
    BaselineBurden(
        baseline="z_image_turbo",
        vram_base_mb=11000,
        ram_base_mb=14000,
        vram_per_megapixel_mb=1200,
        native_resolution=(1024, 1024),
        min_recommended_vram_mb=12000,
        max_recommended_batch=2,
        typical_disk_gb=8.0,
        # The DiT plus its text encoder and VAE stay resident together. The prior folded estimate of 10000
        # charged roughly half the real set, so the forecast read it as comfortably co-resident and never gave
        # Z-Image the card it needs. Split from the measured weights (test_registry_footprint_calibration): the
        # core diffusion weights and the support components each carried separately.
        vram_weights_mb=11800,
        vram_support_weights_mb=7900,
    ),
]

_FALLBACK_BASELINE_BURDEN = BaselineBurden(
    baseline="unknown",
    vram_base_mb=8000,
    ram_base_mb=14000,
    vram_per_megapixel_mb=1200,
    native_resolution=(1024, 1024),
    min_recommended_vram_mb=10000,
    max_recommended_batch=1,
    typical_disk_gb=8.0,
    vram_weights_mb=7000,
)
"""Used for baselines with no registry entry (including ``infer``), erring on the heavy side."""

_SDXL_AND_EARLIER = [
    "stable_diffusion_1",
    "stable_diffusion_2_512",
    "stable_diffusion_2_768",
    "stable_diffusion_xl",
]

_FEATURE_SEEDS: list[FeatureImpact] = [
    FeatureImpact(
        feature=FEATURE_KIND.lora,
        vram_delta_mb=400,
        ram_delta_mb=600,
        download=DownloadTrigger(triggered=True, typical_size_mb=150),
        notes="Ad-hoc loras download from CivitAI on first use; SDXL loras are often 300MB+.",
    ),
    FeatureImpact(
        feature=FEATURE_KIND.ti,
        vram_delta_mb=50,
        ram_delta_mb=50,
        download=DownloadTrigger(triggered=True, typical_size_mb=1),
        notes="Textual inversion embeddings are tiny; the download trigger matters, not the size.",
    ),
    FeatureImpact(
        feature=FEATURE_KIND.controlnet,
        vram_delta_mb=1800,
        vram_delta_per_megapixel_mb=300,
        ram_delta_mb=3000,
        download=DownloadTrigger(triggered=True, typical_size_mb=1500),
        applies_to_baselines=_SDXL_AND_EARLIER,
        notes="Includes the annotator/preprocessor models; SD15 controlnets ~700MB, SDXL ~2.5GB.",
    ),
    FeatureImpact(
        feature=FEATURE_KIND.hires_fix,
        vram_delta_mb=500,
        vram_delta_per_megapixel_mb=900,
        ram_delta_mb=500,
        download=DownloadTrigger(triggered=False),
        notes="Second sampling pass at the upscaled resolution dominates the cost.",
    ),
    FeatureImpact(
        feature=FEATURE_KIND.img2img,
        vram_delta_mb=300,
        ram_delta_mb=300,
        download=DownloadTrigger(triggered=False),
        notes="Extra VAE encode of the source image.",
    ),
    FeatureImpact(
        feature=FEATURE_KIND.post_processing_upscale,
        vram_delta_mb=2000,
        vram_delta_per_megapixel_mb=300,
        ram_delta_mb=2000,
        download=DownloadTrigger(triggered=True, typical_size_mb=70),
        phase=FEATURE_PHASE.post_processing,
        notes="ESRGAN-family upscalers; model is small but activations on large outputs are not.",
    ),
    FeatureImpact(
        feature=FEATURE_KIND.post_processing_facefix,
        vram_delta_mb=1500,
        ram_delta_mb=2000,
        download=DownloadTrigger(triggered=True, typical_size_mb=350),
        phase=FEATURE_PHASE.post_processing,
        notes="GFPGAN/CodeFormer weights.",
    ),
    FeatureImpact(
        feature=FEATURE_KIND.strip_background,
        vram_delta_mb=1000,
        ram_delta_mb=1500,
        download=DownloadTrigger(triggered=True, typical_size_mb=170),
        phase=FEATURE_PHASE.post_processing,
    ),
]

_REGISTRY = FeatureImpactRegistry(
    baselines={burden.baseline: burden for burden in _BASELINE_SEEDS},
    features={impact.feature: impact for impact in _FEATURE_SEEDS},
)


def get_feature_impact_registry() -> FeatureImpactRegistry:
    """Return the registry of baseline burdens and feature impacts."""
    return _REGISTRY


def get_baseline_burden(baseline: str) -> BaselineBurden | None:
    """Return the burden entry for *baseline*, or None when it has no registry entry."""
    return _REGISTRY.baselines.get(baseline)


def estimate_job_burden(
    *,
    baseline: str,
    width: int,
    height: int,
    batch: int = 1,
    features: list[FEATURE_KIND] | None = None,
    model_record: ImageGenerationModelRecord | None = None,
    aux_model_weights_mb: dict[FEATURE_KIND, float] | None = None,
    post_processing_upscale_factor: float = 1.0,
) -> BurdenEstimate:
    """Estimate the total resource burden of one job configuration.

    The VRAM total is reported both as a single ``vram_mb`` (back-compat) and split by
    :class:`FEATURE_PHASE`: ``vram_sampling_mb`` (baseline plus sampling-phase features) and
    ``vram_post_processing_mb`` (the marginal upscaler/face-fixer peak that lands after sampling). A
    scheduler reserves the post-processing figure against concurrent dispatch because it is claimed once
    the inference slot has already been released for the next job.

    ``aux_model_weights_mb`` lets a caller that has resolved a specific auxiliary model (e.g. a particular
    ESRGAN upscaler or controlnet) supply its real resident weight (MB), replacing that feature's flat
    ``vram_delta_mb`` term while keeping the activation (per-megapixel) term. Unsupplied features keep the
    conservative seed.

    ``post_processing_upscale_factor`` is the linear scale of the requested upscaler (see
    :func:`hordelib.pipeline.constants.max_upscale_factor`). The upscale activation peak scales with the
    *output* megapixels, so the ``post_processing_upscale`` feature's per-megapixel term is applied against
    ``factor**2`` the generation megapixels: a 4x upscale of a 1 MP image works a 16 MP tensor, the dominant
    cost of the post-processing phase. The default of 1.0 leaves the activation at generation resolution.

    Never raises on unknown baselines: pre-flight callers need an answer for every
    job, so unknown baselines use a heavy fallback seed and are flagged via
    ``baseline_known=False``.
    """
    burden = _REGISTRY.baselines.get(baseline)
    baseline_known = burden is not None
    if burden is None:
        burden = _FALLBACK_BASELINE_BURDEN

    megapixels = (width * height) / 1_000_000

    vram_sampling_mb = burden.vram_base_mb + round(burden.vram_per_megapixel_mb * megapixels * batch)
    vram_post_processing_mb = 0
    ram_mb = burden.ram_base_mb
    downloads_expected: list[DownloadTrigger] = []

    for kind in features or []:
        impact = _REGISTRY.features.get(kind)
        if impact is None:
            continue
        if impact.applies_to_baselines is not None and baseline not in impact.applies_to_baselines:
            continue
        if aux_model_weights_mb is not None and kind in aux_model_weights_mb:
            weight_mb = round(aux_model_weights_mb[kind])
        else:
            weight_mb = impact.vram_delta_mb
        # The upscaler enlarges the image, so its activation peak scales with the *output* megapixels
        # (factor**2 the generation megapixels); every other feature works at generation resolution.
        activation_megapixels = megapixels
        if kind == FEATURE_KIND.post_processing_upscale and post_processing_upscale_factor > 1.0:
            activation_megapixels = megapixels * (post_processing_upscale_factor**2)
        vram_delta = weight_mb + round(impact.vram_delta_per_megapixel_mb * activation_megapixels)
        if impact.phase == FEATURE_PHASE.post_processing:
            vram_post_processing_mb += vram_delta
        else:
            vram_sampling_mb += vram_delta
        ram_mb += impact.ram_delta_mb
        if impact.download.triggered:
            downloads_expected.append(impact.download)

    if model_record is not None and model_record.size_on_disk_bytes is not None:
        disk_bytes = model_record.size_on_disk_bytes
    else:
        disk_bytes = round(burden.typical_disk_gb * _BYTES_PER_GB)
    disk_bytes += sum((trigger.typical_size_mb or 0) * 1024 * 1024 for trigger in downloads_expected)

    return BurdenEstimate(
        vram_mb=vram_sampling_mb + vram_post_processing_mb,
        vram_sampling_mb=vram_sampling_mb,
        vram_post_processing_mb=vram_post_processing_mb,
        ram_mb=ram_mb,
        disk_bytes_needed=disk_bytes,
        downloads_expected=downloads_expected,
        baseline_known=baseline_known,
    )
