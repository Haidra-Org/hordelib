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


class BurdenEstimate(BaseModel):
    """The estimated total burden of one job configuration."""

    vram_mb: int
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
    ),
    BaselineBurden(
        baseline="z_image_turbo",
        vram_base_mb=8000,
        ram_base_mb=14000,
        vram_per_megapixel_mb=1200,
        native_resolution=(1024, 1024),
        min_recommended_vram_mb=10000,
        max_recommended_batch=2,
        typical_disk_gb=8.0,
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
        notes="ESRGAN-family upscalers; model is small but activations on large outputs are not.",
    ),
    FeatureImpact(
        feature=FEATURE_KIND.post_processing_facefix,
        vram_delta_mb=1500,
        ram_delta_mb=2000,
        download=DownloadTrigger(triggered=True, typical_size_mb=350),
        notes="GFPGAN/CodeFormer weights.",
    ),
    FeatureImpact(
        feature=FEATURE_KIND.strip_background,
        vram_delta_mb=1000,
        ram_delta_mb=1500,
        download=DownloadTrigger(triggered=True, typical_size_mb=170),
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
) -> BurdenEstimate:
    """Estimate the total resource burden of one job configuration.

    Never raises on unknown baselines: pre-flight callers need an answer for every
    job, so unknown baselines use a heavy fallback seed and are flagged via
    ``baseline_known=False``.
    """
    burden = _REGISTRY.baselines.get(baseline)
    baseline_known = burden is not None
    if burden is None:
        burden = _FALLBACK_BASELINE_BURDEN

    megapixels = (width * height) / 1_000_000

    vram_mb = burden.vram_base_mb + round(burden.vram_per_megapixel_mb * megapixels * batch)
    ram_mb = burden.ram_base_mb
    downloads_expected: list[DownloadTrigger] = []

    for kind in features or []:
        impact = _REGISTRY.features.get(kind)
        if impact is None:
            continue
        if impact.applies_to_baselines is not None and baseline not in impact.applies_to_baselines:
            continue
        vram_mb += impact.vram_delta_mb + round(impact.vram_delta_per_megapixel_mb * megapixels)
        ram_mb += impact.ram_delta_mb
        if impact.download.triggered:
            downloads_expected.append(impact.download)

    if model_record is not None and model_record.size_on_disk_bytes is not None:
        disk_bytes = model_record.size_on_disk_bytes
    else:
        disk_bytes = round(burden.typical_disk_gb * _BYTES_PER_GB)
    disk_bytes += sum((trigger.typical_size_mb or 0) * 1024 * 1024 for trigger in downloads_expected)

    return BurdenEstimate(
        vram_mb=vram_mb,
        ram_mb=ram_mb,
        disk_bytes_needed=disk_bytes,
        downloads_expected=downloads_expected,
        baseline_known=baseline_known,
    )
