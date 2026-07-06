"""The declared public API surface of hordelib (horde-engine) for consumers.

Workers and other consumers should import from this module only; anything not re-exported
here is internal and may change without notice. The import of this module is comfy-free:
nothing here triggers a ComfyUI import (``hordelib.initialise()`` does that explicitly).

The contract is enforced by ``tests/meta/test_public_api_contract.py``.
"""

from typing import TYPE_CHECKING

from hordelib import initialise, is_initialised
from hordelib.execution.interface import (
    ExecutionBackend,
    OutputArtifact,
    ProgressCallback,
    VRAMStats,
)
from hordelib.execution.sampling_lease import (
    SamplingLease,
    install_sampling_lease_hook,
    set_gpu_sampling_lease,
)
from hordelib.feature_impact import (
    FEATURE_KIND,
    FEATURE_PHASE,
    BaselineBurden,
    BurdenEstimate,
    CalibrationSample,
    DownloadTrigger,
    FeatureImpact,
    FeatureImpactRegistry,
    estimate_job_burden,
    get_baseline_burden,
    get_feature_impact_registry,
)
from hordelib.feature_requirements import (
    FeatureRequirement,
    available_features,
    feature_available,
    get_feature_requirement,
    get_feature_requirement_registry,
    missing_packages,
)
from hordelib.horde import (
    HordeLib,
    ProgressReport,
    ProgressState,
    ResultingImageReturn,
)
from hordelib.metrics import (
    DownloadEvent,
    JobPhaseMetrics,
    MetricsCollector,
    ModelLoadEvent,
    SamplingStats,
    get_metrics_collector,
)
from hordelib.pipeline.constants import (
    CONTROLNET_ANNOTATOR_DOWNLOAD_BYTES,
    controlnet_annotator_download_bytes,
)
from hordelib.pipeline.identifiers import (
    AUTO_PIPELINE,
    AutoPipeline,
    ImagePipeline,
)
from hordelib.pipeline.payload import ImageGenPayload
from hordelib.pipeline.payload_pp import (
    FacefixPayload,
    PostProcessingPayload,
    PostProcessorKind,
    StripBackgroundPayload,
    UpscalePayload,
    classify_post_processor,
)
from hordelib.preload import controlnet_annotators_present
from hordelib.utils.ioredirect import ComfyUIProgress, ComfyUIProgressUnit
from hordelib.utils.logger import HordeLog
from hordelib.utils.torch_memory import (
    AcceleratorInfo,
    AcceleratorKind,
    JobVramProfile,
    ProcessVramStats,
    ResidentFootprint,
    ResidentFootprintRecorder,
    clear_accelerator_cache,
    enumerate_accelerators,
    get_accelerator_utilization_percent,
    get_free_ram_mb,
    get_loaded_weights_offloaded_mb,
    get_process_vram_stats,
    get_torch_device_free_vram_mb,
    get_torch_free_vram_mb,
    get_torch_total_vram_mb,
    log_free_ram,
    offthread_vram_sampling_ready,
    record_job_vram_profile,
    record_resident_footprint,
)
from hordelib.vram_planning import (
    compute_extra_reserved_mb,
    compute_inference_reserve_mb,
    compute_weight_budget_mb,
)

if TYPE_CHECKING:
    # Re-exported lazily (see ``__getattr__`` below) so the type checker still sees the name without the
    # eager import that would drag torch into every consumer of this facade.
    from hordelib.shared_model_manager import SharedModelManager

__all__ = [
    "AUTO_PIPELINE",
    "CONTROLNET_ANNOTATOR_DOWNLOAD_BYTES",
    "FEATURE_KIND",
    "FEATURE_PHASE",
    "AcceleratorInfo",
    "AcceleratorKind",
    "AutoPipeline",
    "BaselineBurden",
    "BurdenEstimate",
    "CalibrationSample",
    "ComfyUIProgress",
    "ComfyUIProgressUnit",
    "DownloadEvent",
    "DownloadTrigger",
    "FeatureImpact",
    "FeatureImpactRegistry",
    "FeatureRequirement",
    "ExecutionBackend",
    "FacefixPayload",
    "HordeLib",
    "HordeLog",
    "ImageGenPayload",
    "ImagePipeline",
    "JobPhaseMetrics",
    "MetricsCollector",
    "ModelLoadEvent",
    "OutputArtifact",
    "PostProcessingPayload",
    "PostProcessorKind",
    "ProgressCallback",
    "ProgressReport",
    "ProgressState",
    "ResultingImageReturn",
    "SamplingLease",
    "SamplingStats",
    "SharedModelManager",
    "StripBackgroundPayload",
    "UpscalePayload",
    "VRAMStats",
    "available_features",
    "classify_post_processor",
    "clear_accelerator_cache",
    "compute_extra_reserved_mb",
    "compute_inference_reserve_mb",
    "compute_weight_budget_mb",
    "controlnet_annotator_download_bytes",
    "controlnet_annotators_present",
    "enumerate_accelerators",
    "estimate_job_burden",
    "feature_available",
    "get_baseline_burden",
    "get_feature_impact_registry",
    "get_feature_requirement",
    "get_feature_requirement_registry",
    "missing_packages",
    "get_accelerator_utilization_percent",
    "get_free_ram_mb",
    "get_loaded_weights_offloaded_mb",
    "get_process_vram_stats",
    "get_torch_device_free_vram_mb",
    "get_metrics_collector",
    "get_torch_free_vram_mb",
    "get_torch_total_vram_mb",
    "JobVramProfile",
    "ProcessVramStats",
    "ResidentFootprint",
    "ResidentFootprintRecorder",
    "record_job_vram_profile",
    "record_resident_footprint",
    "initialise",
    "install_sampling_lease_hook",
    "is_initialised",
    "log_free_ram",
    "offthread_vram_sampling_ready",
    "set_gpu_sampling_lease",
]


def __getattr__(name: str) -> object:
    """Lazily resolve the one torch-heavy re-export so the facade's import cost stays honest.

    Every other name above comes from a torch-free submodule, so ``import hordelib.api`` is cheap
    (~75MB) and pulls no torch. ``SharedModelManager`` is the exception: it transitively imports the
    concrete model managers, which import torch (~500MB RSS). Re-exporting it eagerly would tax every
    consumer of this facade with a full torch load, including the worker's torch-free orchestrator and
    benchmark planner that only want the pure-Python helpers. PEP 562 module ``__getattr__`` defers that
    import until ``SharedModelManager`` is actually accessed, so a consumer pays for torch only when it
    asks for something that genuinely needs it.
    """
    if name == "SharedModelManager":
        from hordelib.shared_model_manager import SharedModelManager

        return SharedModelManager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
