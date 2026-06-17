"""The declared public API surface of hordelib (horde-engine) for consumers.

Workers and other consumers should import from this module only; anything not re-exported
here is internal and may change without notice. The import of this module is comfy-free:
nothing here triggers a ComfyUI import (``hordelib.initialise()`` does that explicitly).

The contract is enforced by ``tests/meta/test_public_api_contract.py``.
"""

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
from hordelib.pipeline.payload import ImageGenPayload
from hordelib.pipeline.payload_pp import (
    FacefixPayload,
    PostProcessingPayload,
    PostProcessorKind,
    StripBackgroundPayload,
    UpscalePayload,
    classify_post_processor,
)
from hordelib.shared_model_manager import SharedModelManager
from hordelib.utils.ioredirect import ComfyUIProgress, ComfyUIProgressUnit
from hordelib.utils.logger import HordeLog
from hordelib.utils.torch_memory import (
    AcceleratorInfo,
    AcceleratorKind,
    clear_accelerator_cache,
    enumerate_accelerators,
    get_free_ram_mb,
    get_torch_free_vram_mb,
    get_torch_total_vram_mb,
    log_free_ram,
)

__all__ = [
    "FEATURE_KIND",
    "AcceleratorInfo",
    "AcceleratorKind",
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
    "enumerate_accelerators",
    "estimate_job_burden",
    "feature_available",
    "get_baseline_burden",
    "get_feature_impact_registry",
    "get_feature_requirement",
    "get_feature_requirement_registry",
    "missing_packages",
    "get_free_ram_mb",
    "get_metrics_collector",
    "get_torch_free_vram_mb",
    "get_torch_total_vram_mb",
    "initialise",
    "install_sampling_lease_hook",
    "is_initialised",
    "log_free_ram",
    "set_gpu_sampling_lease",
]
