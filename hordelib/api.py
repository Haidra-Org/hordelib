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
from hordelib.horde import (
    HordeLib,
    ProgressReport,
    ProgressState,
    ResultingImageReturn,
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
    get_free_ram_mb,
    get_torch_free_vram_mb,
    get_torch_total_vram_mb,
    log_free_ram,
)

__all__ = [
    "ComfyUIProgress",
    "ComfyUIProgressUnit",
    "ExecutionBackend",
    "FacefixPayload",
    "HordeLib",
    "HordeLog",
    "ImageGenPayload",
    "OutputArtifact",
    "PostProcessingPayload",
    "PostProcessorKind",
    "ProgressCallback",
    "ProgressReport",
    "ProgressState",
    "ResultingImageReturn",
    "SharedModelManager",
    "StripBackgroundPayload",
    "UpscalePayload",
    "VRAMStats",
    "classify_post_processor",
    "get_free_ram_mb",
    "get_torch_free_vram_mb",
    "get_torch_total_vram_mb",
    "initialise",
    "is_initialised",
    "log_free_ram",
]
