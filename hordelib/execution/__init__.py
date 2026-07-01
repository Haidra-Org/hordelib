"""The execution bridge: the interface hordelib uses to run pipelines on a ComfyUI backend.

This package isolates ComfyUI from the rest of hordelib:

- :mod:`hordelib.execution.interface` defines the backend-agnostic protocol and result types.
- :mod:`hordelib.execution.graph_utils` contains pure functions for manipulating API-format
  pipeline graphs (no ComfyUI imports; unit-testable without a GPU).
- :mod:`hordelib.execution.in_process` provides the in-process ComfyUI backend.

Only modules inside this package (and the legacy :mod:`hordelib.comfy_horde`) may import
ComfyUI internals.
"""

from hordelib.execution.interface import (
    ExecutionBackend,
    OutputArtifact,
    OutputKind,
    OutputSpec,
    VRAMStats,
)

__all__ = [
    "ExecutionBackend",
    "OutputArtifact",
    "OutputKind",
    "OutputSpec",
    "VRAMStats",
]
