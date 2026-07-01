"""Backend-agnostic execution interface.

This module must remain importable at any time (before ``hordelib.initialise()``), so it must
never import ComfyUI or anything that transitively does.
"""

import io
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum, auto
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from hordelib.utils.ioredirect import ComfyUIProgress

ProgressCallback = Callable[[ComfyUIProgress, str], None]
"""Callback invoked with progress updates and the latest output message during a pipeline run."""


class OutputKind(StrEnum):
    """The modality of a pipeline output."""

    IMAGE = auto()
    # Future modalities (VIDEO, AUDIO, TEXT) are added here; collection is keyed by the
    # declared output node, so new kinds need no changes to the collection path.


@dataclass(frozen=True)
class OutputSpec:
    """Declares one node a pipeline produces results from.

    Pipeline definitions declare their outputs with these; the execution backend collects
    artifacts per declared node and fails loudly (naming the node) when one produces nothing.
    """

    node: str
    """The graph node title, e.g. ``"output_image"``."""
    kind: OutputKind = OutputKind.IMAGE


DEFAULT_IMAGE_OUTPUTS: tuple[OutputSpec, ...] = (OutputSpec(node="output_image"),)
"""The historical single-image-output convention, used where no explicit declaration exists."""


class OutputArtifact(BaseModel):
    """A single output produced by a pipeline run.

    Currently always a PNG image; the mime type field exists so that future modalities
    (audio/video) can flow through the same interface.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    data: io.BytesIO
    mime_type: str = "image/png"
    kind: OutputKind = OutputKind.IMAGE
    source_node: str | None = None
    """The graph node title this artifact was collected from, when the backend knows it."""
    metadata: dict[str, Any] = {}


class VRAMStats(BaseModel):
    """A point-in-time snapshot of VRAM usage on the active torch device."""

    total_mb: int
    free_mb: int


@runtime_checkable
class ExecutionBackend(Protocol):
    """The contract hordelib uses to execute pipelines on some ComfyUI runtime.

    Implementations own all ComfyUI specifics (imports, monkeypatches, memory management).
    Callers hand over a fully materialized API-format graph and receive output artifacts.
    """

    def start(self) -> None:
        """Make the backend ready to run pipelines.

        Raises:
            RuntimeError: If the backend's prerequisites are not met
                (e.g. ``hordelib.initialise()`` was never called for the in-process backend).
        """
        ...

    def run_pipeline(
        self,
        graph: dict[str, Any],
        *,
        outputs: tuple[OutputSpec, ...] = DEFAULT_IMAGE_OUTPUTS,
        progress_callback: ProgressCallback | None = None,
        defer_vram_unload: bool = False,
    ) -> list[OutputArtifact]:
        """Execute a fully materialized API-format graph and return its outputs.

        Args:
            graph: The pipeline graph in ComfyUI API format, with all parameters already set.
            outputs: The nodes the graph is declared to produce results from. Every declared
                node must yield at least one artifact or the run fails naming the node.
            progress_callback: Optionally called with progress updates during execution.
            defer_vram_unload: When True, keep the model resident in VRAM after this run instead of
                evicting it, so a following job that reuses it skips the RAM->VRAM reload. The caller
                owns the VRAM-safety decision (it must know the model fits alongside the live set);
                backends that never evict between runs ignore this. Defaults to False.

        Returns:
            list[OutputArtifact]: The outputs produced by the run, tagged with their source node.

        Raises:
            RuntimeError: If a declared output produced no artifacts (e.g. an execution error
                inside the ComfyUI runtime).
        """
        ...

    def interrupt(self) -> None:
        """Request that the currently running pipeline be interrupted as soon as possible."""
        ...

    def free_vram(self) -> None:
        """Move models out of VRAM (to system RAM where applicable)."""
        ...

    def free_ram(self) -> None:
        """Aggressively unload models from system RAM."""
        ...

    def vram_stats(self) -> VRAMStats:
        """Return current VRAM usage for the active device."""
        ...
