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


class StageGraphUnsupportedError(RuntimeError):
    """A disaggregated stage was asked to cut a graph whose shape it does not support.

    The stage entry points cut a materialized family graph at a fixed set of canonical nodes
    (a combined ``model_loader``, ``prompt``/``negative_prompt``, ``sampler``, ``vae_encode``,
    ``vae_decode``). Families whose graph differs (Flux's ``SamplerCustomAdvanced``, Qwen/Z-Image's
    split ``CLIPLoader``/``VAELoader``, controlnet/inpaint/cascade) do not present that shape;
    raising here keeps them off the disaggregated path instead of running mis-wired.
    """


class OutputKind(StrEnum):
    """The modality of a pipeline output."""

    IMAGE = auto()
    # Future modalities (VIDEO, AUDIO, TEXT) are added here; collection is keyed by the
    # declared output node, so new kinds need no changes to the collection path.
    CONDITIONING = auto()
    LATENT = auto()
    # CONDITIONING/LATENT are the disaggregated-stage intermediates (text-encode -> sample ->
    # decode). They ride the same output-collection path as images: the stage output nodes emit
    # a bytes blob under the standard ui-entry contract; only the OutputSpec.kind differs.


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


RETAINED_WEIGHTS_EVICTED_METADATA_KEY: str = "retained_weights_evicted"
"""Artifact metadata key set when a run granted a retention deferral ended with nothing on the device.

The host grants ``defer_vram_unload`` so the weights survive the run for the next same-model job to
reuse. ComfyUI's own memory manager can still free every other model on the device to satisfy an
allocation, which takes those weights with it, and the host cannot see that from outside the process.
Carried per artifact (like the sampler-truncation record) so the host's prediction of what the card
holds can be corrected from the result rather than from a log line."""


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


_UNLOAD_REMAINDER_FLOOR_MB: float = 256.0
"""Weights (MB) a completed unload may leave behind before it counts as incomplete.

An unload is judged on what the device still holds, not on what the command returned. Some residue is
ordinary: an allocator block the driver has not handed back yet, a small support module a live reference
still pins. A quarter of a gigabyte is well under any checkpoint or text encoder, so anything above it is a
real model still sitting on the card."""


class VramUnloadResult(BaseModel):
    """What a full VRAM unload actually achieved, measured either side of the free.

    ComfyUI frees by walking its loaded-model list and dropping what it can; an entry a live reference still
    pins is skipped and stays on the device, and the caller is told nothing. A worker that reports the model
    moved to host RAM on the strength of having *asked* then carries gigabytes the card is holding but its
    ledger is not. These fields are what let the caller judge the unload by its result.
    """

    freed_mb: float
    """Device free VRAM gained across the unload, from the readings taken either side of it.

    An estimate: the reading is process-local and other tenants move underneath it. Negative readings are
    clamped away, so this is a floor on what came back rather than an exact figure."""
    remaining_loaded_models: int
    """Loaded models ComfyUI still lists for this process after the unload. Zero is a complete unload."""
    remaining_loaded_weights_mb: float | None
    """On-device weights those remaining models still hold, or None when the figure cannot be read.

    None is not zero: an unreadable figure means the caller must fall back on the count."""
    dead_model_refs_dropped: int
    """Loaded-model entries whose patcher reference had died, dropped from ComfyUI's list here.

    Such an entry answers no question about itself (every accessor raises) and can never be unloaded, so it
    would otherwise keep the list non-empty forever and make every later unload read as incomplete."""

    @property
    def complete(self) -> bool:
        """Whether the device came back: nothing still listed, or nothing of consequence still resident."""
        if self.remaining_loaded_models == 0:
            return True
        if self.remaining_loaded_weights_mb is None:
            return False
        return self.remaining_loaded_weights_mb <= _UNLOAD_REMAINDER_FLOOR_MB


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
        device_free_truth_mb: float | None = None,
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
            device_free_truth_mb: The caller-measured device-level free VRAM (MB) at dispatch. When
                provided, ComfyUI's view of free VRAM during this run is clamped so shortfall-based
                freeing acts against measured device truth rather than the process-local reading,
                which overstates free memory under WDDM. Defaults to None (no clamp).

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

    def free_vram(self) -> VramUnloadResult:
        """Move models out of VRAM (to system RAM where applicable) and report what the device gave back.

        Returns:
            VramUnloadResult: What the unload freed and what it left behind, so the caller can judge the
                unload by its result rather than by having issued it.
        """
        ...

    def free_ram(self) -> None:
        """Aggressively unload models from system RAM."""
        ...

    def vram_stats(self) -> VRAMStats:
        """Return current VRAM usage for the active device."""
        ...
