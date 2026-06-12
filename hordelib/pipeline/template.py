"""Pipeline templates: a graph file plus its declared, typed parameter surface.

A template's bindings are the complete list of ways a payload can influence its graph. This
declared surface is what makes templates auditable — and is the future vetting boundary for
user-provided pipelines (data-only bindings + a class_type allowlist).
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hordelib.pipeline.graph import ComfyGraph
from hordelib.pipeline.payload import ImageGenPayload

Transform = Callable[[ImageGenPayload], Any]
"""Computes a binding value from the payload."""

PatchStep = Callable[[ComfyGraph, ImageGenPayload], None]
"""Mutates a materialized graph based on the payload (e.g. LoRA chain insertion)."""


@dataclass(frozen=True)
class ParamBinding:
    """Binds one graph input to a payload field or a computed value.

    Exactly one of ``source`` or ``transform`` must be set.
    """

    target: str
    """Dotted graph input, e.g. ``"sampler.steps"``."""
    source: str | None = None
    """The ImageGenPayload field name to copy from."""
    transform: Transform | None = None
    """A function computing the value from the whole payload."""
    multiplier: float | None = None
    """Optional multiplier applied (and rounded) to a numeric source value."""

    def __post_init__(self) -> None:
        if (self.source is None) == (self.transform is None):
            raise ValueError(f"Binding for {self.target!r} must set exactly one of source/transform")
        if self.multiplier is not None and self.source is None:
            raise ValueError(f"Binding for {self.target!r}: multiplier requires a source field")

    def resolve(self, payload: ImageGenPayload) -> Any:
        if self.transform is not None:
            return self.transform(payload)
        value = getattr(payload, self.source)  # type: ignore[arg-type]
        if self.multiplier is not None and value is not None:
            return round(value * self.multiplier)
        return value


@dataclass(frozen=True)
class PipelineTemplate:
    """A named pipeline: graph file + bindings + optional payload-driven patch steps."""

    name: str
    graph_file: Path
    bindings: tuple[ParamBinding, ...]
    patch_steps: tuple[PatchStep, ...] = ()
    extra_inputs: dict[str, Any] = field(default_factory=dict)
    """Static inputs always applied (e.g. ``model_loader.will_load_loras``)."""

    def load_graph(self) -> ComfyGraph:
        """Load (and cache) the template's graph; returns a fresh copy each call."""
        cached = _GRAPH_CACHE.get(self.graph_file)
        if cached is None:
            cached = ComfyGraph.from_file(self.graph_file)
            _GRAPH_CACHE[self.graph_file] = cached
        return cached.copy()

    def materialize(self, payload: ImageGenPayload) -> ComfyGraph:
        """Produce a fully parameterized graph for this payload."""
        graph = self.load_graph()

        params: dict[str, Any] = dict(self.extra_inputs)
        for binding in self.bindings:
            params[binding.target] = binding.resolve(payload)
        # Bindings may legitimately target nodes pruned from pipeline variants sharing a
        # binding set; missing nodes are skipped (same semantics as the legacy mapping).
        graph.set_inputs(params)

        for patch_step in self.patch_steps:
            patch_step(graph, payload)

        return graph


_GRAPH_CACHE: dict[Path, ComfyGraph] = {}
