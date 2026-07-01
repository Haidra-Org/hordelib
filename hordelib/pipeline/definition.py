"""Pipeline definitions: everything one pipeline declares, in one object.

A :class:`PipelineDefinition` colocates the whole authoring surface of a pipeline: the graph
file, the typed parameter bindings (the complete list of ways a payload can influence the
graph), the selection declaration, the opt-in patch steps, and the declared outputs. This
declared surface is what makes pipelines auditable, and is the future vetting boundary for
user-provided pipelines (data-only bindings + a class_type allowlist).

Selection is declarative: a :class:`Selector` names its :class:`SelectionTier` and criteria
instead of encoding them in a lambda and a magic priority number. Families with
context-specific selection axes (e.g. the image family's baseline) extend ``Selector`` with
typed fields; see ``hordelib.pipeline.families``.

The machinery is generic over the payload model: each pipeline family (image generation,
post-processing, future modalities) declares its own pydantic payload type, and the bindings,
features, and patch steps are typed against it.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from hordelib.execution.interface import OutputKind, OutputSpec
from hordelib.pipeline.graph import ComfyGraph

__all__ = [
    "PACKAGED_PIPELINES_DIR",
    "OutputKind",
    "OutputSpec",
    "ParamBinding",
    "pipeline_graph",
    "PatchStep",
    "PayloadFeature",
    "PipelineDefinition",
    "Predicate",
    "SelectionTier",
    "Selector",
    "Transform",
]

PACKAGED_PIPELINES_DIR = Path(__file__).parent.parent / "pipelines"
"""Where hordelib's packaged ComfyUI graph exports live."""


def pipeline_graph(name: str) -> Path:
    """Return the path of a packaged pipeline graph (``hordelib/pipelines/pipeline_<name>.json``)."""
    return PACKAGED_PIPELINES_DIR / f"pipeline_{name}.json"


type Transform[PayloadT: BaseModel] = Callable[[PayloadT], Any]
"""Computes a binding value from the payload."""

type PatchStep[PayloadT: BaseModel, ContextT] = Callable[[ComfyGraph, PayloadT, ContextT], None]
"""Mutates a materialized graph from the payload and the resolved context.

Bindings carry pure payload intent; patch steps are where resolved facts (model files on
disk, downloaded LoRAs) and structural surgery (chain insertion, rewires) meet the graph.
"""

type Predicate[PayloadT: BaseModel, ContextT] = Callable[[PayloadT, ContextT], bool]
"""An arbitrary selection predicate (the discouraged escape hatch on :class:`Selector`)."""


@dataclass(frozen=True)
class ParamBinding[PayloadT: BaseModel]:
    """Binds one graph input to a payload field or a computed value.

    Exactly one of ``source`` or ``transform`` must be set.
    """

    target: str
    """Dotted graph input, e.g. ``"sampler.steps"``."""
    source: str | None = None
    """The payload field name to copy from."""
    transform: Transform[PayloadT] | None = None
    """A function computing the value from the whole payload."""
    multiplier: float | None = None
    """Optional multiplier applied (and rounded) to a numeric source value."""

    def __post_init__(self) -> None:
        if (self.source is None) == (self.transform is None):
            raise ValueError(f"Binding for {self.target!r} must set exactly one of source/transform")
        if self.multiplier is not None and self.source is None:
            raise ValueError(f"Binding for {self.target!r}: multiplier requires a source field")

    def resolve(self, payload: PayloadT) -> Any:
        if self.transform is not None:
            return self.transform(payload)
        value = getattr(payload, self.source)  # type: ignore[arg-type]
        if self.multiplier is not None and value is not None:
            return round(value * self.multiplier)
        return value


@dataclass(frozen=True)
class PayloadFeature[PayloadT: BaseModel]:
    """A named, reusable payload condition used as a declarative selection criterion.

    Families declare their feature vocabulary once (e.g. ``CONTROLNET``, ``HIRES_FIX`` in the
    image family) and selectors reference the named instances, so the conditions a pipeline
    selects on are auditable in one place instead of scattered across lambdas.
    """

    name: str
    is_set: Callable[[PayloadT], bool]

    def __repr__(self) -> str:
        return f"PayloadFeature({self.name!r})"


class SelectionTier(IntEnum):
    """Named selection precedence tiers; higher tiers are checked first.

    Within a tier, :attr:`Selector.order` breaks ties (lower checked first). The tier names
    describe the *reason* a pipeline outranks another; families use the tiers that apply to
    them. This replaces the legacy magic priority integers.
    """

    WORKFLOW_OVERRIDE = 100
    """An explicit workflow request (e.g. qr_code) beats all inference.

    Everything else about the payload is reinterpreted in the workflow's terms."""
    BASELINE_FAMILY = 80
    """A model architecture with its own graph shape (cascade, flux, qwen, ...)."""
    FEATURE = 60
    """A payload feature needing a dedicated graph variant (e.g. controlnet)."""
    PAINTING = 40
    """Masked/inpainting/outpainting source-image modes."""
    GENERIC_VARIANT = 20
    """A variant of the fallback graph (e.g. hires fix)."""
    FALLBACK = 0
    """The catch-all; exactly one per family, with no selection criteria."""


@dataclass(frozen=True)
class Selector[PayloadT: BaseModel, ContextT]:
    """Declares when a pipeline is selected: a precedence tier plus payload criteria.

    All criteria must hold (AND semantics). Families with context-specific selection axes
    subclass this with additional typed fields and extend :meth:`matches`.

    ``extra_predicate`` is the escape hatch for conditions the declarative fields cannot
    express; pipelines using it are flagged by the contract tests, so reach for a named
    :class:`PayloadFeature` first.
    """

    tier: SelectionTier
    order: int = 0
    """Tie-break within the tier; lower is checked first."""
    features: tuple[PayloadFeature[PayloadT], ...] = ()
    extra_predicate: Predicate[PayloadT, ContextT] | None = None

    def matches(self, payload: PayloadT, context: ContextT) -> bool:
        """Return whether this selector's criteria all hold for the payload and context."""
        if any(not feature.is_set(payload) for feature in self.features):
            return False
        if self.extra_predicate is not None and not self.extra_predicate(payload, context):
            return False
        return True

    def has_criteria(self) -> bool:
        """Return whether any selection criterion is declared (subclasses add their axes).

        A non-fallback selector without criteria would shadow everything below it; the
        registration audit rejects that.
        """
        return bool(self.features) or self.extra_predicate is not None


@dataclass(frozen=True)
class PipelineDefinition[PayloadT: BaseModel, ContextT]:
    """A named pipeline: graph file, bindings, selection, patch steps, and outputs.

    This is the single artifact a pipeline author writes; see ``docs/adding-a-pipeline.md``
    for the worked tutorial.
    """

    name: str
    graph_file: Path
    selector: Selector[PayloadT, ContextT]
    bindings: tuple[ParamBinding[PayloadT], ...]
    outputs: tuple[OutputSpec, ...]
    patch_steps: tuple[PatchStep[PayloadT, ContextT], ...] = ()
    extra_inputs: dict[str, Any] = field(default_factory=dict)
    """Static inputs always applied (e.g. ``model_loader.will_load_loras``)."""
    known_title_collisions: frozenset[str] = frozenset()
    """Grandfathered duplicate node titles in the graph file.

    Duplicate titles silently drop a node at load time, so the audit rejects them; listing a
    title here acknowledges a legacy graph's collision until the export is repaired (which
    changes the loaded graph and therefore needs GPU-oracle verification). Never add entries
    for new pipelines; fix the export instead.
    """

    def load_graph(self) -> ComfyGraph:
        """Load (and cache) the definition's graph; returns a fresh copy each call."""
        cached = _GRAPH_CACHE.get(self.graph_file)
        if cached is None:
            cached = ComfyGraph.from_file(self.graph_file)
            _GRAPH_CACHE[self.graph_file] = cached
        return cached.copy()

    def materialize(self, payload: PayloadT, context: ContextT) -> ComfyGraph:
        """Produce a fully parameterized graph for this payload and resolved context."""
        graph = self.load_graph()

        params: dict[str, Any] = dict(self.extra_inputs)
        for binding in self.bindings:
            params[binding.target] = binding.resolve(payload)
        # Definitions declare their exact binding sets (audited at registration), so a
        # missing target here is a bug, not a pruned variant.
        graph.set_inputs(params, strict=True)

        for patch_step in self.patch_steps:
            patch_step(graph, payload, context)

        return graph


_GRAPH_CACHE: dict[Path, ComfyGraph] = {}
