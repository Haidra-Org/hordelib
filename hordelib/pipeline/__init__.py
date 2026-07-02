"""The typed pipeline layer.

This package replaces the dict-based payload validation and string-keyed pipeline surgery in
``hordelib.horde`` with typed, unit-testable components:

- :mod:`hordelib.pipeline.constants` — sampler/scheduler/controlnet vocabularies.
- :mod:`hordelib.pipeline.payload` — pydantic payload models with Horde's clamp-don't-reject
  semantics.
- :mod:`hordelib.pipeline.graph` — :class:`ComfyGraph`, a typed wrapper over API-format JSON.
- :mod:`hordelib.pipeline.definition` — :class:`PipelineDefinition`, everything one pipeline
  declares (graph, bindings, selection, outputs) in one object.
- :mod:`hordelib.pipeline.registry` — explicit, tier-ordered pipeline selection.

Nothing in this package imports ComfyUI.
"""

from hordelib.pipeline.definition import (
    NodeHandle,
    OutputKind,
    OutputSpec,
    ParamBinding,
    PayloadFeature,
    PipelineDefinition,
    ScaledSource,
    SelectionTier,
    Selector,
    node,
    scaled,
)
from hordelib.pipeline.graph import ComfyGraph, NodeRef
from hordelib.pipeline.payload import ImageGenPayload
from hordelib.pipeline.registry import PipelineRegistry

__all__ = [
    "ComfyGraph",
    "ImageGenPayload",
    "NodeHandle",
    "NodeRef",
    "OutputKind",
    "OutputSpec",
    "ParamBinding",
    "PayloadFeature",
    "PipelineDefinition",
    "PipelineRegistry",
    "ScaledSource",
    "SelectionTier",
    "Selector",
    "node",
    "scaled",
]
