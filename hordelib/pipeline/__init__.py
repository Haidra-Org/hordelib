"""The typed pipeline layer.

This package replaces the dict-based payload validation and string-keyed pipeline surgery in
``hordelib.horde`` with typed, unit-testable components:

- :mod:`hordelib.pipeline.constants` — sampler/scheduler/controlnet vocabularies.
- :mod:`hordelib.pipeline.payload` — pydantic payload models with Horde's clamp-don't-reject
  semantics.
- :mod:`hordelib.pipeline.graph` — :class:`ComfyGraph`, a typed wrapper over API-format JSON.
- :mod:`hordelib.pipeline.template` — :class:`PipelineTemplate`, a graph plus its declared,
  typed parameter surface.
- :mod:`hordelib.pipeline.registry` — explicit, priority-ordered pipeline selection.

Nothing in this package imports ComfyUI.
"""

from hordelib.pipeline.graph import ComfyGraph, NodeRef
from hordelib.pipeline.payload import ImageGenPayload
from hordelib.pipeline.registry import PipelineRegistry, PipelineSpec
from hordelib.pipeline.template import ParamBinding, PipelineTemplate

__all__ = [
    "ComfyGraph",
    "ImageGenPayload",
    "NodeRef",
    "ParamBinding",
    "PipelineRegistry",
    "PipelineSpec",
    "PipelineTemplate",
]
