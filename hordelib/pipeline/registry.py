"""Explicit, priority-ordered pipeline selection.

Replaces the legacy if/elif tree in ``HordeLib._get_appropriate_pipeline``: each registered
pipeline declares a predicate and a priority, and the highest-priority matching template wins.

The registry is generic over the payload model and a family-specific selection context, so
each pipeline family (image generation, post-processing, future modalities) gets its own
fully typed registry. The context type is deliberately unconstrained: families that select on
model facts use ``ModelContext``, while simpler families may use a lighter object (or None).
"""

from collections.abc import Callable
from dataclasses import dataclass

from loguru import logger
from pydantic import BaseModel

from hordelib.pipeline.template import PipelineTemplate

type Predicate[PayloadT: BaseModel, ContextT] = Callable[[PayloadT, ContextT], bool]


@dataclass(frozen=True)
class PipelineSpec[PayloadT: BaseModel, ContextT]:
    template: PipelineTemplate[PayloadT, ContextT]
    predicate: Predicate[PayloadT, ContextT]
    priority: int
    """Higher priorities are evaluated first; ties resolve by registration order."""


class PipelineRegistry[PayloadT: BaseModel, ContextT]:
    def __init__(self) -> None:
        self._specs: list[PipelineSpec[PayloadT, ContextT]] = []

    def register(self, spec: PipelineSpec[PayloadT, ContextT]) -> None:
        if any(existing.template.name == spec.template.name for existing in self._specs):
            raise ValueError(f"Pipeline {spec.template.name!r} is already registered")
        self._specs.append(spec)
        # Stable sort preserves registration order within a priority
        self._specs.sort(key=lambda s: -s.priority)

    def select(self, payload: PayloadT, context: ContextT) -> PipelineTemplate[PayloadT, ContextT] | None:
        """Return the highest-priority template whose predicate matches, or None."""
        for spec in self._specs:
            if spec.predicate(payload, context):
                logger.debug(
                    "Pipeline selected: name={}, priority={}",
                    spec.template.name,
                    spec.priority,
                )
                return spec.template
        return None

    def get(self, name: str) -> PipelineTemplate[PayloadT, ContextT] | None:
        for spec in self._specs:
            if spec.template.name == name:
                return spec.template
        return None

    def all_specs(self) -> list[PipelineSpec[PayloadT, ContextT]]:
        return list(self._specs)
