"""Explicit, priority-ordered pipeline selection.

Replaces the legacy if/elif tree in ``HordeLib._get_appropriate_pipeline``: each registered
pipeline declares a predicate and a priority, and the highest-priority matching template wins.
"""

from collections.abc import Callable
from dataclasses import dataclass

from loguru import logger

from hordelib.pipeline.context import ModelContext
from hordelib.pipeline.payload import ImageGenPayload
from hordelib.pipeline.template import PipelineTemplate

Predicate = Callable[[ImageGenPayload, ModelContext], bool]


@dataclass(frozen=True)
class PipelineSpec:
    template: PipelineTemplate
    predicate: Predicate
    priority: int
    """Higher priorities are evaluated first; ties resolve by registration order."""


class PipelineRegistry:
    def __init__(self) -> None:
        self._specs: list[PipelineSpec] = []

    def register(self, spec: PipelineSpec) -> None:
        if any(existing.template.name == spec.template.name for existing in self._specs):
            raise ValueError(f"Pipeline {spec.template.name!r} is already registered")
        self._specs.append(spec)
        # Stable sort preserves registration order within a priority
        self._specs.sort(key=lambda s: -s.priority)

    def select(self, payload: ImageGenPayload, context: ModelContext) -> PipelineTemplate | None:
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

    def get(self, name: str) -> PipelineTemplate | None:
        for spec in self._specs:
            if spec.template.name == name:
                return spec.template
        return None

    def all_specs(self) -> list[PipelineSpec]:
        return list(self._specs)
