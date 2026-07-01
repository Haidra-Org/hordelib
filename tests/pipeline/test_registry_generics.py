"""The pipeline core is generic: a second family with its own payload/context types works.

These tests use a toy family (no graph IO beyond a real packaged pipeline file) to prove the
definition/registry machinery carries no image-generation assumptions, and that families can
extend :class:`Selector` with their own typed selection axes.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import override

import pytest
from pydantic import BaseModel

from hordelib.pipeline.definition import (
    OutputSpec,
    ParamBinding,
    PipelineDefinition,
    SelectionTier,
    Selector,
)
from hordelib.pipeline.registry import PipelineRegistry

PIPELINES_DIR = Path(__file__).parent.parent.parent / "hordelib" / "pipelines"


class ToyPayload(BaseModel):
    model: str
    strength: float = 1.0


class ToyContext(BaseModel):
    model_file: str | None = None


@dataclass(frozen=True)
class ToySelector(Selector[ToyPayload, ToyContext]):
    """A family-specific selector axis, mirroring how ImageSelector adds baselines."""

    requires_model_file: bool = False

    @override
    def matches(self, payload: ToyPayload, context: ToyContext) -> bool:
        if self.requires_model_file and context.model_file is None:
            return False
        return super().matches(payload, context)

    @override
    def has_criteria(self) -> bool:
        return self.requires_model_file or super().has_criteria()


def _toy_definition(name: str, selector: ToySelector) -> PipelineDefinition[ToyPayload, ToyContext]:
    return PipelineDefinition(
        name=name,
        graph_file=PIPELINES_DIR / "pipeline_image_upscale.json",
        selector=selector,
        bindings=(
            ParamBinding(target="model_loader.model_name", source="model"),
            ParamBinding(target="output_image.filename_prefix", transform=lambda p: f"toy-{p.model}"),
        ),
        outputs=(OutputSpec(node="output_image"),),
    )


def _build_registry() -> PipelineRegistry[ToyPayload, ToyContext]:
    registry: PipelineRegistry[ToyPayload, ToyContext] = PipelineRegistry()
    registry.register(
        _toy_definition("toy_special", ToySelector(tier=SelectionTier.FEATURE, requires_model_file=True)),
    )
    registry.register(_toy_definition("toy_fallback", ToySelector(tier=SelectionTier.FALLBACK)))
    return registry


def test_select_uses_non_image_context() -> None:
    registry = _build_registry()

    selected = registry.select(ToyPayload(model="x"), ToyContext(model_file="x.pth"))
    assert selected is not None
    assert selected.name == "toy_special"

    fallback = registry.select(ToyPayload(model="x"), ToyContext())
    assert fallback is not None
    assert fallback.name == "toy_fallback"


def test_materialize_with_non_image_payload() -> None:
    definition = _toy_definition("toy", ToySelector(tier=SelectionTier.FALLBACK))
    graph = definition.materialize(ToyPayload(model="RealESRGAN_x4plus"), ToyContext())

    api_dict = graph.to_api_dict()
    model_loader = next(node for node in api_dict.values() if node["_meta"]["title"] == "model_loader")
    assert model_loader["inputs"]["model_name"] == "RealESRGAN_x4plus"
    output_image = next(node for node in api_dict.values() if node["_meta"]["title"] == "output_image")
    assert output_image["inputs"]["filename_prefix"] == "toy-RealESRGAN_x4plus"


def test_binding_validation_is_payload_agnostic() -> None:
    with pytest.raises(ValueError, match="exactly one of source/transform"):
        ParamBinding[ToyPayload](target="a.b")
    with pytest.raises(ValueError, match="multiplier requires a source"):
        ParamBinding[ToyPayload](target="a.b", transform=lambda p: 1, multiplier=2.0)
