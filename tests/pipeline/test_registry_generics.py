"""The pipeline core is generic: a second family with its own payload/context types works.

These tests use a toy family (no graph IO beyond a real packaged pipeline file) to prove the
template/registry machinery carries no image-generation assumptions.
"""

from pathlib import Path

import pytest
from pydantic import BaseModel

from hordelib.pipeline.registry import PipelineRegistry, PipelineSpec
from hordelib.pipeline.template import ParamBinding, PipelineTemplate

PIPELINES_DIR = Path(__file__).parent.parent.parent / "hordelib" / "pipelines"


class ToyPayload(BaseModel):
    model: str
    strength: float = 1.0


class ToyContext(BaseModel):
    model_file: str | None = None


def _toy_template(name: str) -> PipelineTemplate[ToyPayload, ToyContext]:
    return PipelineTemplate(
        name=name,
        graph_file=PIPELINES_DIR / "pipeline_image_upscale.json",
        bindings=(
            ParamBinding(target="model_loader.model_name", source="model"),
            ParamBinding(target="output_image.filename_prefix", transform=lambda p: f"toy-{p.model}"),
        ),
    )


def _build_registry() -> PipelineRegistry[ToyPayload, ToyContext]:
    registry: PipelineRegistry[ToyPayload, ToyContext] = PipelineRegistry()
    registry.register(
        PipelineSpec(
            template=_toy_template("toy_special"),
            predicate=lambda p, c: c.model_file is not None,
            priority=10,
        ),
    )
    registry.register(
        PipelineSpec(
            template=_toy_template("toy_fallback"),
            predicate=lambda p, c: True,
            priority=0,
        ),
    )
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
    template = _toy_template("toy")
    graph = template.materialize(ToyPayload(model="RealESRGAN_x4plus"), ToyContext())

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
