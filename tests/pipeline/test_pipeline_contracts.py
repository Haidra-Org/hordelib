"""Contract tests every registered pipeline definition must pass. No GPU required.

These are auto-parametrized over all registries, so a new pipeline gets full coverage the
moment it is listed in its family's pipeline tuple: its graph loads, its declared surface
audits clean, and it materializes with a canonical payload under strict binding semantics.
"""

import PIL.Image
import pytest

from hordelib.pipeline.context import ModelContext, PostProcessingContext
from hordelib.pipeline.definition import PipelineDefinition
from hordelib.pipeline.families.image_gen import IMAGE_PIPELINES
from hordelib.pipeline.families.post_processing import POST_PROCESSING_REGISTRY
from hordelib.pipeline.payload import ImageGenPayload
from hordelib.pipeline.payload_pp import FacefixPayload, UpscalePayload
from hordelib.pipeline.registry import audit_definition

ALL_DEFINITIONS: list[tuple[PipelineDefinition, tuple[type, ...]]] = [
    *((definition, (ImageGenPayload,)) for definition in IMAGE_PIPELINES),
    *((definition, (UpscalePayload, FacefixPayload)) for definition in POST_PROCESSING_REGISTRY.all_definitions()),
]


def _definition_ids() -> list[str]:
    return [definition.name for definition, _ in ALL_DEFINITIONS]


@pytest.mark.parametrize(("definition", "payload_types"), ALL_DEFINITIONS, ids=_definition_ids())
def test_graph_file_exists_and_loads(definition: PipelineDefinition, payload_types: tuple[type, ...]) -> None:
    assert definition.graph_file.exists(), f"Missing graph file {definition.graph_file}"
    graph = definition.load_graph()
    assert graph.node_titles(), f"{definition.name} loaded an empty graph"


@pytest.mark.parametrize(("definition", "payload_types"), ALL_DEFINITIONS, ids=_definition_ids())
def test_definition_audits_clean(definition: PipelineDefinition, payload_types: tuple[type, ...]) -> None:
    assert audit_definition(definition, payload_types=payload_types) == []


@pytest.mark.parametrize(("definition", "payload_types"), ALL_DEFINITIONS, ids=_definition_ids())
def test_extra_predicate_is_not_used(definition: PipelineDefinition, payload_types: tuple[type, ...]) -> None:
    """The escape hatch stays empty: express conditions as named features/selector fields.

    If a pipeline genuinely cannot express its condition declaratively, add it to an explicit
    allowlist here with a comment explaining why.
    """
    allowed_extra_predicate_users: set[str] = set()
    if definition.name in allowed_extra_predicate_users:
        return
    assert definition.selector.extra_predicate is None, (
        f"{definition.name} uses Selector.extra_predicate; prefer a named PayloadFeature or a "
        "typed selector field so the selection surface stays auditable"
    )


@pytest.mark.parametrize("definition", IMAGE_PIPELINES, ids=[d.name for d in IMAGE_PIPELINES])
def test_image_definition_materializes_strictly(definition: PipelineDefinition) -> None:
    """A canonical payload materializes without any binding missing its node."""
    payload = ImageGenPayload.from_horde_dict({"seed": 1, "prompt": "contract test"})
    context = ModelContext(
        horde_model_name="Contract Model",
        main_file="contract_model.safetensors",
        extra_files={
            "stable_cascade_stage_b": "stage_b.safetensors",
            "stable_cascade_stage_c": "stage_c.safetensors",
        },
    )

    graph = definition.materialize(payload, context)

    for output in definition.outputs:
        assert graph.has_node(output.node)


@pytest.mark.parametrize(
    ("payload", "expected_name"),
    [
        (UpscalePayload(model="RealESRGAN_x4plus", source_image=PIL.Image.new("RGB", (8, 8))), "image_upscale"),
        (FacefixPayload(model="GFPGAN", source_image=PIL.Image.new("RGB", (8, 8))), "image_facefix"),
    ],
    ids=["upscale", "facefix"],
)
def test_post_processing_definition_materializes_strictly(payload, expected_name: str) -> None:
    definition = POST_PROCESSING_REGISTRY.select(
        payload,
        PostProcessingContext(model_name=payload.model, model_file=f"{payload.model}.pth"),
    )
    assert definition is not None
    assert definition.name == expected_name

    graph = definition.materialize(
        payload,
        PostProcessingContext(model_name=payload.model, model_file=f"{payload.model}.pth"),
    )
    for output in definition.outputs:
        assert graph.has_node(output.node)
