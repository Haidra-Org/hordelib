"""The post-processing pipeline family: upscaling and face restoration.

This is the first non-image-generation family and the template for future modalities: its own
payload types (:mod:`hordelib.pipeline.payload_pp`), its own lightweight selection context
(:class:`hordelib.pipeline.context.PostProcessingContext`), and its own registry.

``strip_background`` is intentionally absent: it is a pure-Python rembg call, not a ComfyUI
graph (see ``HordeLib.post_process``).
"""

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from hordelib.pipeline.context import PostProcessingContext
from hordelib.pipeline.definition import (
    OutputSpec,
    PayloadFeature,
    PipelineDefinition,
    SelectionTier,
    Selector,
    node,
)
from hordelib.pipeline.graph import ComfyGraph, NodeRef
from hordelib.pipeline.payload_pp import FacefixPayload, PostProcessingGraphPayload, UpscalePayload
from hordelib.pipeline.registry import PipelineRegistry

PIPELINES_DIR = Path(__file__).parent.parent.parent / "pipelines"

type PostProcessingDefinition = PipelineDefinition[PostProcessingGraphPayload, PostProcessingContext]


def _is_upscale_payload(payload: PostProcessingGraphPayload) -> bool:
    return isinstance(payload, UpscalePayload)


def _is_facefix_payload(payload: PostProcessingGraphPayload) -> bool:
    return isinstance(payload, FacefixPayload)


UPSCALE_REQUESTED = PayloadFeature[PostProcessingGraphPayload](name="upscale_payload", is_set=_is_upscale_payload)
FACEFIX_REQUESTED = PayloadFeature[PostProcessingGraphPayload](name="facefix_payload", is_set=_is_facefix_payload)


def _apply_model_file(
    graph: ComfyGraph,
    payload: PostProcessingGraphPayload,
    context: PostProcessingContext,
) -> None:
    # The model file is an IO-resolved fact (PostProcessingContext), not payload intent,
    # so it is applied as a patch step rather than a payload binding.
    graph.set_input("model_loader.model_name", context.model_file)


IMAGE_UPSCALE_DEFINITION: PostProcessingDefinition = PipelineDefinition(
    name="image_upscale",
    graph_file=PIPELINES_DIR / "pipeline_image_upscale.json",
    selector=Selector(tier=SelectionTier.FEATURE, order=0, features=(UPSCALE_REQUESTED,)),
    bindings=node("image_loader", "HordeImageLoader").bind(image="source_image"),
    outputs=(OutputSpec(node="output_image"),),
    patch_steps=(_apply_model_file,),
)

IMAGE_FACEFIX_DEFINITION: PostProcessingDefinition = PipelineDefinition(
    name="image_facefix",
    graph_file=PIPELINES_DIR / "pipeline_image_facefix.json",
    selector=Selector(tier=SelectionTier.FEATURE, order=1, features=(FACEFIX_REQUESTED,)),
    bindings=(
        *node("image_loader", "HordeImageLoader").bind(image="source_image"),
        *node("face_restore_with_model", "FaceRestoreCFWithModel").bind(codeformer_fidelity="fidelity"),
        # The restorer replaces every detected face outright, so requested strength is honored as a
        # blend of its output back over the untouched input: at 1.0 the blend returns the restored
        # image bit-for-bit, at 0.0 the original. Doing it in the graph keeps one image encode and
        # works for GFPGAN too, whose architecture takes no strength of its own.
        *node("facefix_blend", "ImageBlend").bind(blend_factor="strength"),
    ),
    outputs=(OutputSpec(node="output_image"),),
    patch_steps=(_apply_model_file,),
)


def build_post_processing_registry() -> PipelineRegistry[PostProcessingGraphPayload, PostProcessingContext]:
    registry: PipelineRegistry[PostProcessingGraphPayload, PostProcessingContext] = PipelineRegistry(
        payload_types=(UpscalePayload, FacefixPayload),
    )
    registry.register(IMAGE_UPSCALE_DEFINITION)
    registry.register(IMAGE_FACEFIX_DEFINITION)
    return registry


POST_PROCESSING_REGISTRY = build_post_processing_registry()


@dataclass(frozen=True)
class ComposedPostProcessingChain:
    """A post-processing chain joined into one graph, so the image never leaves it between stages."""

    graph: ComfyGraph
    outputs: tuple[OutputSpec, ...]
    """The single output node of the last stage."""


@dataclass(frozen=True)
class _Stage:
    graph: ComfyGraph
    loader: str
    output: OutputSpec


def _materialize_stage(payload: PostProcessingGraphPayload, context: PostProcessingContext) -> _Stage:
    """Materialise one stage and find its image loader and output node from its definition.

    The loader is the bound node with no connected inputs, i.e. the node the payload's image enters
    through; the output is the definition's single declared output.
    """
    definition = POST_PROCESSING_REGISTRY.select(payload, context)
    if definition is None:
        raise ValueError(f"No post-processing definition matches {type(payload).__name__}")
    graph = definition.materialize(payload, context)
    bound_titles = {binding.target.split(".", 1)[0] for binding in definition.bindings}
    loaders = sorted(title for title in bound_titles if not graph.links_into(title))
    if len(loaders) != 1 or len(definition.outputs) != 1:
        raise ValueError(
            f"Definition {definition.name!r} does not have exactly one image loader and one output "
            f"(loaders {loaders}, outputs {[output.node for output in definition.outputs]})",
        )
    return _Stage(graph=graph, loader=loaders[0], output=definition.outputs[0])


def _stage_prefix(index: int) -> str:
    """The node title prefix of the stage at a zero-based execution position in a multi-stage chain."""
    return f"stage{index}_"


def _stage_result(graph: ComfyGraph, output_title: str) -> NodeRef:
    """The node output feeding a stage's output node."""
    sources = list(graph.links_into(output_title).values())
    if len(sources) != 1:
        raise ValueError(f"Output node {output_title!r} has {len(sources)} connected inputs, expected 1")
    return sources[0]


def compose_post_processing_chain(
    stages: Sequence[tuple[PostProcessingGraphPayload, PostProcessingContext]],
) -> ComposedPostProcessingChain:
    """Join ordered post-processing stages into one graph.

    Each stage is its registered definition materialised with its own payload and context. The first
    stage keeps its image loader; each later stage's loader is removed and its consumers are connected to
    the node that fed the previous stage's output node. Only the last stage keeps its output node.

    A single stage is returned exactly as materialised. With several stages every node title is prefixed
    with ``stage{index}_`` (zero-based execution position), e.g. ``stage1_model_loader``, so repeated
    operations of one kind stay distinct.

    Raises:
        ValueError: If ``stages`` is empty, or a definition does not fit the one loader, one output shape.
    """
    if not stages:
        raise ValueError("A post-processing chain needs at least one stage")
    materialized = [_materialize_stage(payload, context) for payload, context in stages]
    if len(materialized) == 1:
        only = materialized[0]
        return ComposedPostProcessingChain(graph=only.graph, outputs=(only.output,))

    composed = ComfyGraph({})
    first = materialized[0]
    previous_output = composed.graft(first.graph, prefix=_stage_prefix(0))[first.output.node]
    for index, stage in enumerate(materialized[1:], start=1):
        titles = composed.graft(stage.graph, prefix=_stage_prefix(index))
        previous_result = _stage_result(composed, previous_output)
        loader = titles[stage.loader]
        for target in composed.inputs_referencing(loader):
            composed.connect(target, previous_result)
        composed.remove_node(loader)
        composed.remove_node(previous_output)
        previous_output = titles[stage.output.node]

    last_output = dataclasses.replace(materialized[-1].output, node=previous_output)
    return ComposedPostProcessingChain(graph=composed, outputs=(last_output,))
