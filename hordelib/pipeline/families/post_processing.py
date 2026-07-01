"""The post-processing pipeline family: upscaling and face restoration.

This is the first non-image-generation family and the template for future modalities: its own
payload types (:mod:`hordelib.pipeline.payload_pp`), its own lightweight selection context
(:class:`hordelib.pipeline.context.PostProcessingContext`), and its own registry.

``strip_background`` is intentionally absent: it is a pure-Python rembg call, not a ComfyUI
graph (see ``HordeLib.post_process``).
"""

from pathlib import Path

from hordelib.pipeline.context import PostProcessingContext
from hordelib.pipeline.definition import (
    OutputSpec,
    ParamBinding,
    PayloadFeature,
    PipelineDefinition,
    SelectionTier,
    Selector,
)
from hordelib.pipeline.graph import ComfyGraph
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
    bindings=(ParamBinding(target="image_loader.image", source="source_image"),),
    outputs=(OutputSpec(node="output_image"),),
    patch_steps=(_apply_model_file,),
)

IMAGE_FACEFIX_DEFINITION: PostProcessingDefinition = PipelineDefinition(
    name="image_facefix",
    graph_file=PIPELINES_DIR / "pipeline_image_facefix.json",
    selector=Selector(tier=SelectionTier.FEATURE, order=1, features=(FACEFIX_REQUESTED,)),
    bindings=(
        ParamBinding(target="image_loader.image", source="source_image"),
        ParamBinding(target="face_restore_with_model.codeformer_fidelity", source="fidelity"),
    ),
    outputs=(OutputSpec(node="output_image"),),
    patch_steps=(_apply_model_file,),
)


def build_post_processing_registry() -> PipelineRegistry[PostProcessingGraphPayload, PostProcessingContext]:
    registry: PipelineRegistry[PostProcessingGraphPayload, PostProcessingContext] = PipelineRegistry()
    registry.register(IMAGE_UPSCALE_DEFINITION)
    registry.register(IMAGE_FACEFIX_DEFINITION)
    return registry


POST_PROCESSING_REGISTRY = build_post_processing_registry()
