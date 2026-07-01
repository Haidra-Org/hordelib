"""The Flux pipeline: split sampling nodes, guidance-based CFG, no hires-fix variant."""

from hordelib.pipeline.context import ModelContext
from hordelib.pipeline.definition import OutputSpec, PipelineDefinition, SelectionTier, pipeline_graph
from hordelib.pipeline.families.image_gen import bindings, steps
from hordelib.pipeline.families.image_gen.baselines import FLUX_BASELINES
from hordelib.pipeline.families.image_gen.features import ImageSelector
from hordelib.pipeline.payload import ImageGenPayload

__all__ = ["FLUX"]

FLUX: PipelineDefinition[ImageGenPayload, ModelContext] = PipelineDefinition(
    name="flux",
    graph_file=pipeline_graph("flux"),
    selector=ImageSelector(tier=SelectionTier.BASELINE_FAMILY, order=3, baselines=FLUX_BASELINES),
    bindings=bindings.compose(
        bindings.EMPTY_LATENT,
        bindings.BATCH_REPEAT,
        bindings.PROMPTS,
        bindings.SEAMLESS_TILING,
        bindings.SOURCE_IMAGE,
        bindings.FLUX_SAMPLING,
    ),
    outputs=(OutputSpec(node="output_image"),),
    patch_steps=steps.IMAGE_PATCH_STEPS,
)
