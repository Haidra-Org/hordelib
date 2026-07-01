"""The Stable Cascade pipelines: base two-stage generation, 2pass hires, and remix."""

from hordelib.pipeline.context import ModelContext
from hordelib.pipeline.definition import OutputSpec, PipelineDefinition, SelectionTier, pipeline_graph
from hordelib.pipeline.families.image_gen import bindings, features, steps
from hordelib.pipeline.families.image_gen.baselines import CASCADE_BASELINES
from hordelib.pipeline.families.image_gen.features import ImageSelector
from hordelib.pipeline.payload import ImageGenPayload

__all__ = [
    "STABLE_CASCADE",
    "STABLE_CASCADE_2PASS",
    "STABLE_CASCADE_REMIX",
]

STABLE_CASCADE_REMIX: PipelineDefinition[ImageGenPayload, ModelContext] = PipelineDefinition(
    name="stable_cascade_remix",
    graph_file=pipeline_graph("stable_cascade_remix"),
    selector=ImageSelector(
        tier=SelectionTier.BASELINE_FAMILY,
        order=0,
        baselines=CASCADE_BASELINES,
        features=(features.REMIX,),
    ),
    bindings=bindings.compose(
        bindings.PROMPTS,
        bindings.CASCADE_EMPTY_LATENT,
        bindings.CASCADE_REMIX_SOURCE_IMAGE,
        bindings.CASCADE_SAMPLERS,
    ),
    # Legacy export title (not the family-wide "output_image" convention); retitling changes
    # the loaded graph, so it waits for a deliberate, GPU-verified export repair.
    outputs=(OutputSpec(node="Save Image"),),
    patch_steps=steps.IMAGE_PATCH_STEPS,
)

STABLE_CASCADE_2PASS: PipelineDefinition[ImageGenPayload, ModelContext] = PipelineDefinition(
    name="stable_cascade_2pass",
    graph_file=pipeline_graph("stable_cascade_2pass"),
    selector=ImageSelector(
        tier=SelectionTier.BASELINE_FAMILY,
        order=1,
        baselines=CASCADE_BASELINES,
        features=(features.HIRES_FIX,),
    ),
    bindings=bindings.compose(
        bindings.EMPTY_LATENT,
        bindings.PROMPTS,
        bindings.CASCADE_SAMPLERS,
        bindings.CASCADE_SECOND_PASS,
    ),
    outputs=(OutputSpec(node="output_image"),),
    patch_steps=steps.IMAGE_PATCH_STEPS,
)

STABLE_CASCADE: PipelineDefinition[ImageGenPayload, ModelContext] = PipelineDefinition(
    name="stable_cascade",
    graph_file=pipeline_graph("stable_cascade"),
    selector=ImageSelector(tier=SelectionTier.BASELINE_FAMILY, order=2, baselines=CASCADE_BASELINES),
    bindings=bindings.compose(
        bindings.EMPTY_LATENT,
        bindings.BATCH_REPEAT,
        bindings.PROMPTS,
        bindings.CASCADE_SOURCE_IMAGE,
        bindings.CASCADE_SAMPLERS,
    ),
    # Legacy export title (not the family-wide "output_image" convention); see the remix note.
    outputs=(OutputSpec(node="save_image"),),
    patch_steps=steps.IMAGE_PATCH_STEPS,
)
