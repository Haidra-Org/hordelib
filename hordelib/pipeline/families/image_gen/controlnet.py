"""The controlnet pipelines: guided generation, its hires-fix variant, and the annotator.

The annotator variant returns the preprocessed control map itself instead of a generation,
which is why its graph exposes only the source image loader; everything else is configured
by the controlnet patch step.
"""

from hordelib.pipeline.context import ModelContext
from hordelib.pipeline.definition import OutputSpec, PipelineDefinition, SelectionTier, pipeline_graph
from hordelib.pipeline.families.image_gen import bindings, features, steps
from hordelib.pipeline.families.image_gen.features import ImageSelector
from hordelib.pipeline.payload import ImageGenPayload

__all__ = [
    "CONTROLNET",
    "CONTROLNET_ANNOTATOR",
    "CONTROLNET_HIRES_FIX",
]

_CONTROLNET_BINDINGS = bindings.compose(
    bindings.SAMPLER_CORE,
    bindings.EMPTY_LATENT,
    bindings.PROMPTS,
    bindings.SEAMLESS_TILING,
    bindings.SOURCE_IMAGE,
    bindings.CONTROLNET,
)

_CONTROLNET_SPURIOUS = frozenset({"sampler.noise_seed"})
"""SAMPLER_CORE's grandfathered KSampler no-op (see the group's comment in bindings.py)."""

CONTROLNET_ANNOTATOR: PipelineDefinition[ImageGenPayload, ModelContext] = PipelineDefinition(
    name="controlnet_annotator",
    graph_file=pipeline_graph("controlnet_annotator"),
    selector=ImageSelector(
        tier=SelectionTier.FEATURE,
        order=0,
        features=(features.CONTROLNET, features.RETURN_CONTROL_MAP),
    ),
    bindings=bindings.compose(bindings.SOURCE_IMAGE),
    outputs=(OutputSpec(node="output_image"),),
    patch_steps=steps.IMAGE_PATCH_STEPS,
)

CONTROLNET_HIRES_FIX: PipelineDefinition[ImageGenPayload, ModelContext] = PipelineDefinition(
    name="controlnet_hires_fix",
    graph_file=pipeline_graph("controlnet_hires_fix"),
    selector=ImageSelector(
        tier=SelectionTier.FEATURE,
        order=1,
        features=(features.CONTROLNET, features.HIRES_FIX),
    ),
    bindings=bindings.compose(_CONTROLNET_BINDINGS, bindings.HIRES_FIX_UPSCALE),
    outputs=(OutputSpec(node="output_image"),),
    patch_steps=steps.IMAGE_PATCH_STEPS,
    known_spurious_inputs=_CONTROLNET_SPURIOUS,
)

CONTROLNET: PipelineDefinition[ImageGenPayload, ModelContext] = PipelineDefinition(
    name="controlnet",
    graph_file=pipeline_graph("controlnet"),
    selector=ImageSelector(tier=SelectionTier.FEATURE, order=2, features=(features.CONTROLNET,)),
    bindings=_CONTROLNET_BINDINGS,
    outputs=(OutputSpec(node="output_image"),),
    patch_steps=steps.IMAGE_PATCH_STEPS,
    known_spurious_inputs=_CONTROLNET_SPURIOUS,
)
