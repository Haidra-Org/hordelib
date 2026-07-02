"""The Z-Image-Turbo pipeline: split-files loading on the standard sampler."""

from hordelib.pipeline.context import ModelContext
from hordelib.pipeline.definition import OutputSpec, PipelineDefinition, SelectionTier, pipeline_graph
from hordelib.pipeline.families.image_gen import bindings, steps
from hordelib.pipeline.families.image_gen.baselines import Z_IMAGE_BASELINES
from hordelib.pipeline.families.image_gen.features import ImageSelector
from hordelib.pipeline.payload import ImageGenPayload

__all__ = ["Z_IMAGE"]

Z_IMAGE: PipelineDefinition[ImageGenPayload, ModelContext] = PipelineDefinition(
    name="z_image",
    graph_file=pipeline_graph("z_image"),
    selector=ImageSelector(tier=SelectionTier.BASELINE_FAMILY, order=5, baselines=Z_IMAGE_BASELINES),
    bindings=bindings.compose(
        bindings.SAMPLER_CORE,
        bindings.EMPTY_LATENT,
        bindings.PROMPTS,
        bindings.SEAMLESS_TILING,
    ),
    outputs=(OutputSpec(node="output_image"),),
    patch_steps=steps.IMAGE_PATCH_STEPS,
    # SAMPLER_CORE's grandfathered KSampler no-op (see the group's comment in bindings.py).
    known_spurious_inputs=frozenset({"sampler.noise_seed"}),
)
