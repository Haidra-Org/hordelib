"""The creative_upscale workflow: latent-space upscale of a source image plus a low-denoise
re-sample with the requested checkpoint.

Unlike the ESRGAN-style post-processors (pure pixel-space, no model), this re-diffuses the
image at the target resolution, so the checkpoint invents coherent new detail. The payload's
``denoising_strength`` is the fidelity/creativity knob: ~0.2 stays close to the source,
~0.5 repaints heavily. ``width``/``height`` are the target resolution.

This pipeline is also the reference for the post-refactor authoring surface: one graph JSON,
one module, node()-authored bindings with no grandfathered no-ops, and opt-in patch steps
(only the model loader and LoRA chain; no shared-step spray).
"""

from hordelib.pipeline.context import ModelContext
from hordelib.pipeline.definition import OutputSpec, PipelineDefinition, SelectionTier, node, pipeline_graph
from hordelib.pipeline.families.image_gen import bindings, steps
from hordelib.pipeline.families.image_gen.features import ImageSelector
from hordelib.pipeline.payload import ImageGenPayload

__all__ = ["CREATIVE_UPSCALE"]

CREATIVE_UPSCALE: PipelineDefinition[ImageGenPayload, ModelContext] = PipelineDefinition(
    name="creative_upscale",
    graph_file=pipeline_graph("creative_upscale"),
    # An explicit workflow request, like qr_code: the payload asks for this by name instead
    # of it being inferred from other fields.
    selector=ImageSelector(tier=SelectionTier.WORKFLOW_OVERRIDE, order=1, workflow="creative_upscale"),
    bindings=bindings.compose(
        node("sampler", "KSampler").bind(
            sampler_name=bindings.comfy_sampler,
            cfg="cfg_scale",
            denoise="denoising_strength",
            seed="seed",
            scheduler="scheduler",
            steps="ddim_steps",
        ),
        node("latent_upscale", "LatentUpscale").bind(
            width="width",
            height="height",
        ),
        bindings.PROMPTS,
        bindings.CLIP_SKIP,
        bindings.SOURCE_IMAGE,
        bindings.SEAMLESS_TILING,
    ),
    outputs=(OutputSpec(node="output_image"),),
    patch_steps=(steps.apply_lora_chain, steps.apply_main_model),
)
