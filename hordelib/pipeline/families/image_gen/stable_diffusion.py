"""The generic Stable Diffusion pipelines: the fallback graph and its variants.

These serve every checkpoint-style baseline without a dedicated family (SD1/SD2/SDXL and
unknown custom models): plain txt2img/img2img, the hires-fix two-pass variant, and the
masked/inpainting graphs.
"""

from hordelib.pipeline.context import ModelContext
from hordelib.pipeline.definition import OutputSpec, PipelineDefinition, SelectionTier, pipeline_graph
from hordelib.pipeline.families.image_gen import bindings, features, steps
from hordelib.pipeline.families.image_gen.features import ImageSelector
from hordelib.pipeline.payload import ImageGenPayload

__all__ = [
    "STABLE_DIFFUSION",
    "STABLE_DIFFUSION_HIRES_FIX",
    "STABLE_DIFFUSION_IMG2IMG_MASK",
    "STABLE_DIFFUSION_PAINT",
]

_SD_BINDINGS = bindings.compose(
    bindings.SAMPLER_CORE,
    bindings.EMPTY_LATENT,
    bindings.BATCH_REPEAT,
    bindings.CLIP_SKIP,
    bindings.PROMPTS,
    bindings.SEAMLESS_TILING,
    bindings.SOURCE_IMAGE,
)

STABLE_DIFFUSION: PipelineDefinition[ImageGenPayload, ModelContext] = PipelineDefinition(
    name="stable_diffusion",
    graph_file=pipeline_graph("stable_diffusion"),
    selector=ImageSelector(tier=SelectionTier.FALLBACK),
    bindings=_SD_BINDINGS,
    outputs=(OutputSpec(node="output_image"),),
    patch_steps=steps.IMAGE_PATCH_STEPS,
)

STABLE_DIFFUSION_HIRES_FIX: PipelineDefinition[ImageGenPayload, ModelContext] = PipelineDefinition(
    name="stable_diffusion_hires_fix",
    graph_file=pipeline_graph("stable_diffusion_hires_fix"),
    selector=ImageSelector(tier=SelectionTier.GENERIC_VARIANT, features=(features.HIRES_FIX,)),
    bindings=bindings.compose(_SD_BINDINGS, bindings.HIRES_FIX_UPSCALE),
    outputs=(OutputSpec(node="output_image"),),
    patch_steps=steps.IMAGE_PATCH_STEPS,
)

STABLE_DIFFUSION_IMG2IMG_MASK: PipelineDefinition[ImageGenPayload, ModelContext] = PipelineDefinition(
    name="stable_diffusion_img2img_mask",
    graph_file=pipeline_graph("stable_diffusion_img2img_mask"),
    selector=ImageSelector(tier=SelectionTier.PAINTING, order=0, features=(features.IMG2IMG_MASK,)),
    bindings=_SD_BINDINGS,
    outputs=(OutputSpec(node="output_image"),),
    patch_steps=steps.IMAGE_PATCH_STEPS,
)

STABLE_DIFFUSION_PAINT: PipelineDefinition[ImageGenPayload, ModelContext] = PipelineDefinition(
    name="stable_diffusion_paint",
    graph_file=pipeline_graph("stable_diffusion_paint"),
    selector=ImageSelector(tier=SelectionTier.PAINTING, order=1, features=(features.PAINTING,)),
    bindings=_SD_BINDINGS,
    outputs=(OutputSpec(node="output_image"),),
    patch_steps=steps.IMAGE_PATCH_STEPS,
)
