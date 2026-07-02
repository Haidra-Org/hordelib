"""The qr_code workflow pipeline: layered QR composition over a standard generation."""

from hordelib.pipeline.context import ModelContext
from hordelib.pipeline.definition import OutputSpec, PipelineDefinition, SelectionTier, pipeline_graph
from hordelib.pipeline.families.image_gen import bindings, steps
from hordelib.pipeline.families.image_gen.features import ImageSelector
from hordelib.pipeline.payload import ImageGenPayload

__all__ = ["QR_CODE"]

QR_CODE: PipelineDefinition[ImageGenPayload, ModelContext] = PipelineDefinition(
    name="qr_code",
    graph_file=pipeline_graph("qr_code"),
    selector=ImageSelector(tier=SelectionTier.WORKFLOW_OVERRIDE, workflow="qr_code"),
    bindings=bindings.compose(
        # The qr graph's main "sampler" is a KSamplerAdvanced, unlike every other graph.
        bindings.QR_SAMPLER_CORE,
        bindings.PROMPTS,
        bindings.SEAMLESS_TILING,
        bindings.QR_LAYERS,
    ),
    # The legacy export titles its output node "Save Image" rather than the family-wide
    # "output_image" convention; retitling changes the loaded graph, so it waits for a
    # deliberate, GPU-verified export repair.
    outputs=(OutputSpec(node="Save Image"),),
    patch_steps=steps.IMAGE_PATCH_STEPS,
    # The export contains two nodes titled "Convert Mask to Image"; one is silently dropped
    # at load time (pre-existing behavior, pinned by the snapshot corpus). Repairing the
    # export needs GPU-oracle verification.
    known_title_collisions=frozenset({"Convert Mask to Image"}),
    # KSamplerAdvanced has no seed/denoise inputs (it takes noise_seed); these have always
    # been silently-dropped no-ops on this graph, pinned by the snapshot corpus.
    known_spurious_inputs=frozenset(
        {
            "sampler.seed",
            "sampler.denoise",
            "sampler_bg.seed",
            "sampler_bg.denoise",
            "sampler_fg.seed",
            "sampler_fg.denoise",
        },
    ),
)
