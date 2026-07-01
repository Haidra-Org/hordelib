"""The reference pipeline definition ``docs/adding-a-pipeline.md`` walks through line by line.

This is a complete, registrable image pipeline built against the trivial txt2img graph in
``example_pipeline.json``. It is registered only by tests (never by hordelib itself), so it
can be edited freely to experiment with the authoring surface.
"""

from pathlib import Path

from hordelib.pipeline.context import ModelContext
from hordelib.pipeline.definition import OutputSpec, PipelineDefinition, SelectionTier
from hordelib.pipeline.families.image_gen import bindings, steps
from hordelib.pipeline.families.image_gen.features import ImageSelector
from hordelib.pipeline.payload import ImageGenPayload

# A real pipeline uses pipeline_graph("<name>") to reference hordelib/pipelines/; the example
# graph lives next to this module instead so the packaged set stays clean.
EXAMPLE_GRAPH = Path(__file__).parent / "example_pipeline.json"

EXAMPLE_PIPELINE: PipelineDefinition[ImageGenPayload, ModelContext] = PipelineDefinition(
    # The unique name; selection tests and logs refer to it.
    name="example_txt2img",
    # The pure ComfyUI API export. Node _meta.title values are the parameter namespace.
    graph_file=EXAMPLE_GRAPH,
    # When this pipeline is chosen. Tiers replace magic priorities; a real pipeline would
    # select on a baseline set or named payload features (see families/image_gen/features.py).
    selector=ImageSelector(tier=SelectionTier.FALLBACK),
    # The complete parameter surface: what user payloads can set on this graph. Composed
    # from the named groups in families/image_gen/bindings.py; every target is audited
    # against the graph at registration, so a typo fails immediately.
    bindings=bindings.compose(
        bindings.SAMPLER_CORE,
        bindings.EMPTY_LATENT,
        bindings.PROMPTS,
    ),
    # Where results come from. The node must exist and be an output class for its kind.
    outputs=(OutputSpec(node="output_image"),),
    # Context-dependent graph mutation, opted into per pipeline. This example only needs
    # the resolved model file applied; real pipelines pick from families/image_gen/steps.py.
    patch_steps=(steps.apply_main_model,),
)
