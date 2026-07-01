"""The tutorial's reference pipeline registers, audits, selects, and materializes.

This is the executable proof behind ``docs/adding-a-pipeline.md``: if the authoring surface
drifts, this file fails before the tutorial can go stale silently.
"""

from hordelib.pipeline.context import ModelContext
from hordelib.pipeline.payload import ImageGenPayload
from hordelib.pipeline.registry import PipelineRegistry, audit_definition

from .examples.example_definition import EXAMPLE_PIPELINE


def test_example_definition_audits_clean() -> None:
    assert audit_definition(EXAMPLE_PIPELINE) == []


def test_example_definition_registers_and_selects() -> None:
    registry: PipelineRegistry[ImageGenPayload, ModelContext] = PipelineRegistry()
    registry.register(EXAMPLE_PIPELINE)

    selected = registry.select(
        ImageGenPayload.from_horde_dict({"seed": 1}),
        ModelContext(horde_model_name="Example Model"),
    )
    assert selected is EXAMPLE_PIPELINE


def test_example_definition_materializes_the_declared_surface() -> None:
    payload = ImageGenPayload.from_horde_dict(
        {"seed": 42, "prompt": "a lighthouse", "ddim_steps": 12, "cfg_scale": 5.5, "width": 640},
    )
    context = ModelContext(horde_model_name="Example Model", main_file="example.safetensors")

    graph = EXAMPLE_PIPELINE.materialize(payload, context)

    sampler_inputs = graph.node("sampler")["inputs"]
    assert sampler_inputs["seed"] == 42
    assert sampler_inputs["steps"] == 12
    assert sampler_inputs["cfg"] == 5.5
    assert graph.node("prompt")["inputs"]["text"] == "a lighthouse"
    assert graph.node("empty_latent_image")["inputs"]["width"] == 640
    # The patch step applied the resolved model file (a context fact, not payload intent)
    assert graph.node("model_loader")["inputs"]["model_name"] == "example.safetensors"
    # Horde node replacement happened at load time
    assert graph.node("output_image")["class_type"] == "HordeImageOutput"
