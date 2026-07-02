"""The schema-backed binding audit catches every half of a bad binding, loudly.

ComfyUI silently drops prompt inputs whose name a node class does not declare
(``ComfyUI/execution.py``, ``get_input_data``), so these violations would otherwise ship as
no-ops. Each test forges one specific mistake against a real packaged graph and asserts the
audit names it.
"""

import pytest
from pydantic import BaseModel

from hordelib.pipeline.context import ModelContext
from hordelib.pipeline.definition import (
    OutputSpec,
    ParamBinding,
    PipelineDefinition,
    SelectionTier,
    node,
    pipeline_graph,
    scaled,
)
from hordelib.pipeline.families.image_gen.features import ImageSelector
from hordelib.pipeline.payload import ImageGenPayload
from hordelib.pipeline.registry import audit_definition


class _MiniPayload(BaseModel):
    steps: int = 20


def _definition(
    bindings: tuple[ParamBinding, ...],
    *,
    known_spurious_inputs: frozenset[str] = frozenset(),
) -> PipelineDefinition[ImageGenPayload, ModelContext]:
    return PipelineDefinition(
        name="audit_probe",
        graph_file=pipeline_graph("stable_diffusion"),
        selector=ImageSelector(tier=SelectionTier.FALLBACK),
        bindings=bindings,
        outputs=(OutputSpec(node="output_image"),),
        known_spurious_inputs=known_spurious_inputs,
    )


def test_unknown_input_name_is_a_violation() -> None:
    definition = _definition(node("sampler", "KSampler").bind(denosie="denoising_strength"))
    violations = audit_definition(definition)
    assert any("declares no input 'denosie'" in violation for violation in violations)


def test_wrong_node_class_is_a_violation() -> None:
    definition = _definition(node("sampler", "KSamplerAdvanced").bind(steps="ddim_steps"))
    violations = audit_definition(definition)
    assert any("graph node is class 'KSampler'" in violation for violation in violations)


def test_unknown_payload_source_is_a_violation() -> None:
    definition = _definition(node("sampler", "KSampler").bind(steps="ddim_stepps"))
    violations = audit_definition(definition, payload_types=(ImageGenPayload,))
    assert any("payload field 'ddim_stepps'" in violation for violation in violations)
    # Without payload types the source half is not checked
    assert audit_definition(definition) == []


def test_grandfathered_spurious_input_is_accepted() -> None:
    definition = _definition(
        node("sampler", "KSampler").bind(noise_seed="seed"),
        known_spurious_inputs=frozenset({"sampler.noise_seed"}),
    )
    assert audit_definition(definition) == []


def test_stale_grandfather_entry_is_a_violation() -> None:
    definition = _definition(
        node("sampler", "KSampler").bind(steps="ddim_steps"),
        known_spurious_inputs=frozenset({"sampler.steps"}),
    )
    violations = audit_definition(definition)
    assert any("remove the stale grandfather entry" in violation for violation in violations)


def test_grandfather_entry_without_binding_is_a_violation() -> None:
    definition = _definition(
        node("sampler", "KSampler").bind(steps="ddim_steps"),
        known_spurious_inputs=frozenset({"sampler.noise_seed"}),
    )
    violations = audit_definition(definition)
    assert any("no matching binding" in violation for violation in violations)


def test_extra_inputs_are_audited_too() -> None:
    definition = PipelineDefinition[ImageGenPayload, ModelContext](
        name="audit_probe_extras",
        graph_file=pipeline_graph("stable_diffusion"),
        selector=ImageSelector(tier=SelectionTier.FALLBACK),
        bindings=node("sampler", "KSampler").bind(steps="ddim_steps"),
        outputs=(OutputSpec(node="output_image"),),
        extra_inputs={"model_loader.will_load_lorass": True},
    )
    violations = audit_definition(definition)
    assert any("will_load_lorass" in violation for violation in violations)


def test_undotted_target_is_a_violation() -> None:
    definition = _definition((ParamBinding(target="sampler", source="seed"),))
    violations = audit_definition(definition)
    assert any("not a dotted" in violation for violation in violations)


def test_node_requires_a_class() -> None:
    with pytest.raises(ValueError, match="must declare at least one expected class_type"):
        node("sampler")


def test_bind_rejects_invalid_values() -> None:
    with pytest.raises(TypeError, match="field name, transform, or scaled"):
        node("sampler", "KSampler").bind(steps=12)  # type: ignore[arg-type]


def test_scaled_builds_a_multiplied_source_binding() -> None:
    (binding,) = node("sampler", "KSampler").bind(steps=scaled("steps", 0.5))
    assert binding.source == "steps"
    assert binding.multiplier == 0.5
    assert binding.node_classes == ("KSampler",)
    assert binding.resolve(_MiniPayload(steps=21)) == 10  # round(21 * 0.5) banker's-rounds to 10
