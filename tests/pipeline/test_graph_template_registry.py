"""ComfyGraph, PipelineDefinition, and PipelineRegistry behavior. No GPU required."""

from pathlib import Path

import pytest

from hordelib.pipeline.context import ModelContext
from hordelib.pipeline.definition import (
    OutputSpec,
    ParamBinding,
    PayloadFeature,
    PipelineDefinition,
    SelectionTier,
    Selector,
)
from hordelib.pipeline.graph import ComfyGraph, NodeRef
from hordelib.pipeline.payload import ImageGenPayload
from hordelib.pipeline.registry import PipelineRegistry, audit_definition

ANY_CONTEXT = ModelContext(horde_model_name="test")

PIPELINES_DIR = Path(__file__).parent.parent.parent / "hordelib" / "pipelines"
SD_PIPELINE = PIPELINES_DIR / "pipeline_stable_diffusion.json"

HIRES_FIX = PayloadFeature[ImageGenPayload](name="hires_fix", is_set=lambda p: p.hires_fix)


class TestComfyGraph:
    def test_from_file_applies_titles_and_replacements(self):
        graph = ComfyGraph.from_file(SD_PIPELINE)
        assert graph.has_node("sampler")
        assert graph.node("model_loader")["class_type"] == "HordeCheckpointLoader"
        assert graph.node("output_image")["class_type"] == "HordeImageOutput"

    def test_set_input(self):
        graph = ComfyGraph.from_file(SD_PIPELINE)
        graph.set_input("sampler.steps", 42)
        assert graph.node("sampler")["inputs"]["steps"] == 42

    def test_set_input_missing_node_raises(self):
        graph = ComfyGraph.from_file(SD_PIPELINE)
        with pytest.raises(KeyError):
            graph.set_input("no_such_node.steps", 1)

    def test_connect_and_add_node(self):
        graph = ComfyGraph.from_file(SD_PIPELINE)
        graph.add_node(
            "lora_0",
            "HordeLoraLoader",
            {
                "model": NodeRef("model_loader", 0),
                "clip": NodeRef("model_loader", 1),
                "lora_name": "x.safetensors",
                "strength_model": 1.0,
                "strength_clip": 1.0,
            },
        )
        graph.connect("sampler.model", "lora_0")
        graph.connect("clip_skip.clip", NodeRef("lora_0", 1))
        assert graph.node("sampler")["inputs"]["model"] == ["lora_0", 0]
        assert graph.node("clip_skip")["inputs"]["clip"] == ["lora_0", 1]
        assert graph.node("lora_0")["inputs"]["model"] == ["model_loader", 0]

    def test_add_duplicate_node_raises(self):
        graph = ComfyGraph.from_file(SD_PIPELINE)
        with pytest.raises(ValueError, match="already exists"):
            graph.add_node("sampler", "KSampler", {})

    def test_copy_is_independent(self):
        graph = ComfyGraph.from_file(SD_PIPELINE)
        clone = graph.copy()
        clone.set_input("sampler.steps", 99)
        assert graph.node("sampler")["inputs"]["steps"] != 99

    def test_class_types(self):
        graph = ComfyGraph.from_file(SD_PIPELINE)
        assert "KSampler" in graph.class_types()
        assert "HordeCheckpointLoader" in graph.class_types()


def _definition(
    name: str,
    *,
    selector: Selector[ImageGenPayload, ModelContext] | None = None,
    bindings: tuple[ParamBinding[ImageGenPayload], ...] = (),
    patch_steps: tuple = (),
) -> PipelineDefinition[ImageGenPayload, ModelContext]:
    return PipelineDefinition(
        name=name,
        graph_file=SD_PIPELINE,
        selector=selector if selector is not None else Selector(tier=SelectionTier.FALLBACK),
        bindings=bindings,
        outputs=(OutputSpec(node="output_image"),),
        patch_steps=patch_steps,
    )


class TestPipelineDefinition:
    def _bound_definition(self) -> PipelineDefinition[ImageGenPayload, ModelContext]:
        return _definition(
            "stable_diffusion",
            bindings=(
                ParamBinding(target="sampler.steps", source="ddim_steps"),
                ParamBinding(target="sampler.cfg", source="cfg_scale"),
                ParamBinding(target="empty_latent_image.width", source="width"),
                ParamBinding(target="prompt.text", source="prompt"),
                ParamBinding(target="sampler.half_steps", source="ddim_steps", multiplier=0.5),
                ParamBinding(target="sampler.double_cfg", transform=lambda p: p.cfg_scale * 2),
            ),
        )

    def test_materialize_applies_bindings(self):
        payload = ImageGenPayload.from_horde_dict(
            {"ddim_steps": 21, "cfg_scale": 5.0, "width": 640, "prompt": "a test"},
        )
        graph = self._bound_definition().materialize(payload, ANY_CONTEXT)
        sampler_inputs = graph.node("sampler")["inputs"]
        assert sampler_inputs["steps"] == 21
        assert sampler_inputs["cfg"] == 5.0
        assert sampler_inputs["half_steps"] == 10  # round(21 * 0.5) banker's rounding
        assert sampler_inputs["double_cfg"] == 10.0
        assert graph.node("empty_latent_image")["inputs"]["width"] == 640
        assert graph.node("prompt")["inputs"]["text"] == "a test"

    def test_materialize_runs_patch_steps(self):
        def add_marker(graph, payload, context):
            graph.set_input("sampler.marker", payload.n_iter)

        definition = _definition("patched", patch_steps=(add_marker,))
        graph = definition.materialize(ImageGenPayload.from_horde_dict({"n_iter": 3}), ANY_CONTEXT)
        assert graph.node("sampler")["inputs"]["marker"] == 3

    def test_binding_requires_exactly_one_source(self):
        with pytest.raises(ValueError):
            ParamBinding(target="x.y")
        with pytest.raises(ValueError):
            ParamBinding(target="x.y", source="width", transform=lambda p: 1)

    def test_materialize_returns_fresh_graph(self):
        definition = self._bound_definition()
        payload = ImageGenPayload.from_horde_dict({"ddim_steps": 11})
        first = definition.materialize(payload, ANY_CONTEXT)
        second = definition.materialize(ImageGenPayload.from_horde_dict({"ddim_steps": 22}), ANY_CONTEXT)
        assert first.node("sampler")["inputs"]["steps"] == 11
        assert second.node("sampler")["inputs"]["steps"] == 22


class TestPipelineRegistry:
    def test_tier_order(self):
        registry: PipelineRegistry[ImageGenPayload, ModelContext] = PipelineRegistry()
        registry.register(_definition("generic"))
        registry.register(
            _definition(
                "hires",
                selector=Selector(tier=SelectionTier.GENERIC_VARIANT, features=(HIRES_FIX,)),
            ),
        )

        selected = registry.select(ImageGenPayload.from_horde_dict({"hires_fix": True}), ANY_CONTEXT)
        assert selected is not None and selected.name == "hires"

        selected = registry.select(ImageGenPayload.from_horde_dict({}), ANY_CONTEXT)
        assert selected is not None and selected.name == "generic"

    def test_order_breaks_ties_within_a_tier(self):
        registry: PipelineRegistry[ImageGenPayload, ModelContext] = PipelineRegistry()
        # Registered out of order; the `order` field must decide, not registration order.
        registry.register(
            _definition("second", selector=Selector(tier=SelectionTier.FEATURE, order=1, features=(HIRES_FIX,))),
        )
        registry.register(
            _definition("first", selector=Selector(tier=SelectionTier.FEATURE, order=0, features=(HIRES_FIX,))),
        )

        selected = registry.select(ImageGenPayload.from_horde_dict({"hires_fix": True}), ANY_CONTEXT)
        assert selected is not None and selected.name == "first"

    def test_no_match_returns_none(self):
        registry: PipelineRegistry[ImageGenPayload, ModelContext] = PipelineRegistry()
        registry.register(
            _definition("hires", selector=Selector(tier=SelectionTier.GENERIC_VARIANT, features=(HIRES_FIX,))),
        )
        assert registry.select(ImageGenPayload.from_horde_dict({}), ANY_CONTEXT) is None

    def test_duplicate_name_rejected(self):
        registry: PipelineRegistry[ImageGenPayload, ModelContext] = PipelineRegistry()
        registry.register(_definition("a"))
        with pytest.raises(ValueError, match="already registered"):
            registry.register(_definition("a"))

    def test_unbound_binding_target_fails_registration(self):
        definition = _definition(
            "typo",
            bindings=(
                ParamBinding(target="no_such_node.steps", source="ddim_steps"),
                ParamBinding(target="sampler.steps", source="ddim_steps"),
            ),
        )
        assert any("no_such_node.steps" in violation for violation in audit_definition(definition))

        registry: PipelineRegistry[ImageGenPayload, ModelContext] = PipelineRegistry()
        with pytest.raises(ValueError, match="no_such_node.steps"):
            registry.register(definition)

    def test_missing_output_node_fails_audit(self):
        definition = PipelineDefinition(
            name="bad_output",
            graph_file=SD_PIPELINE,
            selector=Selector(tier=SelectionTier.FALLBACK),
            bindings=(),
            outputs=(OutputSpec(node="no_such_output"),),
        )
        assert any("no_such_output" in violation for violation in audit_definition(definition))

    def test_non_fallback_selector_without_criteria_fails_audit(self):
        definition = PipelineDefinition(
            name="shadow_everything",
            graph_file=SD_PIPELINE,
            selector=Selector(tier=SelectionTier.FEATURE),
            bindings=(),
            outputs=(OutputSpec(node="output_image"),),
        )
        assert any("no criteria" in violation for violation in audit_definition(definition))

    def test_sound_definition_passes_audit(self):
        assert audit_definition(_definition("sound")) == []
