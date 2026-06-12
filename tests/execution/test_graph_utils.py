"""Unit tests for the pure graph-manipulation helpers.

These run without a GPU, without models, and without ComfyUI.
"""

import copy
import json
from pathlib import Path

import pytest

from hordelib.execution.graph_utils import (
    apply_dotted_params,
    fix_node_names,
    fix_pipeline_types,
    reconnect_input,
)

PIPELINES_DIR = Path(__file__).parent.parent.parent / "hordelib" / "pipelines"


@pytest.fixture
def raw_sd_graph() -> dict:
    """The stable diffusion pipeline as shipped (numeric node names, _meta titles intact)."""
    return json.loads((PIPELINES_DIR / "pipeline_stable_diffusion.json").read_text(encoding="utf-8"))


@pytest.fixture
def named_sd_graph(raw_sd_graph: dict) -> dict:
    return fix_node_names(raw_sd_graph)


class TestFixNodeNames:
    def test_nodes_renamed_to_titles(self, raw_sd_graph: dict):
        graph = fix_node_names(raw_sd_graph)
        assert "sampler" in graph
        assert "model_loader" in graph
        assert "3" not in graph
        assert graph["sampler"]["class_type"] == "KSampler"

    def test_references_updated(self, raw_sd_graph: dict):
        graph = fix_node_names(raw_sd_graph)
        assert graph["sampler"]["inputs"]["model"][0] == "model_loader"
        assert graph["sampler"]["inputs"]["positive"][0] == "prompt"
        assert graph["sampler"]["inputs"]["negative"][0] == "negative_prompt"
        assert graph["vae_decode"]["inputs"]["samples"][0] == "sampler"

    def test_nodes_without_titles_keep_names(self):
        graph = {"1": {"class_type": "Foo", "inputs": {}}}
        renamed = fix_node_names(graph)
        assert "1" in renamed


class TestFixPipelineTypes:
    def test_class_types_replaced(self, raw_sd_graph: dict):
        graph = fix_pipeline_types(
            raw_sd_graph,
            {"CheckpointLoaderSimple": "HordeCheckpointLoader", "SaveImage": "HordeImageOutput"},
        )
        assert graph["4"]["class_type"] == "HordeCheckpointLoader"
        assert graph["9"]["class_type"] == "HordeImageOutput"
        # Unmapped types are untouched
        assert graph["3"]["class_type"] == "KSampler"

    def test_parameter_renames_remove_old_key(self):
        graph = {
            "1": {
                "class_type": "HordeCheckpointLoader",
                "inputs": {"ckpt_name": "foo.ckpt"},
            },
        }
        fix_pipeline_types(graph, {}, {"HordeCheckpointLoader": {"ckpt_name": "model_name"}})
        assert graph["1"]["inputs"] == {"model_name": "foo.ckpt"}


class TestApplyDottedParams:
    def test_sets_existing_params(self, named_sd_graph: dict):
        skipped = apply_dotted_params(named_sd_graph, {"sampler.steps": 33, "sampler.cfg": 4.5})
        assert skipped == 0
        assert named_sd_graph["sampler"]["inputs"]["steps"] == 33
        assert named_sd_graph["sampler"]["inputs"]["cfg"] == 4.5

    def test_explicit_inputs_segment(self, named_sd_graph: dict):
        apply_dotted_params(named_sd_graph, {"sampler.inputs.steps": 12})
        assert named_sd_graph["sampler"]["inputs"]["steps"] == 12

    def test_creates_missing_input_key(self, named_sd_graph: dict):
        apply_dotted_params(named_sd_graph, {"sampler.brand_new_param": "hello"})
        assert named_sd_graph["sampler"]["inputs"]["brand_new_param"] == "hello"

    def test_missing_node_is_skipped(self, named_sd_graph: dict):
        before = copy.deepcopy(named_sd_graph)
        skipped = apply_dotted_params(named_sd_graph, {"no_such_node.steps": 1})
        assert skipped == 1
        assert named_sd_graph == before


class TestReconnectInput:
    def test_reconnects_to_existing_output(self, named_sd_graph: dict):
        assert reconnect_input(named_sd_graph, "sampler.model", "layer_diffuse_apply") is True
        assert named_sd_graph["sampler"]["inputs"]["model"][0] == "layer_diffuse_apply"

    def test_missing_output_returns_none(self, named_sd_graph: dict):
        before = copy.deepcopy(named_sd_graph)
        assert reconnect_input(named_sd_graph, "sampler.model", "no_such_node") is None
        assert named_sd_graph == before

    def test_missing_input_returns_none(self, named_sd_graph: dict):
        assert reconnect_input(named_sd_graph, "sampler.no_such_input", "model_loader") is None

    def test_lora_chain_rewiring(self, named_sd_graph: dict):
        """Simulates the dynamic LoRA chain insertion done by HordeLib."""
        named_sd_graph["lora_0"] = {
            "class_type": "HordeLoraLoader",
            "inputs": {
                "model": ["model_loader", 0],
                "clip": ["model_loader", 1],
                "lora_name": "some_lora.safetensors",
                "strength_model": 1.0,
                "strength_clip": 1.0,
            },
        }
        assert reconnect_input(named_sd_graph, "sampler.model", "lora_0") is True
        assert reconnect_input(named_sd_graph, "clip_skip.clip", "lora_0") is True
        assert named_sd_graph["sampler"]["inputs"]["model"][0] == "lora_0"
        assert named_sd_graph["clip_skip"]["inputs"]["clip"][0] == "lora_0"
