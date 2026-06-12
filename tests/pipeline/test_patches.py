"""Golden-graph tests for the pure patch operations. No GPU required."""

import json
from pathlib import Path

from hordelib.execution.graph_utils import fix_node_names
from hordelib.pipeline.patches import ResolvedLora, insert_lora_chain

PIPELINES_DIR = Path(__file__).parent.parent.parent / "hordelib" / "pipelines"


def _sd_graph() -> dict:
    raw = json.loads((PIPELINES_DIR / "pipeline_stable_diffusion.json").read_text(encoding="utf-8"))
    return fix_node_names(raw)


def _flux_graph() -> dict:
    raw = json.loads((PIPELINES_DIR / "pipeline_flux.json").read_text(encoding="utf-8"))
    return fix_node_names(raw)


class TestInsertLoraChain:
    def test_no_loras_is_a_noop(self):
        graph = _sd_graph()
        before = json.dumps(graph, default=str, sort_keys=True)
        insert_lora_chain(graph, [])
        assert json.dumps(graph, default=str, sort_keys=True) == before

    def test_single_lora(self):
        graph = _sd_graph()
        insert_lora_chain(graph, [ResolvedLora(filename="a.safetensors", strength_model=0.8, strength_clip=0.7)])

        assert graph["lora_0"]["class_type"] == "HordeLoraLoader"
        assert graph["lora_0"]["inputs"]["model"] == ["model_loader", 0]
        assert graph["lora_0"]["inputs"]["clip"] == ["model_loader", 1]
        assert graph["lora_0"]["inputs"]["strength_model"] == 0.8
        assert graph["sampler"]["inputs"]["model"][0] == "lora_0"
        assert graph["clip_skip"]["inputs"]["clip"][0] == "lora_0"

    def test_three_lora_chain(self):
        graph = _sd_graph()
        loras = [ResolvedLora(filename=f"l{i}.safetensors", strength_model=1.0, strength_clip=1.0) for i in range(3)]
        insert_lora_chain(graph, loras)

        assert graph["lora_0"]["inputs"]["model"] == ["model_loader", 0]
        assert graph["lora_1"]["inputs"]["model"] == ["lora_0", 0]
        assert graph["lora_2"]["inputs"]["model"] == ["lora_1", 0]
        assert graph["lora_1"]["inputs"]["clip"] == ["lora_0", 1]
        # Only the last lora feeds the consumers
        assert graph["sampler"]["inputs"]["model"][0] == "lora_2"
        assert graph["clip_skip"]["inputs"]["clip"][0] == "lora_2"

    def test_missing_upscale_sampler_is_skipped(self):
        graph = _sd_graph()
        assert "upscale_sampler" not in graph
        insert_lora_chain(graph, [ResolvedLora(filename="a", strength_model=1, strength_clip=1)])
        assert "upscale_sampler" not in graph  # no phantom node created

    def test_flux_targets(self):
        graph = _flux_graph()
        insert_lora_chain(
            graph,
            [ResolvedLora(filename="a.safetensors", strength_model=1.0, strength_clip=1.0)],
            flux=True,
        )
        assert graph["cfg_guider"]["inputs"]["model"][0] == "lora_0"
        assert graph["basic_scheduler"]["inputs"]["model"][0] == "lora_0"
        # The non-flux targets must be untouched
        assert "sampler" not in graph or graph["sampler"]["inputs"].get("model", [None])[0] != "lora_0"
