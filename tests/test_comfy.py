# test_comfy.py
import glob
from pathlib import Path

from hordelib.horde import HordeLib
from hordelib.pipeline.graph import ComfyGraph


class TestComfyInterfaceCompatibility:
    def test_load_all_pipeline_graphs(self, init_horde):
        """Every packaged pipeline JSON loads, with node replacement and title renaming applied."""
        pipeline_files = glob.glob("hordelib/pipelines/pipeline_*.json")
        assert pipeline_files

        names = {Path(f).stem.removeprefix("pipeline_") for f in pipeline_files}
        for expected in ("stable_diffusion", "stable_diffusion_hires_fix", "image_upscale", "controlnet"):
            assert expected in names

        for file in pipeline_files:
            graph = ComfyGraph.from_file(Path(file))
            assert graph.node_titles(), file
            # Title renaming: API exports key nodes by numeric id; loading keys them by title
            assert not all(title.isdigit() for title in graph.node_titles()), file

        sd_graph = ComfyGraph.from_file(Path("hordelib/pipelines/pipeline_stable_diffusion.json"))
        # Node replacement: comfyui standard loaders become horde loaders
        assert "HordeCheckpointLoader" in sd_graph.class_types()
        assert "CheckpointLoaderSimple" not in sd_graph.class_types()

    def test_load_custom_nodes(self, hordelib_instance: HordeLib):
        hordelib_instance.backend.comfy_horde._load_custom_nodes()

        # Look for our nodes in the ComfyUI nodes list
        import execution

        assert "HordeCheckpointLoader" in execution.nodes.NODE_CLASS_MAPPINGS
        assert "HordeImageOutput" in execution.nodes.NODE_CLASS_MAPPINGS
        assert "HordeImageLoader" in execution.nodes.NODE_CLASS_MAPPINGS

    def test_parameter_injection(self, init_horde):
        graph = ComfyGraph(
            {
                "a": {
                    "inputs": {"b": False},
                },
                "c": {"inputs": {"d": False}},
            },
        )

        skipped = graph.set_inputs(
            {
                "a.b": True,
                "c.d": True,
                "unknown.parameter": False,
            },
        )

        # Missing nodes are skipped, matching the legacy apply-where-present semantics
        assert skipped == 1
        assert graph.node("a")["inputs"]["b"] is True
        assert graph.node("c")["inputs"]["d"] is True
        assert not graph.has_node("unknown.parameter")
        assert not graph.has_node("unknown")
