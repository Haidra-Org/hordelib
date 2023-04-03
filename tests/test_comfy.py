# test_setup.py
import pytest

from hordelib.comfy import Comfy


class TestSetup:
    NUMBER_OF_PIPELINES = 2

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        self.comfy = Comfy()
        yield
        self.comfy = None

    def test_load_pipelines(self):
        loaded = self.comfy._load_pipelines()
        assert loaded == TestSetup.NUMBER_OF_PIPELINES
        # Check the built in pipelines
        assert "stable_diffusion" in self.comfy.pipelines
        assert "stable_diffusion_hires_fix" in self.comfy.pipelines

    def test_load_invalid_pipeline(self):
        loaded = self.comfy._load_pipeline("no-such-pipeline")
        assert loaded is None

    def test_load_invalid_node(self, capsys):
        self.comfy._load_node("no-such-node")
        captured = capsys.readouterr()
        assert "No such file or directory" in str(captured)

    def test_load_custom_nodes(self):
        self.comfy._load_custom_nodes()

        # Look for our nodes in the ComfyUI nodes list
        from hordelib.ComfyUI import execution

        assert "HordeCheckpointLoader" in execution.nodes.NODE_CLASS_MAPPINGS
        assert "HordeImageOutput" in execution.nodes.NODE_CLASS_MAPPINGS

    def test_parameter_injection(self):
        test_dict = {
            "a": {
                "inputs": {"b": False},
            },
            "c": {"inputs": {"d": {"e": False, "f": False}}},
        }

        params = {
            "a.b": True,
            "c.d.e": True,
            "c.inputs.d.f": True,
            "unknown.parameter": False,
        }
        self.comfy._set(test_dict, **params)
        assert test_dict["a"]["inputs"]["b"]
        assert test_dict["c"]["inputs"]["d"]["e"]
        assert test_dict["c"]["inputs"]["d"]["f"]
        assert "unknown.parameter" not in test_dict

    def test_fix_pipeline_types(self):
        data = {
            "node1": {"class_type": "ShouldNotBeReplaced"},
            "node2": {"no_class": "NoClassType"},
            "node3-should-be-replaced": {"class_type": "CheckpointLoaderSimple"},
        }
        data = self.comfy._fix_pipeline_types(data)

        assert data["node1"]["class_type"] == "ShouldNotBeReplaced"
        assert data["node2"]["no_class"] == "NoClassType"
        assert data["node3-should-be-replaced"]["class_type"] == "HordeCheckpointLoader"

    def test_fix_node_names(self):
        # basically we are expecting a search and replace of "1" with the "title" of id 1, etc.
        data = {
            "1": {
                "inputs": {
                    "input1": ["2", 0],
                    "input2": ["3", 0],
                    "input3": "foo",
                    "input4": 33,
                    "input5": None,
                }
            },
            "2": {
                "inputs": {
                    "input1": ["3", 0],
                    "input2": ["1", 0],
                    "input3": "foo",
                    "input4": 33,
                    "input5": None,
                }
            },
            "3": {
                "inputs": {
                    "input1": ["2", 0],
                    "input2": ["1", 0],
                    "input3": "foo",
                    "input4": 33,
                    "input5": None,
                }
            },
        }
        design = {
            "nodes": [
                {"id": 1, "title": "Node1"},
                {"id": 2, "title": "Node2"},
                {"id": 3, "no_title": "Node3"},
            ]
        }
        data = self.comfy._fix_node_names(data, design)

        assert "Node1" in data
        assert data["Node1"]["inputs"]["input1"][0] == "Node2"
        assert data["Node1"]["inputs"]["input2"][0] == "3"
        assert "Node2" in data
        assert data["Node2"]["inputs"]["input1"][0] == "3"
        assert data["Node2"]["inputs"]["input2"][0] == "Node1"
        assert "3" in data
        assert data["3"]["inputs"]["input1"][0] == "Node2"
        assert data["3"]["inputs"]["input2"][0] == "Node1"
