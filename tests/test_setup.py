# test_setup.py
import pytest
from hordelib.comfy import Comfy


class TestSetup:
    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        self.comfy = Comfy()
        yield
        self.comfy = None

    def test_load_pipelines(self):
        loaded = self.comfy._load_pipelines()
        assert loaded == 1
        # Check the built in pipelines
        assert "stable_diffusion" in self.comfy.pipelines.keys()

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

        assert "HordeCheckpointLoader" in execution.nodes.NODE_CLASS_MAPPINGS.keys()
        assert "HordeImageOutput" in execution.nodes.NODE_CLASS_MAPPINGS.keys()

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
        }
        self.comfy._set(test_dict, **params)
        assert test_dict["a"]["inputs"]["b"]
        assert test_dict["c"]["inputs"]["d"]["e"]
        assert test_dict["c"]["inputs"]["d"]["f"]
