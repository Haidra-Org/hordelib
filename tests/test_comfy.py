# test_setup.py
import glob

from hordelib.horde import HordeLib


class TestComfyInterfaceCompatibility:
    def test_load_pipelines(self, hordelib_instance: HordeLib):
        loaded = hordelib_instance.generator._load_pipelines()
        assert loaded == len(glob.glob("hordelib/pipelines/*.json"))
        # Check the built in pipelines
        assert "stable_diffusion" in hordelib_instance.generator.pipelines
        assert "stable_diffusion_hires_fix" in hordelib_instance.generator.pipelines
        assert "image_upscale" in hordelib_instance.generator.pipelines
        assert "stable_diffusion_paint" in hordelib_instance.generator.pipelines
        assert "controlnet" in hordelib_instance.generator.pipelines

    def test_load_invalid_pipeline(self, hordelib_instance: HordeLib):
        loaded = hordelib_instance.generator._load_pipeline("no-such-pipeline")
        assert loaded is None

    def test_load_custom_nodes(self, hordelib_instance: HordeLib):
        hordelib_instance.generator._load_custom_nodes()

        # Look for our nodes in the ComfyUI nodes list
        import execution

        assert "HordeCheckpointLoader" in execution.nodes.NODE_CLASS_MAPPINGS
        assert "HordeImageOutput" in execution.nodes.NODE_CLASS_MAPPINGS
        assert "HordeImageLoader" in execution.nodes.NODE_CLASS_MAPPINGS

    def test_parameter_injection(self, hordelib_instance: HordeLib):
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
        hordelib_instance.generator._set(test_dict, **params)

        assert "a" in test_dict and isinstance(test_dict["a"], dict)
        assert "inputs" in test_dict["a"]
        assert "b" in test_dict["a"]["inputs"]
        assert "c" in test_dict
        assert isinstance(test_dict["c"], dict)
        assert "inputs" in test_dict["c"]
        assert "d" in test_dict["c"]["inputs"]
        assert "e" in test_dict["c"]["inputs"]["d"]
        assert "f" in test_dict["c"]["inputs"]["d"]

        assert test_dict["a"]["inputs"]["b"]
        assert test_dict["c"]["inputs"]["d"]["e"]
        assert test_dict["c"]["inputs"]["d"]["f"]
        assert "unknown.parameter" not in test_dict

    def test_fix_pipeline_types(self, hordelib_instance: HordeLib):
        data = {
            "node1": {"class_type": "ShouldNotBeReplaced"},
            "node2": {"no_class": "NoClassType"},
            "node3-should-be-replaced": {"class_type": "CheckpointLoaderSimple"},
        }
        data = hordelib_instance.generator._fix_pipeline_types(data)

        assert data["node1"]["class_type"] == "ShouldNotBeReplaced"
        assert data["node2"]["no_class"] == "NoClassType"
        assert data["node3-should-be-replaced"]["class_type"] == "HordeCheckpointLoader"

    def test_fix_node_names(self, hordelib_instance: HordeLib):
        # basically we are expecting a search and replace of "1" with the "title" of id 1, etc.
        data = {
            "1": {
                "inputs": {
                    "input1": ["2", 0],
                    "input2": ["3", 0],
                    "input3": "foo",
                    "input4": 33,
                    "input5": None,
                },
            },
            "2": {
                "inputs": {
                    "input1": ["3", 0],
                    "input2": ["1", 0],
                    "input3": "foo",
                    "input4": 33,
                    "input5": None,
                },
            },
            "3": {
                "inputs": {
                    "input1": ["2", 0],
                    "input2": ["1", 0],
                    "input3": "foo",
                    "input4": 33,
                    "input5": None,
                },
            },
        }
        design = {
            "nodes": [
                {"id": 1, "title": "Node1"},
                {"id": 2, "title": "Node2"},
                {"id": 3, "no_title": "Node3"},
            ],
        }
        data = hordelib_instance.generator._fix_node_names(data, design)

        assert data
        assert isinstance(data, dict)

        assert "Node1" in data
        assert "inputs" in data["Node1"] and isinstance(data["Node1"]["inputs"], dict)
        assert "input1" in data["Node1"]["inputs"] and isinstance(data["Node1"]["inputs"]["input1"], list)
        assert "input2" in data["Node1"]["inputs"] and isinstance(data["Node1"]["inputs"]["input2"], list)

        assert "Node2" in data
        assert "inputs" in data["Node2"] and isinstance(data["Node2"]["inputs"], dict)
        assert "input1" in data["Node2"]["inputs"] and isinstance(data["Node2"]["inputs"]["input1"], list)
        assert "input2" in data["Node2"]["inputs"] and isinstance(data["Node2"]["inputs"]["input2"], list)

        assert "3" in data
        assert "inputs" in data["3"] and isinstance(data["3"]["inputs"], dict)
        assert "input1" in data["3"]["inputs"] and isinstance(data["3"]["inputs"]["input1"], list)
        assert "input2" in data["3"]["inputs"] and isinstance(data["3"]["inputs"]["input2"], list)

        assert "Node1" in data
        assert data["Node1"]["inputs"]["input1"][0] == "Node2"
        assert data["Node1"]["inputs"]["input2"][0] == "3"
        assert "Node2" in data
        assert data["Node2"]["inputs"]["input1"][0] == "3"
        assert data["Node2"]["inputs"]["input2"][0] == "Node1"
        assert "3" in data
        assert data["3"]["inputs"]["input1"][0] == "Node2"
        assert data["3"]["inputs"]["input2"][0] == "Node1"

    def test_input_reconnection(self, hordelib_instance: HordeLib):
        # Can we reconnect the latent_image input of the sampler from the
        # empty_latent_image to the vae_encoder? And in the process
        # disconnect any existing connection that is already there?
        data = {
            "sampler": {
                "inputs": {
                    "seed": 760767020359210,
                    "steps": 20,
                    "cfg": 8.0,
                    "sampler_name": "euler",
                    "scheduler": "normal",
                    "denoise": 1.0,
                    "model": ["model_loader", 0],
                    "positive": ["prompt", 0],
                    "negative": ["negative_prompt", 0],
                    "latent_image": ["empty_latent_image", 0],
                },
                "class_type": "KSampler",
            },
            "vae_encoder": {
                "inputs": {"pixels": ["image_loader", 0], "vae": ["model_loader", 2]},
                "class_type": "VAEEncode",
            },
            "empty_latent_image": {
                "inputs": {"width": 512, "height": 512, "batch_size": 1},
                "class_type": "EmptyLatentImage",
            },
        }
        result = hordelib_instance.generator.reconnect_input(data, "sampler.latent_image", "vae_encoder")
        # Should be ok
        assert result

        assert "sampler" in data
        assert "inputs" in data["sampler"]
        assert "latent_image" in data["sampler"]["inputs"]
        assert isinstance(data["sampler"]["inputs"], dict)
        assert isinstance(data["sampler"]["inputs"]["latent_image"], list)

        assert data["sampler"]["inputs"]["latent_image"][0] == "vae_encoder"
        # This is invalid
        result = hordelib_instance.generator.reconnect_input(data, "sampler.non-existant", "somewhere")
        assert not result
