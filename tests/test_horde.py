# test_horde.py
import pytest

from hordelib.horde import HordeLib, SharedModelManager


class TestSharedModelManager:
    horde = HordeLib()
    default_model_manager_args: dict

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        self.horde = HordeLib()

        self.default_model_manager_args = {
            # aitemplate
            "blip": True,
            "clip": True,
            "codeformer": True,
            "compvis": True,
            "controlnet": True,
            "diffusers": True,
            # "esrgan": True,
            # "gfpgan": True,
            "safety_checker": True,
        }
        SharedModelManager.loadModelManagers(**self.default_model_manager_args)
        assert SharedModelManager.manager is not None
        yield
        del self.horde
        SharedModelManager._instance = None
        SharedModelManager.manager = None

    def test_compvis(self):
        from hordelib.model_manager.compvis import CompVisModelManager

        CompVisModelManager()

    def test_horde_model_manager_init(self):
        assert SharedModelManager.manager is not None
        # assert SharedModelManager.manager.aitemplate is not None
        assert SharedModelManager.manager.blip is not None
        assert SharedModelManager.manager.clip is not None
        assert SharedModelManager.manager.codeformer is not None
        assert SharedModelManager.manager.compvis is not None
        assert SharedModelManager.manager.controlnet is not None
        assert SharedModelManager.manager.diffusers is not None
        assert SharedModelManager.manager.safety_checker is not None

    def test_horde_model_manager_reload_db(self):
        assert SharedModelManager.manager is not None
        SharedModelManager.manager.reload_database()

    def test_horde_model_manager_download_model(self):
        assert SharedModelManager.manager is not None
        result: bool | None = SharedModelManager.manager.download_model("Deliberate")
        assert result is True

    def test_horde_model_manager_validate(self):
        assert SharedModelManager.manager is not None
        SharedModelManager.manager.load("Deliberate")
        result: bool | None = SharedModelManager.manager.validate_model("Deliberate")
        assert result is True

    def test_taint_models(self):
        assert SharedModelManager.manager is not None
        SharedModelManager.manager.taint_models(["Deliberate"])
        assert "Deliberate" not in SharedModelManager.manager.get_available_models()
        assert "Deliberate" not in SharedModelManager.manager.get_loaded_models_names()

    # XXX add a test for model missing?
    def test_horde_model_manager_unload_model(self):
        assert SharedModelManager.manager is not None
        SharedModelManager.manager.load("Deliberate")
        assert "Deliberate" in SharedModelManager.manager.get_loaded_models_names()
        result = SharedModelManager.manager.unload_model("Deliberate")
        assert result is True
        assert "Deliberate" not in SharedModelManager.manager.get_loaded_models_names()


class TestHordeInference:
    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        self.horde = HordeLib()

        self.default_model_manager_args = {
            # aitemplate
            "blip": True,
            "clip": True,
            "codeformer": True,
            "compvis": True,
            "controlnet": True,
            "diffusers": True,
            # "esrgan": True,
            # "gfpgan": True,
            "safety_checker": True,
        }
        SharedModelManager.loadModelManagers(**self.default_model_manager_args)
        assert SharedModelManager.manager is not None
        SharedModelManager.manager.load("Deliberate")
        yield
        del self.horde
        SharedModelManager._instance = None
        SharedModelManager.manager = None

    def test_parameter_remap_simple(self):
        data = {
            "sampler_name": "k_lms",
            "cfg_scale": 5,
            "denoising_strength": 0.75,
            "seed": "23113",
            "height": 512,
            "width": 512,
            "karras": True,
            "tiling": False,
            "hires_fix": False,
            "clip_skip": 1,
            "control_type": "canny",
            "image_is_control": False,
            "return_control_map": False,
            "prompt": "a dog ### cat, mouse, lion",
            "ddim_steps": 30,
            "n_iter": 1,
            "model": "Deliberate",
        }

        expected = {
            "sampler.sampler_name": "lms",
            "sampler.cfg": 5,
            "sampler.denoise": 0.75,
            "sampler.seed": 23113,
            "empty_latent_image.height": 512,
            "empty_latent_image.width": 512,
            "sampler.steps": 30,
            "empty_latent_image.batch_size": 1,
            "model_loader.ckpt_name": "Deliberate",
            "sampler.scheduler": "karras",
            "prompt.text": "a dog",
            "negative_prompt.text": "cat, mouse, lion",
            "clip_skip.stop_at_clip_layer": -1,
        }
        assert self.horde is not None
        result = self.horde._parameter_remap(data)
        assert result == expected, f"Dictionaries don't match: {result} != {expected}"

    def test_parameter_remap_variation(self):
        data = {
            "sampler_name": "k_lms",
            "cfg_scale": 5,
            "denoising_strength": 0.75,
            "seed": "23113",
            "height": 512,
            "width": 512,
            "karras": False,
            "tiling": False,
            "hires_fix": False,
            "clip_skip": 1,
            "control_type": "canny",
            "image_is_control": False,
            "return_control_map": False,
            "prompt": "a dog",
            "ddim_steps": 30,
            "n_iter": 1,
            "model": "Deliberate",
        }

        expected = {
            "sampler.sampler_name": "lms",
            "sampler.cfg": 5,
            "sampler.denoise": 0.75,
            "sampler.seed": 23113,
            "empty_latent_image.height": 512,
            "empty_latent_image.width": 512,
            "sampler.steps": 30,
            "empty_latent_image.batch_size": 1,
            "model_loader.ckpt_name": "Deliberate",
            "sampler.scheduler": "normal",
            "prompt.text": "a dog",
            "negative_prompt.text": "",
            "clip_skip.stop_at_clip_layer": -1,
        }
        assert self.horde is not None
        result = self.horde._parameter_remap(data)
        assert result == expected, f"Dictionaries don't match: {result} != {expected}"

    def test_text_to_image(self):
        data = {
            "sampler_name": "k_dpmpp_2m",
            "cfg_scale": 7.5,
            "denoising_strength": 1.0,
            "seed": 123456789,
            "height": 512,
            "width": 512,
            "karras": True,
            "tiling": False,
            "hires_fix": False,
            "clip_skip": 1,
            "control_type": "canny",
            "image_is_control": False,
            "return_control_map": False,
            "prompt": "an ancient llamia monster",
            "ddim_steps": 25,
            "n_iter": 1,
            "model": "Deliberate",
        }
        assert self.horde is not None
        pil_image = self.horde.text_to_image(data)
        assert pil_image is not None
        pil_image.save("horde_text_to_image.png")

    def test_text_to_image_clip_skip_2(self):
        data = {
            "sampler_name": "k_dpmpp_2m",
            "cfg_scale": 7.5,
            "denoising_strength": 1.0,
            "seed": 123456789,
            "height": 512,
            "width": 512,
            "karras": True,
            "tiling": False,
            "hires_fix": False,
            "clip_skip": 2,
            "control_type": "canny",
            "image_is_control": False,
            "return_control_map": False,
            "prompt": "an ancient llamia monster",
            "ddim_steps": 25,
            "n_iter": 1,
            "model": "Deliberate",
        }
        assert self.horde is not None
        pil_image = self.horde.text_to_image(data)
        assert pil_image is not None
        pil_image.save("horde_text_to_image_clip_skip_2.png")
