# test_horde.py
import pytest

from hordelib.horde import HordeLib, SharedModelManager


class TestHordeInit:
    horde = HordeLib()
    model_manager: SharedModelManager
    default_model_manager_args: dict

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        self.horde = HordeLib()

        self.model_manager = SharedModelManager()
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
        self.model_manager.loadModelManagers(**self.default_model_manager_args)
        assert self.model_manager.manager is not None
        self.model_manager.manager.load("Deliberate")
        yield
        del self.horde
        del self.model_manager

    def test_compvis(self):
        from hordelib.model_manager.compvis import CompVisModelManager

        CompVisModelManager()

    def test_horde_model_manager_init(self):
        assert self.model_manager.manager is not None
        assert self.model_manager.manager.blip is not None
        assert self.model_manager.manager.clip is not None
        assert self.model_manager.manager.codeformer is not None
        assert self.model_manager.manager.compvis is not None
        assert self.model_manager.manager.controlnet is not None
        assert self.model_manager.manager.diffusers is not None
        assert self.model_manager.manager.safety_checker is not None

    def test_horde_model_manager_reload_db(self):
        assert self.model_manager.manager is not None
        self.model_manager.manager.reload_database()

    def test_horde_model_manager_download_model(self):
        assert self.model_manager.manager is not None
        dlResult: bool | None = self.model_manager.manager.download_model("Deliberate")
        assert dlResult is True

    def test_horde_model_manager_validate(self):
        assert self.model_manager.manager is not None
        self.model_manager.manager.validate_model(
            "Deliberate"
        )  # XXX add a return value

    # XXX add a test for model missing
    def test_horde_model_manager_unload_model(self):
        assert self.model_manager.manager is not None
        model_unloaded = self.model_manager.manager.unload_model(
            "Deliberate"
        )  # XXX add a return value
        assert model_unloaded is True


class TestHordeInference:
    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        self.horde = HordeLib()

        model_manager = SharedModelManager()
        model_manager.loadModelManagers(compvis=True)
        assert model_manager.manager is not None
        model_manager.manager.load("Deliberate")
        yield
        del self.horde
        del model_manager

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
