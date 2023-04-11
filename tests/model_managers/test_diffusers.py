import pytest

import hordelib
import hordelib.consts as consts


class TestDiffusers:
    _initialised = False

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        if not self._initialised:
            hordelib.initialise({consts.MODEL_CATEGORY_NAMES.diffusers: True})
        from hordelib.model_manager.diffusers import DiffusersModelManager

        self.diffusers_model_manager = DiffusersModelManager()
        assert self.diffusers_model_manager is not None
        yield
        del self.diffusers_model_manager

    def test_diffusers_load_defaults(self):
        success = self.diffusers_model_manager.load("stable_diffusion_inpainting")
        assert success
