import pytest

import hordelib
import hordelib.consts as consts


class TestCompvis:
    _initialised = False

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        if not self._initialised:
            hordelib.initialise({consts.MODEL_CATEGORY_NAMES.compvis: True})
        from hordelib.model_manager.compvis import CompVisModelManager

        self.compvis_model_manager = CompVisModelManager()
        assert self.compvis_model_manager is not None
        yield
        del self.compvis_model_manager

    def test_compvis_load_defaults(self):
        success = self.compvis_model_manager.load("Deliberate")
        assert success
