import pytest

import hordelib


class TestSafetyChecker:
    _initialised = False

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        if not self._initialised:
            hordelib.initialise()
        from hordelib.model_manager.safety_checker import SafetyCheckerModelManager

        self.safety_model_manager = SafetyCheckerModelManager()
        assert self.safety_model_manager is not None
        yield
        del self.safety_model_manager

    def test_safety_checker_load_defaults(self):
        success = self.safety_model_manager.load("safety_checker", cpu_only=True)
        assert success

    # def test_safety_checker_load_cpu_only(self):
    #     success = self.safety_model_manager.load("safety_checker", cpu_only=True)
    #     assert success
