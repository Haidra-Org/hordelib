import pytest

from hordelib.model_manager.safety_checker import SafetyCheckerModelManager


class TestSafetyChecker:
    @pytest.fixture(scope="class")
    def isolated_safety_checker_model_manager(self, init_horde):
        yield SafetyCheckerModelManager()
