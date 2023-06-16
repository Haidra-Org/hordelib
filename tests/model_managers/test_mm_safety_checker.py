import pytest

from hordelib.model_manager.safety_checker import SafetyCheckerModelManager


class TestSafetyChecker:
    @pytest.fixture(scope="class")
    def isolated_safety_checker_model_manager(self, init_horde):
        yield SafetyCheckerModelManager()

    def test_safety_checker_load_defaults(
        self,
        isolated_safety_checker_model_manager: SafetyCheckerModelManager,
    ):
        success = isolated_safety_checker_model_manager.load("safety_checker", cpu_only=True)
        assert success
