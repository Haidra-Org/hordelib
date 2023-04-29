import pytest

import hordelib
import hordelib.initialisation
from hordelib.settings import UserSettings


def test_worker_settings_singleton():
    a = UserSettings._instance
    assert a is not None
    b = UserSettings()
    assert b._instance is not None
    assert a is b._instance


def test_worker_settings_percent_check():
    assert UserSettings._is_percentage("50%") == 50
    assert UserSettings._is_percentage("50") is False
    assert UserSettings._is_percentage("%50") is False


# def test_worker_settings_no_init():
#     # _get_total_vram_mb returns 0 if hordelib is not initialised
#     # We fake this condition with _is_initialised = False
#     hordelib.initialisation._is_initialised = False
#     assert UserSettings.get_vram_to_leave_free_mb() == 0


class TestWorkerSettingsWithInit:
    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        hordelib.initialise()
        yield
        UserSettings._instance = None
        UserSettings._ram_to_leave_free_mb = None
        UserSettings._vram_to_leave_free_mb = None

    def test_worker_settings_properties_comparable(self):
        assert UserSettings.get_ram_to_leave_free_mb() > 0
        assert UserSettings.get_vram_to_leave_free_mb() > 0

    def test_worker_settings_set_get_ram(self):
        UserSettings.set_ram_to_leave_free_mb("50%")
        assert UserSettings.get_ram_to_leave_free_mb() > 0

    def test_worker_settings_set_get_vram(self):
        UserSettings.set_vram_to_leave_free_mb("50%")
        assert UserSettings.get_vram_to_leave_free_mb() > 0
