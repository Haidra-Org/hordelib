from hordelib.settings import UserSettings
from hordelib.utils.distance import (
    CosineSimilarityResultCode,
    HistogramDistanceResultCode,
)
from hordelib.utils.gpuinfo import GPUInfo


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


class TestWorkerSettingsWithInit:
    def test_worker_settings_properties_comparable(self, init_horde):
        assert UserSettings.get_ram_to_leave_free_mb() > 0
        assert UserSettings.get_vram_to_leave_free_mb() > 0

    def test_worker_settings_set_get_ram(self, init_horde):
        UserSettings.set_ram_to_leave_free_mb("50%")
        assert UserSettings.get_ram_to_leave_free_mb() > 0

    def test_worker_settings_set_get_vram(self, init_horde):
        UserSettings.set_vram_to_leave_free_mb("50%")
        assert UserSettings.get_vram_to_leave_free_mb() > 0


class TestImageDistance:
    def test_result_codes_in_order(self):
        last_value = None
        for result_code in CosineSimilarityResultCode:
            if last_value is not None:
                assert result_code > last_value, "CosineSimilarityResultCode values must be in *ascending* order"
            last_value = result_code

        last_value = None
        for result_code in HistogramDistanceResultCode:
            if last_value is not None:
                assert result_code < last_value, "HistogramDistanceResultCode values must be in *descending* order"
            last_value = result_code


class TestGPUInfo:
    def test_gpuinfo_init(self):
        gpu = GPUInfo()
        assert gpu is not None

        info = gpu.get_info()
        assert info is not None

        assert info.vram_total[0] > 0
        assert info.vram_free[0] > 0
        assert info.vram_used[0] > 0
