import pytest

from hordelib.settings import UserSettings
from hordelib.utils.distance import (
    CosineSimilarityResultCode,
    HistogramDistanceResultCode,
)
from hordelib.utils.gpuinfo import GPUInfo
from hordelib.utils.image_utils import ImageUtils


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


@pytest.mark.skip(reason="This refers to code that is not currently used in production.")
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


@pytest.mark.skip(reason="This refers to code that is not currently used in production.")
class TestGPUInfo:
    def test_gpuinfo_init(self):
        gpu = GPUInfo()
        assert gpu is not None

        info = gpu.get_info()
        assert info is not None

        assert info.vram_total[0] > 0
        assert info.vram_free[0] > 0
        assert info.vram_used[0] > 0


class TestImageUtils:
    def test_get_first_pass_image_resolution_min(self):
        expected = (512, 512)
        calculated = ImageUtils.get_first_pass_image_resolution_min(512, 512)

        assert calculated == expected

    def test_under_sized_both_dimensions_min(self):
        expected = (256, 256)
        calculated = ImageUtils.get_first_pass_image_resolution_min(256, 256)

        assert calculated == expected

    def test_under_sized_one_dimension_min(self):
        expected = (512, 256)
        calculated = ImageUtils.get_first_pass_image_resolution_min(512, 256)

        assert calculated == expected

    def test_oversized_one_dimension_min(self):
        expected = (1024, 512)
        calculated = ImageUtils.get_first_pass_image_resolution_min(1024, 512)

        assert calculated == expected

    def test_oversized_other_dimension_min(self):
        expected = (512, 1024)
        calculated = ImageUtils.get_first_pass_image_resolution_min(512, 1024)

        assert calculated == expected

    def test_both_dimensions_oversized_evenly_min(self):
        expected = (512, 512)
        calculated = ImageUtils.get_first_pass_image_resolution_min(1024, 1024)

        assert calculated == expected

    def test_both_dimensions_oversized_unevenly_min(self):
        expected = (512, 768)
        calculated = ImageUtils.get_first_pass_image_resolution_min(1024, 1536)

        assert calculated == expected

    def test_get_first_pass_image_resolution_max(self):
        expected = (1024, 1024)
        calculated = ImageUtils.get_first_pass_image_resolution_max(1024, 1024)

        assert calculated == expected

    def test_under_sized_both_dimensions_max(self):
        expected = (512, 512)
        calculated = ImageUtils.get_first_pass_image_resolution_max(512, 512)

        assert calculated == expected

    def test_oversized_one_dimension_max(self):
        expected = (1024, 512)
        calculated = ImageUtils.get_first_pass_image_resolution_max(2048, 1024)

        assert calculated == expected

    def test_oversized_other_dimension_max(self):
        expected = (512, 1024)
        calculated = ImageUtils.get_first_pass_image_resolution_max(1024, 2048)

        assert calculated == expected

    def test_both_dimensions_oversized_evenly_max(self):
        expected = (1024, 1024)
        calculated = ImageUtils.get_first_pass_image_resolution_max(2048, 2048)

        assert calculated == expected

    def test_both_dimensions_oversized_unevenly_max(self):
        expected = (640, 1024)
        calculated_cascade = ImageUtils.get_first_pass_image_resolution_max(2048, 3072)
        calculated_default = ImageUtils.get_first_pass_image_resolution_min(2048, 3072)

        assert calculated_cascade != calculated_default
        assert calculated_cascade == expected
