import pytest

from hordelib.horde import HordeLib
from hordelib.model_manager.lora import LoraModelManager
from hordelib.utils.logger import HordeLog


class TestModelManagerLora:
    horde = HordeLib()
    default_model_manager_args: dict

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        self.horde = HordeLib()
        HordeLog.initialise(True)
        HordeLog.set_logger_verbosity(5)
        HordeLog.quiesce_logger(0)

        yield
        del self.horde

    def test_downloading_top(self):
        download_amount = 1024
        mml = LoraModelManager(
            download_reference=True,
            allowed_top_lora_storage=download_amount,
            download_wait=True,
        )
        assert mml.calculate_downloaded_loras() > download_amount
        assert mml.calculate_downloaded_loras() < download_amount * 1.3
