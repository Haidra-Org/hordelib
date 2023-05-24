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

    def test_downloading_default_sync(self):
        download_amount = 1024
        mml = LoraModelManager(
            allowed_top_lora_storage=download_amount,
            download_wait=True,
        )
        mml.download_default_loras()
        assert mml.are_downloads_complete() is True
        assert mml.calculate_downloaded_loras() > download_amount
        assert mml.calculate_downloaded_loras() < download_amount * 1.4

    def test_downloading_default_async(self):
        download_amount = 1024
        mml = LoraModelManager(
            allowed_top_lora_storage=download_amount,
            download_wait=False,
        )
        mml.download_default_loras()
        assert mml.are_downloads_complete() is False
        mml.wait_for_downloads()
        assert mml.are_downloads_complete() is True
        assert mml.calculate_downloaded_loras() > download_amount
        assert mml.calculate_downloaded_loras() < download_amount * 1.4

    def test_fuzzy_search(self):
        download_amount = 1024
        mml = LoraModelManager(
            allowed_top_lora_storage=download_amount,
            download_wait=False,
        )
        mml.download_default_loras()
        mml.wait_for_downloads()
        assert mml.fuzzy_find_lora("Glowing Runes") == "glowingrunesai"
        assert mml.fuzzy_find_lora("Glowing Robots") is None
        assert mml.fuzzy_find_lora("GlowingRobots") is None
        assert mml.fuzzy_find_lora("GlowingRobotsAI") is None

    def test_lora_search(self):
        download_amount = 1024
        mml = LoraModelManager(
            allowed_top_lora_storage=download_amount,
            download_wait=False,
        )
        mml.download_default_loras()
        mml.wait_for_downloads()
        assert mml.get_lora_name("GlowingRunesAI") == "GlowingRunesAI"
        assert mml.get_lora_name("GlowingRunes") == "GlowingRunesAI"
        assert mml.get_lora_name("Glowing Runes") == "GlowingRunesAI"
        assert len(mml.get_lora_triggers("GlowingRunesAI")) == 2
        assert mml.find_lora_trigger("GlowingRunesAI", "blue") == "GlowingRunesAIV2_paleblue"
        assert mml.find_lora_trigger("GlowingRunesAI", "pale blue") is None  # This is too much to fuzz
        assert mml.find_lora_trigger("GlowingRunesAI", "red") == "GlowingRunesAIV2_red"
        assert mml.get_lora_name("Dra9onScaleAI") == "Dra9onScaleAI"
        assert mml.get_lora_name("DragonScale") == "Dra9onScaleAI"
        assert mml.get_lora_name("Dragon Scale AI") == "Dra9onScaleAI"
        assert mml.find_lora_trigger("Dra9onScaleAI", "Dr490nSc4leAI") == "Dr490nSc4leAI"
        assert mml.find_lora_trigger("DragonScale", "DragonScaleAI") == "Dr490nSc4leAI"
