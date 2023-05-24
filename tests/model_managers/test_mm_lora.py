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
        assert len(mml.get_lora_triggers("GlowingRunesAI")) > 1
        # We can't rely on triggers not changing
        assert mml.find_lora_trigger("GlowingRunesAI", "blue") is not None
        assert "blue" in mml.find_lora_trigger("GlowingRunesAI", "blue").lower()
        assert "red" in mml.find_lora_trigger("GlowingRunesAI", "red").lower()
        assert mml.find_lora_trigger("GlowingRunesAI", "pale blue") is None  # This is too much to fuzz
        assert mml.get_lora_name("Dra9onScaleAI") is not None
        assert mml.get_lora_name("DragonScale") is not None
        assert mml.get_lora_name("Dragon Scale AI") is not None
        assert mml.find_lora_trigger("Dra9onScaleAI", "Dr490nSc4leAI") is not None
        assert mml.find_lora_trigger("DragonScale", "DragonScaleAI") is not None
