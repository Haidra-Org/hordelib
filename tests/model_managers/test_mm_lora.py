import os

import pytest

from hordelib.model_manager.lora import LoraModelManager


class TestModelManagerLora:
    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        # We don't want to download a ton of loras for tests by mistake
        assert os.getenv("TESTS_ONGOING") == "1"

    def test_downloading_default_sync(self):

        mml = LoraModelManager(
            download_wait=True,
            allowed_adhoc_lora_storage=1024,
        )
        mml.download_default_loras(timeout=600)
        assert mml.are_downloads_complete() is True
        assert mml.calculate_downloaded_loras() > 0
        mml.stop_all()

    def test_downloading_default_async(self):
        mml = LoraModelManager(
            download_wait=False,
            allowed_adhoc_lora_storage=1024,
        )
        mml.download_default_loras()
        assert mml.are_downloads_complete() is False
        mml.wait_for_downloads(300)
        assert mml.are_downloads_complete() is True
        assert mml.calculate_downloaded_loras() > 0
        mml.stop_all()

    def test_fuzzy_search(self):
        mml = LoraModelManager(
            download_wait=False,
            allowed_adhoc_lora_storage=1024,
        )
        mml.download_default_loras()
        mml.wait_for_downloads(600)
        assert mml.fuzzy_find_lora_key("Glowing Runes") == "glowingrunesai"
        assert mml.fuzzy_find_lora_key("Glowing Robots") is None
        assert mml.fuzzy_find_lora_key("GlowingRobots") is None
        assert mml.fuzzy_find_lora_key("GlowingRobotsAI") is None
        assert mml.fuzzy_find_lora_key("blindbox/大概是盲盒") == "blindbox/da gai shi mang he"
        assert mml.fuzzy_find_lora_key(25995) == "blindbox/da gai shi mang he"
        assert mml.fuzzy_find_lora_key("25995") == "blindbox/da gai shi mang he"
        assert mml.fuzzy_find_lora_key("大概是盲盒") == "blindbox/da gai shi mang he"
        mml.stop_all()

    def test_lora_search(self):
        mml = LoraModelManager(
            download_wait=False,
            allowed_adhoc_lora_storage=1024,
        )
        mml.download_default_loras()
        mml.wait_for_downloads(600)
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
        mml.stop_all()

    def test_lora_reference(self):
        download_amount = 1024
        mml = LoraModelManager(
            allowed_top_lora_storage=download_amount,
            download_wait=False,
            allowed_adhoc_lora_storage=1024,
        )
        mml.download_default_loras()
        mml.wait_for_downloads(600)
        assert len(mml.model_reference) > 0
        mml.stop_all()

    def test_fetch_adhoc_lora(self):
        mml = LoraModelManager(
            download_wait=False,
            allowed_adhoc_lora_storage=1024,
        )
        mml.download_default_loras()
        mml.wait_for_downloads(600)
        mml.wait_for_adhoc_reset(15)
        mml.ensure_lora_deleted(22591)
        lora_key = mml.fetch_adhoc_lora("22591")
        assert lora_key == "GAG - RPG Potions  |  LoRa 2.1".lower()
        assert mml.is_local_model("GAG")
        assert mml.is_local_model("22591")
        assert mml.get_lora_name("22591") == "GAG - RPG Potions  |  LoRa 2.1"
        mml.stop_all()

    def test_reject_adhoc_nsfw_lora(self):
        mml = LoraModelManager(
            download_wait=False,
            allowed_adhoc_lora_storage=1024,
        )
        lora_id = 9155
        mml.download_default_loras(nsfw=False)
        mml.wait_for_downloads(300)
        mml.wait_for_adhoc_reset(15)
        mml.ensure_lora_deleted(lora_id)
        lora_key = mml.fetch_adhoc_lora(lora_id)
        assert mml.is_local_model(lora_id) is False
        assert lora_key is None
        mml.stop_all()

    def test_approve_adhoc_lora(self):
        mml = LoraModelManager(
            download_wait=False,
            allowed_adhoc_lora_storage=1024,
        )
        mml.nsfw = False  # Testing that setting like this is ignored
        lora_id = 9155
        mml.download_default_loras(nsfw=True)
        mml.wait_for_downloads(300)
        mml.wait_for_adhoc_reset(15)
        mml.ensure_lora_deleted(lora_id)
        lora_key = mml.fetch_adhoc_lora(lora_id)
        assert mml.is_local_model(lora_id) is True
        assert lora_key is not None
        mml.stop_all()

    def test_adhoc_non_existing(self):
        mml = LoraModelManager(
            download_wait=False,
            allowed_adhoc_lora_storage=1024,
        )
        mml.download_default_loras()
        mml.wait_for_downloads(600)
        mml.wait_for_adhoc_reset(15)
        lora_name = (
            "__THIS SHOULD NOT EXIST. I SWEAR IF ONE OF YOU UPLOADS A LORA WITH THIS NAME I AM GOING TO BE UPSET!"
        )
        lora_key = mml.fetch_adhoc_lora(lora_name)
        assert lora_key is None
        assert not mml.is_local_model(lora_name)
        mml.stop_all()

    ## Disabling this until I can figure out a better way to test these
    # def test_unused_loras(self):
    #     mml = LoraModelManager(
    #         download_wait=False,
    #         allowed_adhoc_lora_storage=1024,
    #     )
    #     mml.download_default_loras()
    #     mml.wait_for_downloads(600)
    #     mml.wait_for_adhoc_reset(15)
    #     assert mml.find_lora_from_filename("GlowingRunesAI.safetensors") == "glowingrunesai"
    #     mml.stop_all()
    #     mml = LoraModelManager(
    #         download_wait=False,
    #         allowed_adhoc_lora_storage=100,
    #     )
    #     assert mml._thread is None
    #     mml.download_default_loras()
    #     mml.wait_for_downloads(15)
    #     mml.wait_for_adhoc_reset(15)
    #     assert len(mml._adhoc_loras) > 0
    #     assert mml.is_adhoc_cache_full()
    #     assert mml.calculate_adhoc_loras_cache() > 100
    #     assert mml.calculate_adhoc_loras_cache() < 300
    #     unused_loras = mml.find_unused_loras()
    #     assert len(unused_loras) > 0
    #     assert "glowingrunesai" not in unused_loras
    #     assert "dra9onscaleai" not in unused_loras
    #     mml.stop_all()
    #     mml = LoraModelManager(
    #         allowed_top_lora_storage=500,
    #         download_wait=False,
    #         allowed_adhoc_lora_storage=100,
    #     )
    #     assert mml._thread is None
    #     mml.download_default_loras()
    #     with pytest.raises(Exception):
    #         mml.delete_unused_loras(0)
    #     deleted_loras = mml.delete_unused_loras(15)
    #     assert len(deleted_loras) > 0
    #     mml.wait_for_adhoc_reset(15)
    #     assert all("last_used" in lora for lora in mml.model_reference.values())
    #     mml.stop_all()
