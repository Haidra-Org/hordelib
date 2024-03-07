import os

import pytest

from hordelib.model_manager.lora import LoraModelManager


class TestModelManagerLora:
    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        # We don't want to download a ton of loras for tests by mistake
        assert os.getenv("TESTS_ONGOING") == "1"

    def test_downloading_default_sync(self):
        lora_model_manager = LoraModelManager(
            download_wait=True,
            allowed_adhoc_lora_storage=1024,
        )
        lora_model_manager.download_default_loras(timeout=600)
        assert lora_model_manager.are_downloads_complete() is True
        assert lora_model_manager.calculate_downloaded_loras() > 0
        lora_model_manager.stop_all()

    def test_downloading_default_async(self):
        lora_model_manager = LoraModelManager(
            download_wait=False,
            allowed_adhoc_lora_storage=1024,
        )
        lora_model_manager.download_default_loras()
        assert lora_model_manager.are_downloads_complete() is False
        lora_model_manager.wait_for_downloads(300)
        assert lora_model_manager.are_downloads_complete() is True
        assert lora_model_manager.calculate_downloaded_loras() > 0
        lora_model_manager.stop_all()

    def test_fuzzy_search(self):
        lora_model_manager = LoraModelManager(
            download_wait=False,
            allowed_adhoc_lora_storage=1024,
        )
        lora_model_manager.download_default_loras()
        lora_model_manager.wait_for_downloads(600)
        assert lora_model_manager.fuzzy_find_lora_key("Glowing Runes - konyconi") == "glowingrunesai - konyconi"
        assert lora_model_manager.fuzzy_find_lora_key("Glowing Robots") is None
        assert lora_model_manager.fuzzy_find_lora_key("GlowingRobots") is None
        assert lora_model_manager.fuzzy_find_lora_key("GlowingRobotsAI") is None
        assert lora_model_manager.fuzzy_find_lora_key(25995) == "blindbox/da gai shi mang he"
        assert lora_model_manager.fuzzy_find_lora_key("25995") == "blindbox/da gai shi mang he"
        assert lora_model_manager.fuzzy_find_lora_key("大概是盲盒") == "blindbox/da gai shi mang he"
        assert lora_model_manager.fuzzy_find_lora_key("blindbox/大概是盲盒") == "blindbox/da gai shi mang he"
        lora_model_manager.stop_all()

    def test_lora_search(self):
        lora_model_manager = LoraModelManager(
            download_wait=False,
            allowed_adhoc_lora_storage=1024,
        )
        lora_model_manager.download_default_loras()
        lora_model_manager.wait_for_downloads(600)
        assert lora_model_manager.get_lora_name("GlowingRunesAI - konyconi") == "glowingrunesai - konyconi"
        assert lora_model_manager.get_lora_name("GlowingRunes - konyconi") == "glowingrunesai - konyconi"
        assert lora_model_manager.get_lora_name("Glowing Runes - konyconi") == "glowingrunesai - konyconi"
        assert len(lora_model_manager.get_lora_triggers("GlowingRunesAI - konyconi")) > 1
        # We can't rely on triggers not changing
        assert lora_model_manager.find_lora_trigger("GlowingRunesAI - konyconi", "blue") is not None
        assert "blue" in lora_model_manager.find_lora_trigger("GlowingRunesAI - konyconi", "blue").lower()
        assert "red" in lora_model_manager.find_lora_trigger("GlowingRunesAI - konyconi", "red").lower()
        assert (
            lora_model_manager.find_lora_trigger("GlowingRunesAI - konyconi", "pale blue") is None
        )  # This is too much to fuzz
        assert lora_model_manager.get_lora_name("Dra9onScaleAI - konyconi") is not None
        assert lora_model_manager.get_lora_name("DragonScale - konyconi") is not None
        assert lora_model_manager.get_lora_name("Dragon Scale AI - konyconi") is not None
        assert lora_model_manager.find_lora_trigger("Dra9onScaleAI - konyconi", "Dr490nSc4leAI") is not None
        assert lora_model_manager.find_lora_trigger("DragonScale - konyconi", "DragonScaleAI") is not None
        lora_model_manager.stop_all()

    def test_lora_reference(self):
        download_amount = 1024
        lora_model_manager = LoraModelManager(
            allowed_top_lora_storage=download_amount,
            download_wait=False,
            allowed_adhoc_lora_storage=1024,
        )
        lora_model_manager.download_default_loras()
        lora_model_manager.wait_for_downloads(600)
        assert len(lora_model_manager.model_reference) > 0
        lora_model_manager.stop_all()

    def test_fetch_adhoc_lora(self):
        lora_model_manager = LoraModelManager(
            download_wait=False,
            allowed_adhoc_lora_storage=1024,
        )
        lora_model_manager.download_default_loras()
        lora_model_manager.wait_for_downloads(600)
        lora_model_manager.wait_for_adhoc_reset(15)

        lora_model_manager.ensure_lora_deleted(22591)
        lora_key = lora_model_manager.fetch_adhoc_lora("22591")
        assert lora_key == "gag - rpg potions  |  lora xl"
        assert lora_model_manager.is_model_available("22591")
        assert lora_model_manager.is_model_available("GAG - rpg potions  |  LoRa xl")
        assert lora_model_manager.get_lora_name("22591") == "GAG - RPG Potions  |  LoRa xl".lower()
        assert lora_model_manager.get_lora_filename("22591") == "GAG-RPGPotionsLoRaXL_197256.safetensors"
        lora_model_manager.stop_all()

    def test_fetch_adhoc_lora_conflicting_fuzz(self):
        lora_model_manager = LoraModelManager(
            download_wait=False,
            allowed_adhoc_lora_storage=1024,
        )
        lora_model_manager.download_default_loras()
        lora_model_manager.wait_for_downloads(600)
        lora_model_manager.wait_for_adhoc_reset(15)

        lora_model_manager.fetch_adhoc_lora("33970")
        lora_model_manager.ensure_lora_deleted("Eula Genshin Impact | Character Lora 1644")
        lora_model_manager.fetch_adhoc_lora("Eula Genshin Impact | Character Lora 1644")
        assert lora_model_manager.get_lora_name("33970") == str("Dehya Genshin Impact | Character Lora 809".lower())
        assert lora_model_manager.get_lora_name("Eula Genshin Impact | Character Lora 1644") == str(
            "Eula Genshin Impact | Character Lora 1644".lower(),
        )
        lora_model_manager.stop_all()

    def test_fetch_specific_lora_version(self):
        lora_model_manager = LoraModelManager(
            download_wait=False,
            allowed_adhoc_lora_storage=1024,
        )
        lora_model_manager.download_default_loras()
        lora_model_manager.wait_for_downloads(600)
        lora_model_manager.wait_for_adhoc_reset(15)

        lora_model_manager.ensure_lora_deleted(22591)
        lora_key = lora_model_manager.fetch_adhoc_lora("26975", is_version=True)
        lora_dict = lora_model_manager.get_model_reference_info("22591")
        assert lora_model_manager.find_latest_version(lora_dict) == "26975"
        assert lora_key == "gag - rpg potions  |  lora xl"
        assert lora_model_manager.is_model_available("22591")
        assert isinstance(lora_model_manager.get_model_reference_info("26975", is_version=True), dict)
        assert lora_model_manager.get_lora_name("22591") == "GAG - RPG Potions  |  lora xl".lower()
        assert lora_model_manager.get_lora_filename("22591") == "GAG-RPGPotionsLoRaXL_26975.safetensors"
        # We test that grabbing the generic lora name afterwards will actually get the latest version
        ln = lora_model_manager.fetch_adhoc_lora("22591")
        lora_dict = lora_model_manager.get_model_reference_info(ln)
        assert lora_model_manager.find_latest_version(lora_dict) == "197256"
        assert lora_model_manager.get_lora_filename("22591") == "GAG-RPGPotionsLoRaXL_197256.safetensors"
        lora_model_manager.stop_all()

    def test_reject_adhoc_nsfw_lora(self):
        lora_model_manager = LoraModelManager(
            download_wait=False,
            allowed_adhoc_lora_storage=1024,
        )
        lora_id = 9155
        lora_model_manager.download_default_loras(nsfw=False)
        lora_model_manager.wait_for_downloads(300)
        lora_model_manager.wait_for_adhoc_reset(15)
        lora_model_manager.ensure_lora_deleted(lora_id)
        lora_key = lora_model_manager.fetch_adhoc_lora(lora_id)
        assert lora_model_manager.is_model_available(lora_id) is False
        assert lora_key is None
        lora_model_manager.stop_all()

    def test_approve_adhoc_lora(self):
        lora_model_manager = LoraModelManager(
            download_wait=False,
            allowed_adhoc_lora_storage=1024,
        )
        lora_model_manager.nsfw = False  # Testing that setting like this is ignored
        lora_id = 9155
        lora_model_manager.download_default_loras(nsfw=True)
        lora_model_manager.wait_for_downloads(300)
        lora_model_manager.wait_for_adhoc_reset(15)
        lora_model_manager.ensure_lora_deleted(lora_id)
        lora_key = lora_model_manager.fetch_adhoc_lora(lora_id)
        assert lora_model_manager.is_model_available(lora_id) is True
        assert lora_key is not None
        lora_model_manager.stop_all()

    def test_adhoc_non_existing(self):
        lora_model_manager = LoraModelManager(
            download_wait=False,
            allowed_adhoc_lora_storage=1024,
        )
        lora_model_manager.download_default_loras()
        lora_model_manager.wait_for_downloads(600)
        lora_model_manager.wait_for_adhoc_reset(15)
        lora_name = (
            "__THIS SHOULD NOT EXIST. I SWEAR IF ONE OF YOU UPLOADS A LORA WITH THIS NAME I AM GOING TO BE UPSET!"
        )
        lora_key = lora_model_manager.fetch_adhoc_lora(lora_name)
        assert lora_key is None
        assert not lora_model_manager.is_model_available(lora_name)
        lora_model_manager.stop_all()

    def test_adhoc_non_existing_intstring_small(self):
        lora_model_manager = LoraModelManager(
            download_wait=False,
            allowed_adhoc_lora_storage=1024,
        )
        lora_model_manager.download_default_loras()
        lora_model_manager.wait_for_downloads(600)
        lora_model_manager.wait_for_adhoc_reset(15)
        lora_name = "12345"
        lora_key = lora_model_manager.fetch_adhoc_lora(lora_name)
        assert lora_model_manager.total_retries_attempted == 0
        assert lora_key is None
        assert not lora_model_manager.is_model_available(lora_name)
        lora_model_manager.stop_all()

    def test_adhoc_non_existing_intstring_large(self):
        lora_model_manager = LoraModelManager(
            download_wait=False,
            allowed_adhoc_lora_storage=1024,
        )
        lora_model_manager.download_default_loras()
        lora_model_manager.wait_for_downloads(600)
        lora_model_manager.wait_for_adhoc_reset(15)
        lora_name = "99999999999999"
        lora_key = lora_model_manager.fetch_adhoc_lora(lora_name)
        assert lora_model_manager.total_retries_attempted == 1
        assert lora_key is None
        assert not lora_model_manager.is_model_available(lora_name)
        lora_model_manager.stop_all()

    ## Disabling this until I can figure out a better way to test these
    # def test_unused_loras(self):
    #     lora_model_manager = LoraModelManager(
    #         download_wait=False,
    #         allowed_adhoc_lora_storage=1024,
    #     )
    #     lora_model_manager.download_default_loras()
    #     lora_model_manager.wait_for_downloads(600)
    #     lora_model_manager.wait_for_adhoc_reset(15)
    #     assert lora_model_manager.find_lora_from_filename("GlowingRunesAI.safetensors") ==
    #     lora_model_manager.stop_all()
    #     lora_model_manager = LoraModelManager(
    #         download_wait=False,
    #         allowed_adhoc_lora_storage=100,
    #     )
    #     assert lora_model_manager._thread is None
    #     lora_model_manager.download_default_loras()
    #     lora_model_manager.wait_for_downloads(15)
    #     lora_model_manager.wait_for_adhoc_reset(15)
    #     assert len(lora_model_manager._adhoc_loras) > 0
    #     assert lora_model_manager.is_adhoc_cache_full()
    #     assert lora_model_manager.calculate_adhoc_loras_cache() > 100
    #     assert lora_model_manager.calculate_adhoc_loras_cache() < 300
    #     unused_loras = lora_model_manager.find_unused_loras()
    #     assert len(unused_loras) > 0
    #     assert "GlowingRunesAI - konyconi" not in unused_loras
    #     assert "dra9onscaleai" not in unused_loras
    #     lora_model_manager.stop_all()
    #     lora_model_manager = LoraModelManager(
    #         allowed_top_lora_storage=500,
    #         download_wait=False,
    #         allowed_adhoc_lora_storage=100,
    #     )
    #     assert lora_model_manager._thread is None
    #     lora_model_manager.download_default_loras()
    #     with pytest.raises(Exception):
    #         lora_model_manager.delete_unused_loras(0)
    #     deleted_loras = lora_model_manager.delete_unused_loras(15)
    #     assert len(deleted_loras) > 0
    #     lora_model_manager.wait_for_adhoc_reset(15)
    #     assert all("last_used" in lora for lora in lora_model_manager.model_reference.values())
    #     lora_model_manager.stop_all()
