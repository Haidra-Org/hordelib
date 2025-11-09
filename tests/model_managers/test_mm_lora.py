import os

import pytest

from hordelib.model_manager.lora import LoraModelManager


class TestModelManagerLora:
    @pytest.fixture(autouse=True)
    def setup_and_teardown(self, test_loras_loaded):
        # We don't want to download a ton of loras for tests by mistake
        assert os.getenv("TESTS_ONGOING") == "1"
        # Ensure test loras are loaded before each test
        assert test_loras_loaded is not None
        required_loras = {"GlowingRunesAI", "Blindbox/Da Gai Shi Mang He"}
        missing = required_loras - set(test_loras_loaded)
        assert not missing, f"Required test LoRAs missing: {', '.join(sorted(missing))}"

    def test_downloading_default_sync(self, test_loras_loaded):
        """Test that the lora manager can be initialized and has models loaded.

        Since the GitHub default lora list is empty, we rely on test_loras_loaded fixture
        to have pre-downloaded the necessary test loras.
        """
        lora_model_manager = LoraModelManager(
            download_wait=True,
            allowed_adhoc_lora_storage=1024,
        )
        # Verify that loras from the fixture are available
        assert lora_model_manager.calculate_downloaded_loras() > 0
        lora_model_manager.stop_all()

    def test_downloading_default_async(self, test_loras_loaded):
        """Test async initialization with pre-loaded loras."""
        lora_model_manager = LoraModelManager(
            download_wait=False,
            allowed_adhoc_lora_storage=1024,
        )
        # Verify that loras from the fixture are available
        assert lora_model_manager.calculate_downloaded_loras() > 0
        lora_model_manager.stop_all()

    def test_fuzzy_search(self, test_loras_loaded):
        lora_model_manager = LoraModelManager(
            download_wait=False,
            allowed_adhoc_lora_storage=1024,
        )
        # No need to download - using fixture loras
        assert lora_model_manager.fuzzy_find_lora_key("Glowing Runes - konyconi") == "glowingrunesai - konyconi"
        assert lora_model_manager.fuzzy_find_lora_key("Glowing Robots") is None
        assert lora_model_manager.fuzzy_find_lora_key("GlowingRobots") is None
        assert lora_model_manager.fuzzy_find_lora_key("GlowingRobotsAI") is None
        assert lora_model_manager.fuzzy_find_lora_key(25995) == "blindbox/da gai shi mang he"
        assert lora_model_manager.fuzzy_find_lora_key("25995") == "blindbox/da gai shi mang he"
        assert lora_model_manager.fuzzy_find_lora_key("大概是盲盒") == "blindbox/da gai shi mang he"
        assert lora_model_manager.fuzzy_find_lora_key("blindbox/大概是盲盒") == "blindbox/da gai shi mang he"
        lora_model_manager.stop_all()

    def test_lora_search(self, test_loras_loaded):
        lora_model_manager = LoraModelManager(
            download_wait=False,
            allowed_adhoc_lora_storage=1024,
        )
        # No need to download - using fixture loras
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

        # DragonScale tests - skip if not available
        if "Dra9onScaleAI" in test_loras_loaded:
            assert lora_model_manager.get_lora_name("Dra9onScaleAI - konyconi") is not None
            assert lora_model_manager.get_lora_name("DragonScale - konyconi") is not None
            assert lora_model_manager.get_lora_name("Dragon Scale AI - konyconi") is not None
            assert lora_model_manager.find_lora_trigger("Dra9onScaleAI - konyconi", "Dr490nSc4leAI") is not None
            assert lora_model_manager.find_lora_trigger("DragonScale - konyconi", "DragonScaleAI") is not None
        else:
            print("\n  ⚠ Skipping DragonScale tests - lora not available")

        lora_model_manager.stop_all()

    def test_lora_reference(self, test_loras_loaded):
        download_amount = 1024
        lora_model_manager = LoraModelManager(
            allowed_top_lora_storage=download_amount,
            download_wait=False,
            allowed_adhoc_lora_storage=1024,
        )
        # No need to download - using fixture loras
        assert len(lora_model_manager.model_reference) > 0
        lora_model_manager.stop_all()

    def test_fetch_adhoc_lora(self, test_loras_loaded):
        lora_model_manager = LoraModelManager(
            download_wait=False,
            allowed_adhoc_lora_storage=1024,
        )
        # No need to download defaults - using fixture loras
        # Wait for adhoc reset to complete
        lora_model_manager.wait_for_adhoc_reset(15)

        lora_model_manager.ensure_lora_deleted(22591)
        lora_key = lora_model_manager.fetch_adhoc_lora("22591")
        assert lora_key == "gag - rpg potions  |  lora xl"
        assert lora_model_manager.is_model_available("22591")
        assert lora_model_manager.is_model_available("GAG - rpg potions  |  LoRa xl")
        assert lora_model_manager.get_lora_name("22591") == "GAG - RPG Potions  |  LoRa xl".lower()
        assert lora_model_manager.get_lora_filename("22591") == "GAG-RPGPotionsLoRaXL_197256.safetensors"
        lora_model_manager.stop_all()

    def test_fetch_adhoc_lora_conflicting_fuzz(self, test_loras_loaded):
        lora_model_manager = LoraModelManager(
            download_wait=False,
            allowed_adhoc_lora_storage=1024,
        )
        # No need to download defaults - using fixture loras
        # Wait for adhoc reset to complete
        lora_model_manager.wait_for_adhoc_reset(15)

        lora_model_manager.fetch_adhoc_lora("33970", timeout=300)
        lora_model_manager.ensure_lora_deleted("Eula Genshin Impact | Character Lora 1644")
        eula_key = lora_model_manager.fetch_adhoc_lora("Eula Genshin Impact | Character Lora 1644", timeout=300)
        assert lora_model_manager.get_lora_name("33970") == str("Dehya Genshin Impact | Character Lora 809".lower())
        assert eula_key is None
        assert lora_model_manager.get_lora_name("Eula Genshin Impact | Character Lora 1644") is None
        assert not lora_model_manager.is_model_available("Eula Genshin Impact | Character Lora 1644")
        lora_model_manager.stop_all()

    def test_fetch_specific_lora_version(self, test_loras_loaded):
        lora_model_manager = LoraModelManager(
            download_wait=False,
            allowed_adhoc_lora_storage=1024,
        )
        # No need to download defaults - using fixture loras
        # Wait for adhoc reset to complete
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

    def test_lora_long_filename(self, test_loras_loaded):
        lora_model_manager = LoraModelManager(
            download_wait=False,
            allowed_adhoc_lora_storage=1024,
            civitai_api_token=os.getenv("CIVIT_API_TOKEN", None),
        )
        # No need to download defaults - using fixture loras
        # Wait for adhoc reset to complete
        lora_model_manager.wait_for_adhoc_reset(15)

        lora_model_manager.ensure_lora_deleted(189973)
        lora_model_manager.fetch_adhoc_lora("372094", is_version=True)
        lora_dict = lora_model_manager.get_model_reference_info("189973")
        assert lora_model_manager.find_latest_version(lora_dict) == "372094"
        lora_model_manager.stop_all()

    def test_reject_adhoc_nsfw_lora(self, test_loras_loaded):
        lora_model_manager = LoraModelManager(
            download_wait=False,
            allowed_adhoc_lora_storage=1024,
        )
        lora_id = 9155
        # Set nsfw to False for this test
        lora_model_manager.nsfw = False
        # Wait for adhoc reset to complete
        lora_model_manager.wait_for_adhoc_reset(15)
        lora_model_manager.ensure_lora_deleted(lora_id)
        lora_key = lora_model_manager.fetch_adhoc_lora(lora_id)
        assert lora_model_manager.is_model_available(lora_id) is False
        assert lora_key is None
        lora_model_manager.stop_all()

    def test_approve_adhoc_lora(self, test_loras_loaded):
        lora_model_manager = LoraModelManager(
            download_wait=False,
            allowed_adhoc_lora_storage=1024,
        )
        lora_model_manager.nsfw = False  # Testing that setting like this is ignored
        lora_id = 9155
        # Set nsfw back to True to allow NSFW loras
        lora_model_manager.nsfw = True
        # Wait for adhoc reset to complete
        lora_model_manager.wait_for_adhoc_reset(15)
        lora_model_manager.ensure_lora_deleted(lora_id)
        lora_key = lora_model_manager.fetch_adhoc_lora(lora_id)
        assert lora_model_manager.is_model_available(lora_id) is True
        assert lora_key is not None
        lora_model_manager.stop_all()

    def test_adhoc_non_existing(self, test_loras_loaded):
        lora_model_manager = LoraModelManager(
            download_wait=False,
            allowed_adhoc_lora_storage=1024,
        )
        # No need to download defaults - using fixture loras
        # Wait for adhoc reset to complete
        lora_model_manager.wait_for_adhoc_reset(15)
        lora_name = (
            "__THIS SHOULD NOT EXIST. I SWEAR IF ONE OF YOU UPLOADS A LORA WITH THIS NAME I AM GOING TO BE UPSET!"
        )
        lora_key = lora_model_manager.fetch_adhoc_lora(lora_name)
        assert lora_key is None
        assert not lora_model_manager.is_model_available(lora_name)
        lora_model_manager.stop_all()

    def test_adhoc_non_existing_intstring_small(self, test_loras_loaded):
        lora_model_manager = LoraModelManager(
            download_wait=False,
            allowed_adhoc_lora_storage=1024,
        )
        # No need to download defaults - using fixture loras
        # Wait for adhoc reset to complete
        lora_model_manager.wait_for_adhoc_reset(15)
        lora_name = "12345"
        lora_key = lora_model_manager.fetch_adhoc_lora(lora_name)
        assert lora_model_manager.total_retries_attempted == 0
        assert lora_key is None
        assert not lora_model_manager.is_model_available(lora_name)
        lora_model_manager.stop_all()

    def test_adhoc_non_existing_intstring_large(self, test_loras_loaded):
        lora_model_manager = LoraModelManager(
            download_wait=False,
            allowed_adhoc_lora_storage=1024,
        )
        # No need to download defaults - using fixture loras
        # Wait for adhoc reset to complete
        lora_model_manager.wait_for_adhoc_reset(15)
        lora_name = "99999999999999"
        lora_key = lora_model_manager.fetch_adhoc_lora(lora_name)
        assert lora_model_manager.total_retries_attempted == 0
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
