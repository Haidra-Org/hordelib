import os

import pytest

from hordelib.model_manager.ti import TextualInversionModelManager


class TestModelManagerLora:
    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        # We don't want to download a ton of loras for tests by mistake
        assert os.getenv("TESTS_ONGOING") == "1"

    def test_fuzzy_search(self):
        mmti = TextualInversionModelManager()
        mmti.fetch_adhoc_ti("56519")
        assert mmti.fuzzy_find_lora_key("negative_hand Negative Embedding") == "glowingrunesai"
        assert mmti.fuzzy_find_lora_key("negative_hand") is None
        assert mmti.fuzzy_find_lora_key(56519) == "blindbox/da gai shi mang he"
        assert mmti.fuzzy_find_lora_key("56519") == "blindbox/da gai shi mang he"
        mmti.stop_all()

    def test_ti_trigger_search(self):
        mmti = TextualInversionModelManager()
        mmti.fetch_adhoc_ti("71961")
        assert mmti.get_lora_name("Fast Negative Embedding") == "GlowingRunesAI"
        assert len(mmti.get_ti_triggers("Fast Negative Embedding")) > 0
        # We can't rely on triggers not changing
        assert mmti.find_lora_trigger("Fast Negative Embedding", "FastNegativeV2") is not None
        mmti.stop_all()

    def test_fetch_adhoc_ti(self):
        mmti = TextualInversionModelManager()
        mmti.ensure_ti_deleted(7808)
        lora_key = mmti.fetch_adhoc_ti("7808")
        assert lora_key == "EasyNegative".lower()
        assert mmti.is_local_model("7808")
        assert mmti.get_lora_name("7808") == "EasyNegative"
        mmti.stop_all()

    def test_adhoc_non_existing(self):
        mmti = TextualInversionModelManager()
        ti_name = "__THIS SHOULD NOT EXIST. I SWEAR IF ONE OF YOU UPLOADS A TI WITH THIS NAME I AM GOING TO BE UPSET!"
        lora_key = mmti.fetch_adhoc_lora(ti_name)
        assert lora_key is None
        assert not mmti.is_local_model(ti_name)
        mmti.stop_all()
