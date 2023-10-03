import os
from pathlib import Path

import pytest

from hordelib.model_manager.ti import TextualInversionModelManager


class TestModelManagerTI:
    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        # We don't want to download a ton of tis for tests by mistake
        assert os.getenv("TESTS_ONGOING") == "1"

    def test_fuzzy_search(self):
        mmti = TextualInversionModelManager()
        mmti.fetch_adhoc_ti("56519")
        assert mmti.fuzzy_find_ti_key("negative_hand") == "negative_hand negative embedding"
        assert mmti.fuzzy_find_ti_key(56519) == "negative_hand negative embedding"
        assert mmti.fuzzy_find_ti_key("56519") == "negative_hand negative embedding"
        mmti.stop_all()

    def test_ti_trigger_search(self):
        mmti = TextualInversionModelManager()
        mmti.fetch_adhoc_ti("71961")
        assert mmti.get_ti_name("Fast Negative Embedding") == "Fast Negative Embedding (+ FastNegativeV2)"
        assert len(mmti.get_ti_triggers("Fast Negative Embedding")) > 0
        # We can't rely on triggers not changing
        assert mmti.find_ti_trigger("Fast Negative Embedding", "FastNegativeV2") is not None
        mmti.stop_all()

    def test_fetch_adhoc_ti(self):
        mmti = TextualInversionModelManager()
        mmti.ensure_ti_deleted(7808)
        ti_key = mmti.fetch_adhoc_ti("7808")
        assert ti_key == "EasyNegative".lower()
        assert mmti.is_local_model("7808")
        assert mmti.get_ti_name("7808") == "EasyNegative"
        assert Path(os.path.join(mmti.model_folder_path, "7808.safetensors")).exists()
        mmti.stop_all()

    def test_adhoc_non_existing(self):
        mmti = TextualInversionModelManager()
        ti_name = "__THIS SHOULD NOT EXIST. I SWEAR IF ONE OF YOU UPLOADS A TI WITH THIS NAME I AM GOING TO BE UPSET!"
        ti_key = mmti.fetch_adhoc_ti(ti_name)
        assert ti_key is None
        assert not mmti.is_local_model(ti_name)
        mmti.stop_all()
