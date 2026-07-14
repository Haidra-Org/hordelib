import os
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import requests

from hordelib.model_manager.civitai_adhoc import SkipDownload, _QueuedDownload
from hordelib.model_manager.civitai_records import HordeTextualInversionModelRecord
from hordelib.model_manager.ti import TextualInversionModelManager, TIRejectionReason


def _make_ti_record(civitai_id: int = 123456) -> HordeTextualInversionModelRecord:
    """Return a minimal ad-hoc TI record suitable for driving download hooks in isolation."""
    return HordeTextualInversionModelRecord(
        name="test embedding",
        civitai_id=civitai_id,
        orig_name="Test Embedding",
        filename=f"{civitai_id}.safetensors",
        url="https://example.invalid/weights.safetensors",
        sha256="0" * 64,
        size_kb=24,
        nsfw=False,
        trigger=["test"],
        base_model="SD 1.5",
        version_id=999,
        adhoc=True,
    )


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

    def test_prepare_download_hordeling_404_records_not_found(self):
        mmti = TextualInversionModelManager()
        record = _make_ti_record()
        response = Mock(ok=False, status_code=404)
        with patch("hordelib.model_manager.ti.requests.get", return_value=response):
            with pytest.raises(SkipDownload):
                mmti._prepare_download(record)
        assert mmti._drain_rejection_reason(record.civitai_id) == TIRejectionReason.NOT_FOUND
        # Draining consumes the reason so a later fetch of the same id cannot inherit a stale verdict.
        assert mmti._drain_rejection_reason(record.civitai_id) is None
        mmti.stop_all()

    def test_prepare_download_unexpected_type_records_reason(self):
        mmti = TextualInversionModelManager()
        record = _make_ti_record()
        response = Mock(ok=False, status_code=422)
        response.json.return_value = {"message": "Unexpected type for the requested embedding"}
        with patch("hordelib.model_manager.ti.requests.get", return_value=response):
            with pytest.raises(SkipDownload):
                mmti._prepare_download(record)
        assert mmti._drain_rejection_reason(record.civitai_id) == TIRejectionReason.UNEXPECTED_TYPE
        mmti.stop_all()

    def test_prepare_download_transient_500_records_no_reason(self):
        mmti = TextualInversionModelManager()
        record = _make_ti_record()
        response = Mock(ok=False, status_code=500)
        response.json.return_value = {"message": "internal server error"}
        response.raise_for_status.side_effect = requests.HTTPError("500 Server Error")
        with patch("hordelib.model_manager.ti.requests.get", return_value=response):
            with pytest.raises(requests.HTTPError):
                mmti._prepare_download(record)
        assert mmti._drain_rejection_reason(record.civitai_id) is None
        mmti.stop_all()

    def test_fetch_adhoc_with_reason_impossible_id_invalid(self):
        mmti = TextualInversionModelManager()
        key, reason = mmti.fetch_adhoc_ti_with_reason(2**32)
        assert key is None
        assert reason == TIRejectionReason.INVALID
        mmti.stop_all()

    def test_fetch_adhoc_with_reason_no_items_not_found(self):
        mmti = TextualInversionModelManager()
        with patch.object(mmti, "_fetch_civitai_json", return_value={"items": []}):
            key, reason = mmti.fetch_adhoc_ti_with_reason("some-missing-name")
        assert key is None
        assert reason == TIRejectionReason.NOT_FOUND
        mmti.stop_all()

    def test_fetch_adhoc_with_reason_unparseable_record_invalid(self):
        mmti = TextualInversionModelManager()
        with (
            patch.object(mmti, "_fetch_civitai_json", return_value={"id": 42}),
            patch.object(mmti, "_parse_civitai_item", return_value=None),
        ):
            key, reason = mmti.fetch_adhoc_ti_with_reason(42)
        assert key is None
        assert reason == TIRejectionReason.INVALID
        mmti.stop_all()

    def test_fetch_adhoc_with_reason_metadata_failure_is_transient(self):
        mmti = TextualInversionModelManager()
        with patch.object(mmti, "_fetch_civitai_json", return_value=None):
            key, reason = mmti.fetch_adhoc_ti_with_reason(42)
        assert key is None
        assert reason is None
        mmti.stop_all()

    def test_fetch_adhoc_with_reason_surfaces_download_rejection(self):
        mmti = TextualInversionModelManager()
        record = _make_ti_record()

        def fake_enqueue(rec, progress_callback=None):
            # Mimic the worker abandoning the download after Hordeling reports it permanently gone.
            mmti._record_rejection_reason(rec.civitai_id, TIRejectionReason.NOT_FOUND)
            queued = _QueuedDownload(record=rec)
            queued.success = False
            queued.completion_event.set()
            return queued

        with (
            patch.object(mmti, "_fetch_civitai_json", return_value={"id": record.civitai_id}),
            patch.object(mmti, "_parse_civitai_item", return_value=record),
            patch.object(mmti, "fuzzy_find_ti_key", return_value=None),
            patch.object(mmti, "_enqueue_download", side_effect=fake_enqueue),
        ):
            key, reason = mmti.fetch_adhoc_ti_with_reason(record.civitai_id)
        assert key is None
        assert reason == TIRejectionReason.NOT_FOUND
        mmti.stop_all()

    def test_fetch_adhoc_with_reason_transient_download_failure_no_reason(self):
        mmti = TextualInversionModelManager()
        record = _make_ti_record()

        def fake_enqueue(rec, progress_callback=None):
            # A plain download failure records no reason; the fetch must stay reason-less.
            queued = _QueuedDownload(record=rec)
            queued.success = False
            queued.completion_event.set()
            return queued

        with (
            patch.object(mmti, "_fetch_civitai_json", return_value={"id": record.civitai_id}),
            patch.object(mmti, "_parse_civitai_item", return_value=record),
            patch.object(mmti, "fuzzy_find_ti_key", return_value=None),
            patch.object(mmti, "_enqueue_download", side_effect=fake_enqueue),
        ):
            key, reason = mmti.fetch_adhoc_ti_with_reason(record.civitai_id)
        assert key is None
        assert reason is None
        mmti.stop_all()

    def test_fetch_adhoc_ti_delegates_and_discards_reason(self):
        mmti = TextualInversionModelManager()
        with patch.object(mmti, "fetch_adhoc_ti_with_reason", return_value=("some-key", None)) as mocked:
            assert mmti.fetch_adhoc_ti("whatever") == "some-key"
        mocked.assert_called_once()
        with patch.object(
            mmti,
            "fetch_adhoc_ti_with_reason",
            return_value=(None, TIRejectionReason.NOT_FOUND),
        ):
            assert mmti.fetch_adhoc_ti("whatever") is None
        mmti.stop_all()
