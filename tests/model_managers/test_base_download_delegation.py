"""GPU-free tests for BaseModelManager's delegation to the horde_model_reference download engine.

These pin the Phase 5 contract: ``download_model``/``download_file`` are thin wrappers over
``horde_model_reference.download_engine`` (and ``on_disk_layout``); the engine is never invoked for
externally-managed categories (LoRA/TI); the CivitAI token is forwarded only for CivitAI-hosted URLs; and
a tainted model's on-disk files are cleared before re-download so the engine will re-fetch them.

The managers are built via ``__new__`` with only the attributes the methods under test require, so no
GPU, network, or ModelReferenceManager initialisation is needed.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from horde_model_reference import MODEL_REFERENCE_CATEGORY, category_folder
from horde_model_reference.download_engine import DownloadOutcome
from horde_model_reference.model_reference_records import (
    DownloadRecord,
    EsrganModelRecord,
    GenericModelRecord,
    GenericModelRecordConfig,
)

from hordelib.consts import CIVITAI_API_PATH
from hordelib.model_manager import base as base_module
from hordelib.model_manager.esrgan import EsrganModelManager


def _make_manager(
    weights_root: Path,
    *,
    category: MODEL_REFERENCE_CATEGORY = MODEL_REFERENCE_CATEGORY.esrgan,
    civitai_api_token: str | None = None,
) -> EsrganModelManager:
    """Build a manager with only the state the download methods touch, bypassing __init__."""
    manager = EsrganModelManager.__new__(EsrganModelManager)
    manager._model_category = category
    manager._weights_root = weights_root
    folder = category_folder(category)
    assert folder is not None
    manager.model_folder_path = weights_root / folder
    manager.model_reference = {}
    manager.available_models = []
    manager.tainted_models = []
    manager._civitai_api_token = civitai_api_token
    return manager


def _record(file_name: str = "model.pth", *, file_url: str = "https://example.test/model.pth") -> EsrganModelRecord:
    return EsrganModelRecord(
        name="m",
        config=GenericModelRecordConfig(download=[DownloadRecord(file_name=file_name, file_url=file_url)]),
    )


def test_download_model_delegates_to_engine_and_validates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manager = _make_manager(tmp_path)
    record = _record()
    manager.model_reference = {"m": record}

    captured: dict[str, object] = {}

    def fake_download_record_files(
        record: GenericModelRecord,
        root: Path,
        *,
        progress_callback: object = None,
        auth_query_token: object = None,
    ) -> bool:
        captured.update(record=record, root=root, callback=progress_callback, token=auth_query_token)
        return True

    monkeypatch.setattr(base_module.download_engine, "download_record_files", fake_download_record_files)
    monkeypatch.setattr(manager, "validate_model", lambda name: True)

    def callback(downloaded: int, total: int) -> None:
        return None

    assert manager.download_model("m", callback=callback) is True
    assert captured["record"] is record
    assert captured["root"] == tmp_path
    assert captured["callback"] is callback
    assert captured["token"] is None


def test_download_model_returns_false_when_engine_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manager = _make_manager(tmp_path)
    manager.model_reference = {"m": _record()}

    monkeypatch.setattr(base_module.download_engine, "download_record_files", lambda *a, **k: False)
    validate_called = False

    def fake_validate(name: str) -> bool:
        nonlocal validate_called
        validate_called = True
        return True

    monkeypatch.setattr(manager, "validate_model", fake_validate)

    assert manager.download_model("m") is False
    assert validate_called is False


def test_download_model_without_record_returns_false(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manager = _make_manager(tmp_path)
    called = False

    def fake_download_record_files(*args: object, **kwargs: object) -> bool:
        nonlocal called
        called = True
        return True

    monkeypatch.setattr(base_module.download_engine, "download_record_files", fake_download_record_files)
    assert manager.download_model("missing") is False
    assert called is False


@pytest.mark.parametrize("category", [MODEL_REFERENCE_CATEGORY.lora, MODEL_REFERENCE_CATEGORY.ti])
def test_externally_managed_categories_bypass_engine(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    category: MODEL_REFERENCE_CATEGORY,
) -> None:
    manager = _make_manager(tmp_path, category=category)
    manager.model_reference = {"m": _record()}

    def fail(*args: object, **kwargs: object) -> bool:
        raise AssertionError("the download engine must not run for externally-managed categories")

    monkeypatch.setattr(base_module.download_engine, "download_record_files", fail)

    assert manager.download_model("m") is None
    assert manager.download_all_models() is True


def test_civitai_token_only_forwarded_for_civitai_urls(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path, civitai_api_token="secret-token")

    civitai_record = _record(file_url=f"https://civitai.test/{CIVITAI_API_PATH}/42")
    other_record = _record(file_url="https://huggingface.test/model.pth")

    assert manager._civitai_token_for(civitai_record) == "secret-token"
    assert manager._civitai_token_for(other_record) is None

    manager_without_token = _make_manager(tmp_path)
    assert manager_without_token._civitai_token_for(civitai_record) is None


def test_tainted_model_files_cleared_before_redownload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manager = _make_manager(tmp_path)
    record = _record(file_name="taint.pth")
    manager.model_reference = {"m": record}
    manager.tainted_models = ["m"]

    weight_path = tmp_path / "esrgan" / "taint.pth"
    weight_path.parent.mkdir(parents=True, exist_ok=True)
    weight_path.write_bytes(b"stale-bytes")
    weight_path.with_suffix(".sha256").write_text("deadbeef *taint.pth")

    def fake_download_record_files(record: GenericModelRecord, root: Path, **kwargs: object) -> bool:
        # The stale copy and its sidecar must be gone by the time the engine is asked to re-fetch.
        assert not weight_path.exists()
        assert not weight_path.with_suffix(".sha256").exists()
        return True

    monkeypatch.setattr(base_module.download_engine, "download_record_files", fake_download_record_files)
    monkeypatch.setattr(manager, "validate_model", lambda name: True)

    assert manager.download_model("m") is True


def test_download_file_wrapper_targets_model_folder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manager = _make_manager(tmp_path)
    captured: dict[str, object] = {}

    def fake_download_file(
        url: str,
        destination: Path,
        *,
        progress_callback: object = None,
        **kwargs: object,
    ) -> DownloadOutcome:
        captured.update(url=url, destination=destination, callback=progress_callback)
        return DownloadOutcome(success=True, final_path=destination, bytes_written=3, sha256="abc")

    monkeypatch.setattr(base_module.download_engine, "download_file", fake_download_file)

    assert manager.download_file("https://example.test/f.pth", "f.pth") is True
    assert captured["destination"] == manager.model_folder_path / "f.pth"
