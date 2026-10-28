"""Unit tests for the file-backed pending (beta) provider in hordelib.beta_models.

A process whose parent owns the pending read is handed a directory of per-category files through
``HORDELIB_BETA_MODELS_DIR`` and must serve beta records from them without a network request.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from horde_model_reference import (
    MODEL_REFERENCE_CATEGORY,
    PENDING_SOURCE_ID,
    PendingModelProvider,
    StaticModelProvider,
)

from hordelib.beta_models import (
    BETA_API_KEY_ENV_VAR,
    BETA_CATEGORIES_ENV_VAR,
    BETA_MODELS_DIR_ENV_VAR,
    build_pending_provider,
)

IMG = MODEL_REFERENCE_CATEGORY.image_generation
ESRGAN = MODEL_REFERENCE_CATEGORY.esrgan


def _image_record_fields() -> dict[str, object]:
    return {
        "model_classification": {"domain": "image", "purpose": "generation"},
        "baseline": "stable_diffusion_xl",
        "nsfw": False,
        "description": "A pending test record",
    }


def _write_category(directory: Path, category: MODEL_REFERENCE_CATEGORY, records: dict[str, object]) -> None:
    (directory / f"{category.value}.json").write_text(json.dumps(records), encoding="utf-8")


def test_file_mode_serves_the_records_under_the_pending_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The provider is static, registered under the pending id, and needs no API key."""
    _write_category(tmp_path, IMG, {"pending_model": _image_record_fields()})
    monkeypatch.setenv(BETA_CATEGORIES_ENV_VAR, "image_generation")
    monkeypatch.setenv(BETA_API_KEY_ENV_VAR, "")
    monkeypatch.setenv(BETA_MODELS_DIR_ENV_VAR, str(tmp_path))

    provider = build_pending_provider()

    assert isinstance(provider, StaticModelProvider)
    assert provider.source_id == PENDING_SOURCE_ID
    records = provider.fetch_category(IMG)
    assert records is not None
    assert set(records) == {"pending_model"}
    assert records["pending_model"].name == "pending_model"


def test_file_mode_missing_file_is_an_empty_category(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(BETA_CATEGORIES_ENV_VAR, "image_generation")
    monkeypatch.setenv(BETA_MODELS_DIR_ENV_VAR, str(tmp_path))

    provider = build_pending_provider()

    assert isinstance(provider, StaticModelProvider)
    assert provider.fetch_category(IMG) == {}


@pytest.mark.parametrize(
    "contents",
    ["{not json", json.dumps(["a", "list"]), json.dumps({"bad_model": {"baseline": 3}})],
    ids=["malformed-json", "not-a-mapping", "invalid-record"],
)
def test_file_mode_unreadable_file_is_an_empty_category(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    contents: str,
) -> None:
    (tmp_path / f"{IMG.value}.json").write_text(contents, encoding="utf-8")
    monkeypatch.setenv(BETA_CATEGORIES_ENV_VAR, "image_generation")
    monkeypatch.setenv(BETA_MODELS_DIR_ENV_VAR, str(tmp_path))

    provider = build_pending_provider()

    assert isinstance(provider, StaticModelProvider)
    assert provider.fetch_category(IMG) == {}


def test_file_mode_does_not_read_a_category_outside_the_opt_in(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _write_category(tmp_path, IMG, {"pending_model": _image_record_fields()})
    _write_category(tmp_path, ESRGAN, {"pending_upscaler": {}})
    monkeypatch.setenv(BETA_CATEGORIES_ENV_VAR, "image_generation")
    monkeypatch.setenv(BETA_MODELS_DIR_ENV_VAR, str(tmp_path))

    provider = build_pending_provider()

    assert provider is not None
    assert provider.provided_categories() == {IMG}
    assert provider.fetch_category(ESRGAN) is None


def test_unset_directory_keeps_the_network_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(BETA_CATEGORIES_ENV_VAR, "image_generation")
    monkeypatch.setenv(BETA_API_KEY_ENV_VAR, "reader-key")
    monkeypatch.delenv(BETA_MODELS_DIR_ENV_VAR, raising=False)

    provider = build_pending_provider()

    assert isinstance(provider, PendingModelProvider)
    assert provider.source_id == PENDING_SOURCE_ID
