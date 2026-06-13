"""Unit tests for opt-in beta model configuration (hordelib.beta_models).

These exercise only the env-driven configuration and source-selection logic; they do
not require a loaded ModelReferenceManager, model files, or a GPU.
"""

from __future__ import annotations

from typing import cast

import pytest
from horde_model_reference import (
    HORDE_SOURCE_ID,
    MODEL_REFERENCE_CATEGORY,
    PENDING_SOURCE_ID,
    ModelReferenceManager,
)

from hordelib.beta_models import (
    BETA_API_KEY_ENV_VAR,
    BETA_CATEGORIES_ENV_VAR,
    beta_model_categories,
    beta_source_for,
    build_pending_provider,
)

IMG = MODEL_REFERENCE_CATEGORY.image_generation


class _FakeManager:
    """Stand-in exposing only the ``get_provider`` method ``beta_source_for`` needs."""

    def __init__(self, registered_source_ids: set[str]) -> None:
        self._registered = registered_source_ids

    def get_provider(self, source_id: str) -> object | None:
        return object() if source_id in self._registered else None


def _as_manager(fake: _FakeManager) -> ModelReferenceManager:
    return cast(ModelReferenceManager, fake)


def test_categories_parse_skips_blanks_and_unknowns(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(BETA_CATEGORIES_ENV_VAR, "image_generation, , not_a_category")
    assert beta_model_categories() == {IMG}


def test_categories_empty_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(BETA_CATEGORIES_ENV_VAR, raising=False)
    assert beta_model_categories() == set()


def test_build_provider_none_without_categories(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(BETA_CATEGORIES_ENV_VAR, raising=False)
    assert build_pending_provider() is None


def test_build_provider_none_without_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(BETA_CATEGORIES_ENV_VAR, "image_generation")
    monkeypatch.delenv(BETA_API_KEY_ENV_VAR, raising=False)
    assert build_pending_provider() is None


def test_build_provider_when_fully_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(BETA_CATEGORIES_ENV_VAR, "image_generation")
    monkeypatch.setenv(BETA_API_KEY_ENV_VAR, "reader-key")
    provider = build_pending_provider()
    assert provider is not None
    assert provider.source_id == PENDING_SOURCE_ID
    assert provider.provided_categories() == {IMG}


def test_beta_source_selects_provider_first_when_registered(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(BETA_CATEGORIES_ENV_VAR, "image_generation")
    manager = _as_manager(_FakeManager({PENDING_SOURCE_ID}))
    assert beta_source_for(IMG, manager) == [PENDING_SOURCE_ID, HORDE_SOURCE_ID]


def test_beta_source_canonical_when_provider_not_registered(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(BETA_CATEGORIES_ENV_VAR, "image_generation")
    manager = _as_manager(_FakeManager(set()))
    assert beta_source_for(IMG, manager) == HORDE_SOURCE_ID


def test_beta_source_canonical_when_category_not_opted_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(BETA_CATEGORIES_ENV_VAR, "")
    manager = _as_manager(_FakeManager({PENDING_SOURCE_ID}))
    assert beta_source_for(IMG, manager) == HORDE_SOURCE_ID
