# test_reference_offline.py
"""Unit tests for hordelib's offline reference policy.

hordelib must not own the reference-download policy: under the worker it must never force a network
prefetch. These tests exercise the offline resolution precedence and the construction behaviour
without requiring a full ComfyUI init.
"""

from collections.abc import Generator
from unittest.mock import MagicMock

import pytest
from horde_model_reference import (
    ModelReferenceManager,
    PrefetchStrategy,
    horde_model_reference_settings,
)

import hordelib.shared_model_manager as smm
from hordelib.shared_model_manager import SharedModelManager


@pytest.fixture
def reset_reference_singletons() -> Generator[None]:
    """Reset the reference/shared-manager singletons and offline knobs around a test."""
    mrm_prev = ModelReferenceManager._instance
    smm_inst_prev = SharedModelManager._instance
    smm_mgr_prev = getattr(SharedModelManager, "manager", None)
    offline_prev = SharedModelManager._reference_offline
    settings_offline_prev = horde_model_reference_settings.offline

    ModelReferenceManager._instance = None
    SharedModelManager._instance = None  # type: ignore[assignment]
    SharedModelManager._reference_offline = None
    try:
        yield
    finally:
        ModelReferenceManager._instance = mrm_prev
        SharedModelManager._instance = smm_inst_prev
        SharedModelManager.manager = smm_mgr_prev  # type: ignore[assignment]
        SharedModelManager._reference_offline = offline_prev
        horde_model_reference_settings.offline = settings_offline_prev


def test_resolve_reference_offline_precedence(reset_reference_singletons: None) -> None:
    horde_model_reference_settings.offline = False

    SharedModelManager._reference_offline = None
    assert SharedModelManager._resolve_reference_offline(True) is True
    assert SharedModelManager._resolve_reference_offline(False) is False
    assert SharedModelManager._resolve_reference_offline(None) is False

    SharedModelManager._reference_offline = True
    assert SharedModelManager._resolve_reference_offline(None) is True

    SharedModelManager._reference_offline = None
    horde_model_reference_settings.offline = True
    assert SharedModelManager._resolve_reference_offline(None) is True


def test_load_model_managers_reuses_existing_instance_without_prefetch(
    reset_reference_singletons: None,
    tmp_path: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pre-constructed reference manager (e.g. by the worker) is reused, never re-prefetched."""
    manager = ModelReferenceManager(
        base_path=tmp_path,  # type: ignore[arg-type]
        offline=True,
        prefetch_strategy=PrefetchStrategy.NONE,
    )

    prefetch_called = False

    def _boom(_ref: ModelReferenceManager) -> None:
        nonlocal prefetch_called
        prefetch_called = True

    monkeypatch.setattr(smm, "_await_prefetch", _boom)

    fake_manager = MagicMock()
    fake_manager.lora = None
    fake_manager.ti = None
    SharedModelManager.manager = fake_manager  # type: ignore[assignment]

    SharedModelManager.load_model_managers([])

    assert prefetch_called is False
    assert SharedModelManager.model_reference_manager is manager
    fake_manager.init_model_managers.assert_called_once()


def test_load_model_managers_creates_offline_without_prefetch(
    reset_reference_singletons: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When offline is resolved True and no instance exists, construct offline and skip prefetch."""
    horde_model_reference_settings.offline = True

    def _boom(_ref: ModelReferenceManager) -> None:
        raise AssertionError("offline reference manager must not trigger a network prefetch")

    monkeypatch.setattr(smm, "_await_prefetch", _boom)

    fake_manager = MagicMock()
    fake_manager.lora = None
    fake_manager.ti = None
    SharedModelManager.manager = fake_manager  # type: ignore[assignment]

    SharedModelManager.load_model_managers([])

    assert SharedModelManager.model_reference_manager.offline is True
