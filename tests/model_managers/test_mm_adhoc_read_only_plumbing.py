"""Unit tests that the ad-hoc read-only flag is plumbed from manager loading to the CivitAI managers.

``ModelManager.init_model_managers`` (and ``SharedModelManager.load_model_managers`` above it) accept
``adhoc_read_only``, which must reach exactly the ad-hoc CivitAI managers (LoRA and TI): a consumer
process that must never write (an inference child) then gets ``ReadOnlyModelManagerError`` enforcement.
Non-ad-hoc managers do not accept the flag. These tests construct the real managers against redirected
temp paths, so no shared cache is touched and no network is involved.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import hordelib.model_manager.base as base_module
from hordelib.model_manager.civitai_adhoc import ReadOnlyModelManagerError
from hordelib.model_manager.hyper import ModelManager
from hordelib.model_manager.lora import LoraModelManager
from hordelib.model_manager.ti import TextualInversionModelManager


@pytest.fixture
def isolated_adhoc_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect the ad-hoc managers' reference and weights roots into a temp dir (no shared cache)."""
    from horde_model_reference import horde_model_reference_paths

    legacy_dir = tmp_path / "legacy"
    legacy_dir.mkdir()
    monkeypatch.setattr(
        type(horde_model_reference_paths),
        "legacy_path",
        property(lambda _self: legacy_dir),
    )
    monkeypatch.setattr(base_module.UserSettings, "get_model_directory", classmethod(lambda _cls: tmp_path))
    return tmp_path


def test_adhoc_read_only_true_yields_read_only_lora_and_ti(isolated_adhoc_paths: Path) -> None:
    """Loading with ``adhoc_read_only=True`` makes both ad-hoc managers read-only and refuse fetches."""
    manager = ModelManager()
    manager.init_model_managers([LoraModelManager, TextualInversionModelManager], adhoc_read_only=True)

    assert manager.lora is not None and manager.lora.read_only is True
    assert manager.ti is not None and manager.ti.read_only is True

    with pytest.raises(ReadOnlyModelManagerError):
        manager.lora.fetch_adhoc_lora("anything")
    with pytest.raises(ReadOnlyModelManagerError):
        manager.ti.fetch_adhoc_ti("anything")


def test_adhoc_read_only_defaults_to_writable(isolated_adhoc_paths: Path) -> None:
    """Omitting ``adhoc_read_only`` keeps the ad-hoc managers writable (the default)."""
    manager = ModelManager()
    manager.init_model_managers([LoraModelManager, TextualInversionModelManager])

    assert manager.lora is not None and manager.lora.read_only is False
    assert manager.ti is not None and manager.ti.read_only is False
    # The writable guard is a no-op: the fetch path relies on it not raising for a writable manager.
    manager.lora._ensure_writable()
    manager.ti._ensure_writable()
