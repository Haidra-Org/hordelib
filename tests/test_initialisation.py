# test_initialisation.py
import sys
import types
from pathlib import Path

import pytest

import hordelib.installation.installer as installer_module
from hordelib.comfy_horde import Comfy_Horde
from hordelib.horde import HordeLib
from hordelib.initialisation import _assert_comfyui_checkout_safe_to_sync


def test_checkout_sync_guard_allows_loaded_comfy_at_the_pinned_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    pinned = "a" * 40
    monkeypatch.setitem(sys.modules, "comfy.sync_guard_probe", types.ModuleType("comfy.sync_guard_probe"))
    monkeypatch.setattr(installer_module, "_head_commit", lambda _path: pinned)

    _assert_comfyui_checkout_safe_to_sync(Path("ComfyUI"), pinned)


def test_checkout_sync_guard_rejects_mixed_comfyui_revisions(monkeypatch: pytest.MonkeyPatch) -> None:
    old = "a" * 40
    pinned = "b" * 40
    monkeypatch.setitem(sys.modules, "comfy.sync_guard_probe", types.ModuleType("comfy.sync_guard_probe"))
    monkeypatch.setattr(installer_module, "_head_commit", lambda _path: old)

    with pytest.raises(RuntimeError, match="after ComfyUI has been imported") as exc_info:
        _assert_comfyui_checkout_safe_to_sync(Path("ComfyUI"), pinned)

    assert old in str(exc_info.value)
    assert pinned in str(exc_info.value)
    assert "comfy.sync_guard_probe" in str(exc_info.value)


def test_find_comfyui(init_horde):
    import execution

    assert hasattr(execution, "get_input_data")


def test_instantiation(hordelib_instance: HordeLib):
    assert isinstance(hordelib_instance.backend.comfy_horde, Comfy_Horde)


def test_path():  # XXX
    from hordelib.config_path import set_system_path

    set_system_path()
