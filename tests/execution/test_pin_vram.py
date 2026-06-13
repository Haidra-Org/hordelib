"""GPU-free unit tests for the HIGH_VRAM residency fallback (``pin_models_in_vram``).

This is the fallback path; the first-class residency lever is ``--highvram`` passed at
``initialise()`` time. The poke must still set the state when used and degrade gracefully
(return False, not raise) when comfy is unavailable.
"""

import enum
import sys
import types

import pytest

from hordelib.comfy_horde import pin_models_in_vram


def _install_fake_model_management(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    class _VRAMState(enum.Enum):
        NORMAL_VRAM = 3
        HIGH_VRAM = 4

    comfy_pkg = types.ModuleType("comfy")
    mm = types.ModuleType("comfy.model_management")
    mm.VRAMState = _VRAMState  # type: ignore[attr-defined]
    mm.vram_state = _VRAMState.NORMAL_VRAM  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "comfy", comfy_pkg)
    monkeypatch.setitem(sys.modules, "comfy.model_management", mm)
    return mm


def test_sets_high_vram_state(monkeypatch: pytest.MonkeyPatch) -> None:
    mm = _install_fake_model_management(monkeypatch)
    assert mm.vram_state is mm.VRAMState.NORMAL_VRAM

    assert pin_models_in_vram() is True
    assert mm.vram_state is mm.VRAMState.HIGH_VRAM


def test_returns_false_when_comfy_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "comfy", types.ModuleType("comfy"))
    monkeypatch.delitem(sys.modules, "comfy.model_management", raising=False)

    # Must swallow the ImportError and report failure rather than propagating.
    assert pin_models_in_vram() is False
