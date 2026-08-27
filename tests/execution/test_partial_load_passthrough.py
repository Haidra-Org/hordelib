"""Pins that a model ComfyUI is partial-loading is not force-loaded whole by the ModelPatcher.load hijack.

``_load_models_gpu_hijack`` forces a full load by default and skips that when the weights exceed free VRAM,
handing ComfyUI a low-VRAM partial load instead. ``ModelPatcher.load`` is hijacked separately and also forces
``full_load``; unless it honours the outer decision, the partial load becomes a full load of a model that was
just found too large for the card, and the process runs out of memory. Seam test: both originals are stand-ins.
"""

from __future__ import annotations

from typing import Any

import pytest

from hordelib.execution import comfy_patches


class _Patcher:
    """A stand-in ModelPatcher: only ``model`` is read by the hijacks."""

    def __init__(self) -> None:
        self.model = object()


@pytest.fixture
def seam(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Route both hijacks through recording originals, with the class-based skip list disabled."""
    calls: dict[str, Any] = {"load_kwargs": [], "gpu_kwargs": None}

    def fake_load_models_gpu(models, **kwargs):
        calls["gpu_kwargs"] = kwargs
        # ComfyUI reaches ModelPatcher.load from inside load_models_gpu; the hijack sits in between.
        for patcher in models:
            comfy_patches._model_patcher_load_hijack(patcher, full_load=False)

    def fake_model_patcher_load(patcher, **kwargs):
        calls["load_kwargs"].append(kwargs)

    monkeypatch.setitem(comfy_patches._originals, "load_models_gpu", fake_load_models_gpu)
    monkeypatch.setitem(comfy_patches._originals, "model_patcher_load", fake_model_patcher_load)
    monkeypatch.setattr(comfy_patches, "_do_not_force_load_model_in_patcher", lambda patcher: False)
    monkeypatch.setattr(comfy_patches, "_small_support_models_only", lambda models: False)
    import hordelib.execution.cpu_weight_retention as retention

    monkeypatch.setattr(retention, "stash_cpu_origins", lambda model: None)
    return calls


def test_a_model_that_does_not_fit_keeps_comfys_partial_load(monkeypatch: pytest.MonkeyPatch, seam) -> None:
    monkeypatch.setattr(comfy_patches, "_force_full_load_would_overflow_vram", lambda models: True)

    comfy_patches._load_models_gpu_hijack([_Patcher()], memory_required=0)

    assert seam["gpu_kwargs"]["memory_required"] == 1e30
    assert "force_full_load" not in seam["gpu_kwargs"]
    assert seam["load_kwargs"] == [{"full_load": False}]
    assert not comfy_patches._partial_load_patchers


def test_a_model_that_fits_is_still_force_loaded(monkeypatch: pytest.MonkeyPatch, seam) -> None:
    monkeypatch.setattr(comfy_patches, "_force_full_load_would_overflow_vram", lambda models: False)

    comfy_patches._load_models_gpu_hijack([_Patcher()], memory_required=0)

    assert seam["gpu_kwargs"]["force_full_load"] is True
    assert seam["load_kwargs"] == [{"full_load": True}]


def test_the_partial_load_allowance_ends_with_the_call(monkeypatch: pytest.MonkeyPatch, seam) -> None:
    monkeypatch.setattr(comfy_patches, "_force_full_load_would_overflow_vram", lambda models: True)
    patcher = _Patcher()

    comfy_patches._load_models_gpu_hijack([patcher], memory_required=0)
    comfy_patches._model_patcher_load_hijack(patcher, full_load=False)

    assert seam["load_kwargs"] == [{"full_load": False}, {"full_load": True}]
