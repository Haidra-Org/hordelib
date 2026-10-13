"""Pins that a model ComfyUI is partial-loading is not force-loaded whole by the ModelPatcher.load hijack.

``_load_models_gpu_hijack`` forces a full load by default and skips that when the weights exceed free VRAM,
handing ComfyUI a low-VRAM partial load instead. ``ModelPatcher.load`` is hijacked separately and also forces
``full_load``; unless it honours the outer decision, the partial load becomes a full load of a model that was
just found too large for the card, and the process runs out of memory. Seam test: both originals are stand-ins.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import pytest
import torch
from loguru import logger

from hordelib.execution import comfy_patches

MB = 1024 * 1024


class _Patcher:
    """A stand-in ModelPatcher: only ``model`` is read by the hijacks."""

    def __init__(self) -> None:
        self.model = object()


class _SizedPatcher:
    """A stand-in ModelPatcher reporting its weight size and how many of those bytes are on the device."""

    def __init__(self, *, size_bytes: int, loaded_bytes: int) -> None:
        self.model = object()
        self._size_bytes = size_bytes
        self._loaded_bytes = loaded_bytes

    def model_size(self) -> int:
        return self._size_bytes

    def loaded_size(self) -> int:
        return self._loaded_bytes

    def is_clone(self, other: Any) -> bool:
        return other.model is self.model


class _SizeOnlyPatcher:
    """A stand-in patcher with no residency figure, as a patcher type without ``loaded_size`` would be."""

    def __init__(self, *, size_bytes: int) -> None:
        self.model = object()
        self._size_bytes = size_bytes

    def model_size(self) -> int:
        return self._size_bytes


@dataclass
class _FreeVram:
    """The free VRAM figure comfy reports to the overflow check, set by the test."""

    free_bytes: int


@pytest.fixture
def free_vram(init_horde: None, monkeypatch: pytest.MonkeyPatch) -> _FreeVram:
    """A CUDA torch device whose free VRAM reading the test controls."""
    import comfy.model_management as model_management

    state = _FreeVram(free_bytes=0)
    monkeypatch.setattr(model_management, "get_torch_device", lambda: torch.device("cuda", 0))
    monkeypatch.setattr(
        model_management,
        "get_free_memory",
        lambda dev=None, torch_free_too=False: state.free_bytes,
    )
    return state


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

    assert seam["gpu_kwargs"] == {"memory_required": 0}, "comfy must size the partial load from the caller's figure"
    assert seam["load_kwargs"] == [{"full_load": False}]
    assert not comfy_patches._partial_load_patchers


@pytest.mark.parametrize(
    ("sampling_reserve_mb", "expect_partial"),
    [
        pytest.param(None, False, id="no-reserve-force-loads"),
        pytest.param(3117.0, True, id="reserve-takes-the-partial-branch"),
    ],
)
def test_a_held_sampling_reserve_is_charged_beside_the_remainder(
    monkeypatch: pytest.MonkeyPatch,
    free_vram: _FreeVram,
    seam,
    sampling_reserve_mb: float | None,
    expect_partial: bool,
) -> None:
    """A staged load's offloaded remainder that fits free VRAM alone is not force-loaded into the priced room.

    The staged partial load left the grant's activation reserve free. Charged against free VRAM alone, the
    remainder passes the 95% check and a forced full load fills the reserve the sample needs.
    """
    from hordelib.execution import sampling_lease

    monkeypatch.setattr(sampling_lease._lease_scope, "sampling_reserve_mb", sampling_reserve_mb, raising=False)
    free_vram.free_bytes = 3200 * MB
    remainder = _SizedPatcher(size_bytes=19600 * MB, loaded_bytes=17200 * MB)

    comfy_patches._load_models_gpu_hijack([remainder], memory_required=0)

    if expect_partial:
        assert "force_full_load" not in seam["gpu_kwargs"]
        assert seam["load_kwargs"] == [{"full_load": False}]
    else:
        assert seam["gpu_kwargs"]["force_full_load"] is True
        assert seam["load_kwargs"] == [{"full_load": True}]


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


def test_a_resident_model_is_not_charged_its_own_size(free_vram: _FreeVram) -> None:
    """A fully loaded encoder has nothing left to move, so the room beside it is not compared to its size.

    Charging the resident bytes sends it down the partial-load path, where ComfyUI unloads it to load it again.
    """
    free_vram.free_bytes = 4652 * MB

    resident = _SizedPatcher(size_bytes=7672 * MB, loaded_bytes=7672 * MB)
    assert comfy_patches._force_full_load_would_overflow_vram([resident]) is False

    offloaded = _SizedPatcher(size_bytes=7672 * MB, loaded_bytes=0)
    assert comfy_patches._force_full_load_would_overflow_vram([offloaded]) is True, "the check must not fail open"


def test_a_half_loaded_model_is_charged_its_unloaded_half(free_vram: _FreeVram) -> None:
    half_loaded = _SizedPatcher(size_bytes=8192 * MB, loaded_bytes=4096 * MB)

    free_vram.free_bytes = 4608 * MB
    assert comfy_patches._force_full_load_would_overflow_vram([half_loaded]) is False

    free_vram.free_bytes = 4096 * MB
    assert comfy_patches._force_full_load_would_overflow_vram([half_loaded]) is True


def test_a_patcher_without_a_residency_figure_is_charged_its_full_size(free_vram: _FreeVram) -> None:
    free_vram.free_bytes = 4608 * MB

    assert comfy_patches._force_full_load_would_overflow_vram([_SizeOnlyPatcher(size_bytes=4096 * MB)]) is False
    assert comfy_patches._force_full_load_would_overflow_vram([_SizeOnlyPatcher(size_bytes=4864 * MB)]) is True


def test_the_skip_path_passes_the_callers_sizing_through(monkeypatch: pytest.MonkeyPatch, seam) -> None:
    """An encode passes no minimum, so a forced ``memory_required`` would become comfy's minimum.

    That sized the partial load below zero and unloaded the very encoder being loaded. The caller's
    figures must reach comfy unchanged, with only ``force_full_load`` removed.
    """
    monkeypatch.setattr(comfy_patches, "_force_full_load_would_overflow_vram", lambda models: True)

    comfy_patches._load_models_gpu_hijack([_Patcher()], memory_required=1219 * MB, force_full_load=True)
    assert seam["gpu_kwargs"] == {"memory_required": 1219 * MB}

    comfy_patches._load_models_gpu_hijack(
        [_Patcher()],
        memory_required=2048 * MB,
        minimum_memory_required=1024 * MB,
    )
    assert seam["gpu_kwargs"] == {"memory_required": 2048 * MB, "minimum_memory_required": 1024 * MB}
    assert seam["load_kwargs"] == [{"full_load": False}, {"full_load": False}]


class _DiffusionModel:
    """Stands in for a skip-listed ``comfy.model_base`` class; its name is what the line reports."""


class _TextEncoder:
    """Stands in for a resident text encoder's module; its name is what the line reports."""


@dataclass
class _LoadedModel:
    """A stand-in for comfy's ``LoadedModel`` entry in ``current_loaded_models``."""

    model: _SizedPatcher
    device: torch.device

    def model_loaded_memory(self) -> int:
        return self.model.loaded_size()


@pytest.fixture
def info_lines() -> Iterator[list[str]]:
    """The messages logged at INFO or above while the test runs."""
    messages: list[str] = []
    handler_id = logger.add(lambda message: messages.append(message.record["message"]), level="INFO")
    yield messages
    logger.remove(handler_id)


def _resident_line_count(messages: list[str]) -> int:
    return sum(message.startswith(comfy_patches.SAMPLING_BESIDE_RESIDENT_MODELS_TEXT) for message in messages)


def _diffusion_patcher() -> _SizedPatcher:
    patcher = _SizedPatcher(size_bytes=11740 * MB, loaded_bytes=0)
    patcher.model = _DiffusionModel()
    return patcher


def _encoder_patcher(*, loaded_bytes: int) -> _SizedPatcher:
    patcher = _SizedPatcher(size_bytes=7672 * MB, loaded_bytes=loaded_bytes)
    patcher.model = _TextEncoder()
    return patcher


def test_a_skipped_load_beside_a_resident_model_logs_the_line_once_with_its_reason(
    monkeypatch: pytest.MonkeyPatch,
    seam,
    free_vram: _FreeVram,
    info_lines: list[str],
) -> None:
    import comfy.model_management as model_management

    free_vram.free_bytes = 4604 * MB
    diffusion = _diffusion_patcher()
    monkeypatch.setattr(
        comfy_patches,
        "_do_not_force_load_model_in_patcher",
        lambda patcher: isinstance(patcher.model, _DiffusionModel),
    )
    resident_encoder = _LoadedModel(model=_encoder_patcher(loaded_bytes=7672 * MB), device=torch.device("cuda", 0))
    monkeypatch.setattr(model_management, "current_loaded_models", [resident_encoder])

    comfy_patches._load_models_gpu_hijack([diffusion], memory_required=0)

    assert _resident_line_count(info_lines) == 1
    assert info_lines[-1] == (
        "Sampling beside resident models: reason=class_list _DiffusionModel loading 11740 MB with 7672 MB of "
        "other models resident (_TextEncoder), free 4604 MB; no forced eviction"
    )
    assert seam["gpu_kwargs"] == {"memory_required": 0}

    monkeypatch.setattr(comfy_patches, "_force_full_load_would_overflow_vram", lambda models: True)
    comfy_patches._load_models_gpu_hijack([_encoder_patcher(loaded_bytes=0)], memory_required=0)

    assert _resident_line_count(info_lines) == 2
    assert info_lines[-1] == (
        "Sampling beside resident models: reason=overflow _TextEncoder loading 7672 MB with 7672 MB of "
        "other models resident (_TextEncoder), free 4604 MB; no forced eviction"
    )


def test_the_resident_models_line_is_silent_without_another_resident(
    monkeypatch: pytest.MonkeyPatch,
    seam,
    free_vram: _FreeVram,
    info_lines: list[str],
) -> None:
    import comfy.model_management as model_management

    free_vram.free_bytes = 4604 * MB
    cuda = torch.device("cuda", 0)
    diffusion = _diffusion_patcher()
    monkeypatch.setattr(
        comfy_patches,
        "_do_not_force_load_model_in_patcher",
        lambda patcher: isinstance(patcher.model, _DiffusionModel),
    )

    monkeypatch.setattr(model_management, "current_loaded_models", [])
    comfy_patches._load_models_gpu_hijack([diffusion], memory_required=0)

    offloaded_encoder = _LoadedModel(model=_encoder_patcher(loaded_bytes=0), device=cuda)
    partially_seated_clone = _SizedPatcher(size_bytes=11740 * MB, loaded_bytes=3226 * MB)
    partially_seated_clone.model = diffusion.model
    the_load_itself = _LoadedModel(model=partially_seated_clone, device=cuda)
    monkeypatch.setattr(model_management, "current_loaded_models", [offloaded_encoder, the_load_itself])
    comfy_patches._load_models_gpu_hijack([diffusion], memory_required=0)

    monkeypatch.setattr(model_management, "current_loaded_models", [offloaded_encoder])
    monkeypatch.setattr(comfy_patches, "_force_full_load_would_overflow_vram", lambda models: True)
    comfy_patches._load_models_gpu_hijack([_encoder_patcher(loaded_bytes=0)], memory_required=0)

    assert _resident_line_count(info_lines) == 0
