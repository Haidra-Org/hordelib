"""Unit tests for the backend-agnostic accelerator abstraction in ``hordelib.utils.torch_memory``.

These never assume NVIDIA: they drive the ComfyUI-loaded path with a fake ``comfy.model_management``
module and the no-ComfyUI fallback path with a simulated CPU-only torch, so the behaviour is verified
without a GPU or any particular accelerator backend installed.
"""

from __future__ import annotations

import sys
import types

import pytest

from hordelib.utils import torch_memory
from hordelib.utils.torch_memory import (
    AcceleratorKind,
    clear_accelerator_cache,
    enumerate_accelerators,
    get_torch_device_free_vram_mb,
    get_torch_free_vram_mb,
    get_torch_total_vram_mb,
    torch_build_is_cpu_only,
)

_MB = 1024 * 1024


@pytest.fixture
def no_comfy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the no-ComfyUI fallback path even when another test has imported ComfyUI."""
    monkeypatch.delitem(sys.modules, torch_memory._COMFY_MODEL_MANAGEMENT, raising=False)


@pytest.fixture
def cpu_only_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulate a torch build with no CUDA/XPU/MPS device available."""
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    if hasattr(torch, "xpu"):
        monkeypatch.setattr(torch.xpu, "is_available", lambda: False, raising=False)
    if hasattr(torch.backends, "mps"):
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)


class _FakeDevice:
    def __init__(self, device_type: str) -> None:
        self.type = device_type


def _make_fake_comfy(devices: list[_FakeDevice], *, total_bytes: int) -> types.ModuleType:
    module = types.ModuleType(torch_memory._COMFY_MODEL_MANAGEMENT)
    module.get_all_torch_devices = lambda: list(devices)  # type: ignore[attr-defined]
    module.get_total_memory = lambda dev=None: total_bytes  # type: ignore[attr-defined]
    module.get_free_memory = lambda dev=None: total_bytes // 2  # type: ignore[attr-defined]
    module.get_torch_device_name = lambda dev: f"fake {dev.type}"  # type: ignore[attr-defined]
    module.cleared = False  # type: ignore[attr-defined]

    def _soft_empty_cache() -> None:
        module.cleared = True  # type: ignore[attr-defined]

    module.soft_empty_cache = _soft_empty_cache  # type: ignore[attr-defined]
    return module


def test_fallback_cpu_yields_single_cpu_pseudo_device(no_comfy: None, cpu_only_torch: None) -> None:
    accelerators = enumerate_accelerators()

    assert len(accelerators) == 1
    assert accelerators[0].kind is AcceleratorKind.cpu
    assert accelerators[0].index == 0
    assert accelerators[0].total_vram_mb > 0  # system RAM stands in for VRAM


def test_fallback_cpu_vram_reports_system_ram(no_comfy: None, cpu_only_torch: None) -> None:
    import psutil

    expected_total = round(psutil.virtual_memory().total / _MB)
    assert get_torch_total_vram_mb() == pytest.approx(expected_total, rel=0.01)
    assert get_torch_free_vram_mb() > 0


def test_fallback_clear_cache_is_noop_on_cpu(no_comfy: None, cpu_only_torch: None) -> None:
    # Must not raise and must not import torch.cuda.empty_cache unconditionally.
    clear_accelerator_cache()


def test_comfy_path_enumerates_via_comfy(monkeypatch: pytest.MonkeyPatch) -> None:
    devices = [_FakeDevice("cuda"), _FakeDevice("cuda")]
    fake = _make_fake_comfy(devices, total_bytes=24 * 1024 * _MB)
    monkeypatch.setitem(sys.modules, torch_memory._COMFY_MODEL_MANAGEMENT, fake)

    accelerators = enumerate_accelerators()

    assert len(accelerators) == 2
    assert accelerators[0].total_vram_mb == 24 * 1024
    assert all(a.kind in (AcceleratorKind.cuda, AcceleratorKind.rocm) for a in accelerators)


def test_comfy_path_clear_cache_delegates_to_soft_empty_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _make_fake_comfy([_FakeDevice("cuda")], total_bytes=_MB)
    monkeypatch.setitem(sys.modules, torch_memory._COMFY_MODEL_MANAGEMENT, fake)

    clear_accelerator_cache()

    assert fake.cleared is True  # type: ignore[attr-defined]


def test_comfy_path_maps_non_cuda_device_kinds(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _make_fake_comfy([_FakeDevice("mps"), _FakeDevice("xpu")], total_bytes=_MB)
    monkeypatch.setitem(sys.modules, torch_memory._COMFY_MODEL_MANAGEMENT, fake)

    kinds = [a.kind for a in enumerate_accelerators()]

    assert kinds == [AcceleratorKind.mps, AcceleratorKind.xpu]


def _set_torch_version(monkeypatch: pytest.MonkeyPatch, *, cuda: str | None, hip: str | None) -> None:
    """Force torch.version.cuda / .hip to the given values for a build-detection test."""
    import torch

    monkeypatch.setattr(torch.version, "cuda", cuda, raising=False)
    monkeypatch.setattr(torch.version, "hip", hip, raising=False)


def test_cpu_only_build_detected(monkeypatch: pytest.MonkeyPatch, cpu_only_torch: None) -> None:
    # A genuine CPU wheel reports no CUDA/HIP version and exposes no XPU/MPS device.
    _set_torch_version(monkeypatch, cuda=None, hip=None)
    assert torch_build_is_cpu_only() is True


def test_cuda_build_is_not_cpu_only_even_when_device_masked(
    monkeypatch: pytest.MonkeyPatch,
    cpu_only_torch: None,
) -> None:
    # A CUDA build whose GPU is merely masked (is_available() False) must NOT be treated as CPU-only:
    # that is a misconfigured GPU, not an intentional CPU install, and forcing --cpu would silently
    # mask the problem with a 100x-slower run.
    _set_torch_version(monkeypatch, cuda="13.2", hip=None)
    assert torch_build_is_cpu_only() is False


def test_rocm_build_is_not_cpu_only(monkeypatch: pytest.MonkeyPatch, cpu_only_torch: None) -> None:
    # ROCm presents through torch.cuda and sets both version.cuda and version.hip.
    _set_torch_version(monkeypatch, cuda="12.0", hip="6.2.0")
    assert torch_build_is_cpu_only() is False


def test_device_free_vram_excludes_comfy_reclaimable_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """``get_torch_device_free_vram_mb`` is the device-wide figure, not comfy's cache-inflated free.

    Regression for the over-commit/streaming root cause: comfy's ``get_free_memory`` returns the
    device-wide free PLUS this process's reserved-but-inactive allocator cache, over-stating what a new
    (cross-process) allocation can use and hiding VRAM held by other processes. The worker must budget
    against the device-wide ``mem_get_info`` figure instead, which this helper exposes regardless of
    whether ComfyUI is loaded.
    """
    fake = _make_fake_comfy([_FakeDevice("cuda")], total_bytes=16000 * _MB)
    # Comfy's view: device-wide free (11000) + this process's reclaimable cache (4000) == 15000.
    fake.get_free_memory = lambda dev=None, torch_free_too=False: 15000 * _MB  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, torch_memory._COMFY_MODEL_MANAGEMENT, fake)
    # The raw device-wide free from mem_get_info is the lower, honest number.
    monkeypatch.setattr(
        torch_memory,
        "_torch_fallback_vram_bytes",
        lambda *, free: (11000 * _MB if free else 16000 * _MB),
    )

    # Comfy's helper still reports the inflated figure (it serves comfy's own in-process allocation).
    assert get_torch_free_vram_mb() == 15000
    # The device-wide helper strips the reclaimable cache the budget must not count.
    assert get_torch_device_free_vram_mb() == 11000
