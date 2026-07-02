"""Unit tests for :mod:`hordelib.utils.device_pinning`.

These tests are fully offline (no GPU, no torch import) since ``device_pin_env`` is
a pure mapping from ``(AcceleratorKind, index)`` to env-var dicts and CLI arg lists.
"""

from __future__ import annotations

import pytest

from hordelib.utils.device_pinning import device_pin_env
from hordelib.utils.torch_memory import AcceleratorKind

# --- No pre-existing env vars (logical == physical) ---


@pytest.mark.parametrize("index", [0, 1, 3])
def test_cuda_no_preexisting_env(monkeypatch: pytest.MonkeyPatch, index: int) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    env, args = device_pin_env(AcceleratorKind.cuda, index)

    assert env == {"CUDA_VISIBLE_DEVICES": str(index)}
    assert args == []


@pytest.mark.parametrize("index", [0, 1, 3])
def test_rocm_no_preexisting_env(monkeypatch: pytest.MonkeyPatch, index: int) -> None:
    monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising=False)
    env, args = device_pin_env(AcceleratorKind.rocm, index)

    assert env.get("HIP_VISIBLE_DEVICES") == str(index)
    assert env.get("CUDA_VISIBLE_DEVICES") == str(index)
    assert args == []


@pytest.mark.parametrize("index", [0, 1])
def test_xpu_no_preexisting_env(monkeypatch: pytest.MonkeyPatch, index: int) -> None:
    monkeypatch.delenv("ZE_AFFINITY_MASK", raising=False)
    env, args = device_pin_env(AcceleratorKind.xpu, index)

    assert env == {"ZE_AFFINITY_MASK": str(index)}
    assert args == []


@pytest.mark.parametrize("index", [0, 1])
def test_directml_returns_cli_arg_no_env(index: int) -> None:
    env, args = device_pin_env(AcceleratorKind.directml, index)

    assert env == {}
    assert args == ["--directml", str(index)]


def test_cpu_is_noop() -> None:
    env, args = device_pin_env(AcceleratorKind.cpu, 0)

    assert env == {}
    assert args == []


def test_mps_is_noop() -> None:
    env, args = device_pin_env(AcceleratorKind.mps, 0)

    assert env == {}
    assert args == []


def test_return_types_are_always_dict_and_list(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("ZE_AFFINITY_MASK", raising=False)
    for kind in AcceleratorKind:
        env, args = device_pin_env(kind, 0)
        assert isinstance(env, dict), f"env is not a dict for kind={kind}"
        assert isinstance(args, list), f"args is not a list for kind={kind}"


# --- Pre-existing env vars: logical index must map to physical identifier ---


def test_cuda_translates_logical_to_physical(monkeypatch: pytest.MonkeyPatch) -> None:
    """With CUDA_VISIBLE_DEVICES=0,2, logical index 1 must pin to physical GPU 2."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,2")
    env, _ = device_pin_env(AcceleratorKind.cuda, 1)

    assert env["CUDA_VISIBLE_DEVICES"] == "2"


def test_cuda_logical_0_maps_to_first_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    env, _ = device_pin_env(AcceleratorKind.cuda, 0)

    assert env["CUDA_VISIBLE_DEVICES"] == "2"


def test_cuda_single_entry_env_respects_physical(monkeypatch: pytest.MonkeyPatch) -> None:
    """User set CUDA_VISIBLE_DEVICES=1; logical index 0 must pin to physical GPU 1."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    env, _ = device_pin_env(AcceleratorKind.cuda, 0)

    assert env["CUDA_VISIBLE_DEVICES"] == "1"


def test_cuda_uuid_entry_passed_through(monkeypatch: pytest.MonkeyPatch) -> None:
    """UUID-format entries are passed through unchanged."""
    uuid = "GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", uuid)
    env, _ = device_pin_env(AcceleratorKind.cuda, 0)

    assert env["CUDA_VISIBLE_DEVICES"] == uuid


def test_rocm_translates_hip_visible_devices(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0,3")
    env, _ = device_pin_env(AcceleratorKind.rocm, 1)

    assert env["HIP_VISIBLE_DEVICES"] == "3"
    assert env["CUDA_VISIBLE_DEVICES"] == "3"


def test_xpu_translates_ze_affinity_mask(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZE_AFFINITY_MASK", "1,2")
    env, _ = device_pin_env(AcceleratorKind.xpu, 0)

    assert env["ZE_AFFINITY_MASK"] == "1"


def test_cuda_out_of_range_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Requesting a logical index beyond the visible set is a programming error."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    with pytest.raises(IndexError):
        device_pin_env(AcceleratorKind.cuda, 5)


def test_gpuinfo_accepts_explicit_index() -> None:
    """GPUInfo stores the explicit index rather than falling back to CUDA_VISIBLE_DEVICES."""
    from hordelib.utils.gpuinfo import GPUInfo

    info = GPUInfo(device_index=2)
    assert info.device == 2


def test_gpuinfo_defaults_to_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without an explicit index, GPUInfo reads CUDA_VISIBLE_DEVICES."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    from hordelib.utils import gpuinfo

    # Re-instantiate so __init__ re-reads the env var in this test's env.
    info = gpuinfo.GPUInfo()
    assert info.device == 1


def test_gpuinfo_falls_back_to_zero_when_env_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    from hordelib.utils import gpuinfo

    info = gpuinfo.GPUInfo()
    assert info.device == 0
