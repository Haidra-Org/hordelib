"""Unit tests for :mod:`hordelib.utils.device_pinning`.

These tests are fully offline (no GPU, no torch import) since ``device_pin_env`` is
a pure mapping from ``(AcceleratorKind, index)`` to env-var dicts and CLI arg lists.
"""

from __future__ import annotations

import pytest

from hordelib.utils.device_pinning import device_pin_env
from hordelib.utils.torch_memory import AcceleratorKind


@pytest.mark.parametrize("index", [0, 1, 3])
def test_cuda_sets_cuda_visible_devices(index: int) -> None:
    env, args = device_pin_env(AcceleratorKind.cuda, index)

    assert env == {"CUDA_VISIBLE_DEVICES": str(index)}
    assert args == []


@pytest.mark.parametrize("index", [0, 1, 3])
def test_rocm_sets_both_hip_and_cuda_visible(index: int) -> None:
    env, args = device_pin_env(AcceleratorKind.rocm, index)

    assert env.get("HIP_VISIBLE_DEVICES") == str(index)
    assert env.get("CUDA_VISIBLE_DEVICES") == str(index)
    assert args == []


@pytest.mark.parametrize("index", [0, 1])
def test_xpu_sets_ze_affinity_mask(index: int) -> None:
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


def test_return_types_are_always_dict_and_list() -> None:
    for kind in AcceleratorKind:
        env, args = device_pin_env(kind, 0)
        assert isinstance(env, dict), f"env is not a dict for kind={kind}"
        assert isinstance(args, list), f"args is not a list for kind={kind}"


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
