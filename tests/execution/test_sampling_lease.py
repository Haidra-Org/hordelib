"""GPU-free unit tests for the cross-process GPU sampling lease hook.

The hook monkeypatches ``comfy.sample.sample`` to hold a host-injected lease around the
denoising loop. These tests inject a fake ``comfy.sample`` module so the wrapper logic can be
exercised without comfy or a GPU.
"""

import sys
import types

import pytest

from hordelib.execution import sampling_lease


class _FakeLease:
    """A threading/multiprocessing.Semaphore-shaped stand-in that records calls."""

    def __init__(self, *, acquire_returns: bool = True, raise_on_acquire: bool = False) -> None:
        self.acquire_calls: list[tuple[bool, float | None]] = []
        self.release_count = 0
        self.held = 0
        self._acquire_returns = acquire_returns
        self._raise_on_acquire = raise_on_acquire

    def acquire(self, block: bool = True, timeout: float | None = None) -> bool:
        self.acquire_calls.append((block, timeout))
        if self._raise_on_acquire:
            raise RuntimeError("acquire boom")
        if self._acquire_returns:
            self.held += 1
        return self._acquire_returns

    def release(self) -> None:
        self.release_count += 1
        self.held -= 1


@pytest.fixture(autouse=True)
def fake_comfy_sample(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Install a fake ``comfy.sample`` module and reset module-global hook state per test.

    Returns a call-order log the fake ``sample`` appends to, so tests can assert the lease is
    acquired *before* and released *after* the wrapped call.
    """
    call_log: list[str] = []

    fake_module = types.ModuleType("comfy.sample")

    def _original_sample(*args: object, **kwargs: object) -> str:
        call_log.append("sample")
        return "result"

    fake_module.sample = _original_sample  # type: ignore[attr-defined]

    comfy_pkg = types.ModuleType("comfy")
    monkeypatch.setitem(sys.modules, "comfy", comfy_pkg)
    monkeypatch.setitem(sys.modules, "comfy.sample", fake_module)

    monkeypatch.setattr(sampling_lease, "_installed", False)
    monkeypatch.setattr(sampling_lease, "_sampling_lease", None)
    monkeypatch.setattr(sampling_lease, "_acquire_timeout_seconds", 120.0)

    return call_log


def _patched_sample() -> object:
    return sys.modules["comfy.sample"].sample


def test_install_is_idempotent() -> None:
    assert sampling_lease.install_sampling_lease_hook() is True
    first = _patched_sample()
    # A second install must not re-wrap the (already wrapped) function.
    assert sampling_lease.install_sampling_lease_hook() is True
    assert _patched_sample() is first


def test_no_lease_is_passthrough(fake_comfy_sample: list[str]) -> None:
    sampling_lease.install_sampling_lease_hook()
    assert _patched_sample()() == "result"
    assert fake_comfy_sample == ["sample"]


def test_lease_acquired_around_and_released_after(fake_comfy_sample: list[str]) -> None:
    lease = _FakeLease()
    sampling_lease.install_sampling_lease_hook()
    sampling_lease.set_gpu_sampling_lease(lease, acquire_timeout_seconds=30.0)

    assert _patched_sample()() == "result"

    assert lease.acquire_calls == [(True, 30.0)]
    assert lease.release_count == 1
    assert lease.held == 0
    assert fake_comfy_sample == ["sample"]


def test_lease_released_even_when_sample_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_module = sys.modules["comfy.sample"]

    def _boom(*args: object, **kwargs: object) -> object:
        raise ValueError("sampling failed")

    monkeypatch.setattr(fake_module, "sample", _boom)

    lease = _FakeLease()
    sampling_lease.install_sampling_lease_hook()
    sampling_lease.set_gpu_sampling_lease(lease)

    with pytest.raises(ValueError, match="sampling failed"):
        _patched_sample()()

    assert lease.release_count == 1
    assert lease.held == 0


def test_acquire_timeout_samples_anyway_without_release(fake_comfy_sample: list[str]) -> None:
    # A lease that never grants (returns False) must not block the job and must not be released
    # (we never acquired it).
    lease = _FakeLease(acquire_returns=False)
    sampling_lease.install_sampling_lease_hook()
    sampling_lease.set_gpu_sampling_lease(lease)

    assert _patched_sample()() == "result"
    assert fake_comfy_sample == ["sample"]
    assert lease.release_count == 0


def test_acquire_exception_samples_anyway(fake_comfy_sample: list[str]) -> None:
    lease = _FakeLease(raise_on_acquire=True)
    sampling_lease.install_sampling_lease_hook()
    sampling_lease.set_gpu_sampling_lease(lease)

    # A broken lease degrades to uncoordinated sampling rather than failing the job.
    assert _patched_sample()() == "result"
    assert lease.release_count == 0


def test_clearing_lease_returns_to_passthrough(fake_comfy_sample: list[str]) -> None:
    lease = _FakeLease()
    sampling_lease.install_sampling_lease_hook()
    sampling_lease.set_gpu_sampling_lease(lease)
    _patched_sample()()
    assert lease.acquire_calls

    sampling_lease.set_gpu_sampling_lease(None)
    _patched_sample()()
    # No further acquires after clearing.
    assert len(lease.acquire_calls) == 1


def test_install_returns_false_when_comfy_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sampling_lease, "_installed", False)
    # A None parent package makes `from comfy import sample` raise ImportError.
    monkeypatch.setitem(sys.modules, "comfy", None)
    monkeypatch.delitem(sys.modules, "comfy.sample", raising=False)
    assert sampling_lease.install_sampling_lease_hook() is False
