"""GPU-free unit tests for the cross-process GPU sampling lease hook.

The hook patches ``comfy.samplers.CFGGuider.sample`` to hold a host-injected lease around the model's
load to VRAM and the denoising loop. These tests run the real ComfyUI sampling entry points down to
``CFGGuider.outer_sample``, which is replaced per test so no model loads and no GPU is touched.
"""

import sys
from collections.abc import Callable
from typing import Any

import pytest
import torch

from hordelib.execution import comfy_patches, sampling_lease, weight_prefetch


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


class _FigureLease(_FakeLease):
    """A lease whose grant carries the host's device-free figure, as a clearance-granting host's does."""

    def __init__(self, *, figure: float | None, acquire_returns: bool = True) -> None:
        super().__init__(acquire_returns=acquire_returns)
        self._figure = figure
        self.take_calls = 0

    def take_cleared_device_free_mb(self) -> float | None:
        self.take_calls += 1
        return self._figure


class _FakeModelPatcher:
    """The ``ModelPatcher`` surface ``CFGGuider.sample`` reads before it reaches ``outer_sample``."""

    def __init__(self) -> None:
        self.model = torch.nn.Linear(1, 1)
        self.model_options: dict[str, Any] = {"transformer_options": {}}
        self.wrappers: dict[str, Any] = {}
        self.callbacks: dict[str, Any] = {}
        self.hook_mode: object = None
        self.load_device = torch.device("cpu")
        self.offload_device = torch.device("cpu")

    def is_dynamic(self) -> bool:
        return False

    def register_all_hook_patches(self, *args: object, **kwargs: object) -> None:
        return None

    def restore_hook_patches(self) -> None:
        return None


class _GuiderSeam:
    """The call log and prefetch record of one test, plus the hook for replacing ``outer_sample``."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, guider_class: type) -> None:
        self.call_log: list[str] = []
        self.prefetched: list[object] = []
        self._monkeypatch = monkeypatch
        self._guider_class = guider_class

    def replace_outer_sample(self, outer_sample: Callable[..., object]) -> None:
        self._monkeypatch.setattr(self._guider_class, "outer_sample", outer_sample)


@pytest.fixture
def guider_seam(init_horde: None, monkeypatch: pytest.MonkeyPatch) -> _GuiderSeam:
    """Stop real guider sampling at ``outer_sample`` and reset module-global hook state per test.

    ``CFGGuider.sample`` is re-bound through monkeypatch so the hook a test installs is removed afterwards.
    The returned call log records each ``outer_sample`` so tests can assert the lease is acquired *before* and
    released *after* the model load and denoise loop.
    """
    import comfy.samplers as samplers

    seam = _GuiderSeam(monkeypatch, samplers.CFGGuider)
    monkeypatch.setattr(samplers.CFGGuider, "sample", samplers.CFGGuider.sample)

    def _outer_sample(
        self: object, noise: torch.Tensor, latent_image: torch.Tensor, *args: object, **kwargs: object
    ) -> torch.Tensor:
        seam.call_log.append("sample")
        return latent_image

    seam.replace_outer_sample(_outer_sample)

    def _record_prefetch(module: object, **kwargs: object) -> None:
        seam.prefetched.append(module)

    monkeypatch.setattr(weight_prefetch, "prefetch_module_weights_async", _record_prefetch)

    monkeypatch.setattr(sampling_lease, "_installed", False)
    monkeypatch.setattr(sampling_lease, "_sampling_lease", None)
    monkeypatch.setattr(sampling_lease, "_acquire_timeout_seconds", 120.0)
    return seam


@pytest.fixture
def rebases(monkeypatch: pytest.MonkeyPatch, guider_seam: _GuiderSeam) -> list[float]:
    """Record each clamp rebase in the shared call log, so its order against the sample is visible."""
    recorded: list[float] = []

    def _record(device_free_truth_mb: float) -> bool:
        recorded.append(device_free_truth_mb)
        guider_seam.call_log.append("rebase")
        return True

    monkeypatch.setattr(comfy_patches, "rebase_free_memory_view_clamp", _record)
    return recorded


def _sigmas() -> torch.Tensor:
    return torch.tensor([1.0, 0.0])


def _latent() -> torch.Tensor:
    return torch.zeros(1, 4, 8, 8)


def _guider_sample(guider_class: type | None = None) -> object:
    """Sample through a guider, as ``SamplerCustomAdvanced`` does."""
    import comfy.samplers as samplers

    guider = (guider_class or samplers.CFGGuider)(_FakeModelPatcher())
    return guider.sample(torch.zeros(1, 4, 8, 8), _latent(), None, _sigmas())


def test_install_is_idempotent(guider_seam: _GuiderSeam) -> None:
    import comfy.samplers as samplers

    assert sampling_lease.install_sampling_lease_hook() is True
    first = samplers.CFGGuider.sample
    # A second install must not re-wrap the (already wrapped) function.
    assert sampling_lease.install_sampling_lease_hook() is True
    assert samplers.CFGGuider.sample is first


def test_no_lease_is_passthrough(guider_seam: _GuiderSeam) -> None:
    sampling_lease.install_sampling_lease_hook()
    assert isinstance(_guider_sample(), torch.Tensor)
    assert guider_seam.call_log == ["sample"]


def test_no_lease_calls_nothing_on_the_lease(guider_seam: _GuiderSeam) -> None:
    lease = _FigureLease(figure=12288.0)
    sampling_lease.install_sampling_lease_hook()
    sampling_lease.set_gpu_sampling_lease(lease)
    sampling_lease.set_gpu_sampling_lease(None)

    _guider_sample()

    assert lease.acquire_calls == []
    assert lease.release_count == 0
    assert lease.take_calls == 0
    assert guider_seam.call_log == ["sample"]


def test_lease_acquired_around_and_released_after(guider_seam: _GuiderSeam) -> None:
    lease = _FakeLease()
    sampling_lease.install_sampling_lease_hook()
    sampling_lease.set_gpu_sampling_lease(lease, acquire_timeout_seconds=30.0)

    _guider_sample()

    assert lease.acquire_calls == [(True, 30.0)]
    assert lease.release_count == 1
    assert lease.held == 0
    assert guider_seam.call_log == ["sample"]


def test_ksampler_path_through_comfy_sample_acquires_once(guider_seam: _GuiderSeam) -> None:
    import comfy.sample

    lease = _FakeLease()
    sampling_lease.install_sampling_lease_hook()
    sampling_lease.set_gpu_sampling_lease(lease)

    # denoise=0.0 keeps KSampler from computing a schedule from the (fake) model; the explicit sigmas still sample.
    comfy.sample.sample(
        _FakeModelPatcher(),
        torch.zeros(1, 4, 8, 8),
        1,
        1.0,
        "euler",
        "normal",
        [],
        [],
        _latent(),
        denoise=0.0,
        sigmas=_sigmas(),
    )

    assert len(lease.acquire_calls) == 1
    assert lease.release_count == 1
    assert guider_seam.call_log == ["sample"]


def test_sampler_custom_path_through_sample_custom_acquires_once(guider_seam: _GuiderSeam) -> None:
    import comfy.sample

    lease = _FakeLease()
    sampling_lease.install_sampling_lease_hook()
    sampling_lease.set_gpu_sampling_lease(lease)

    comfy.sample.sample_custom(_FakeModelPatcher(), torch.zeros(1, 4, 8, 8), 1.0, None, _sigmas(), [], [], _latent())

    assert len(lease.acquire_calls) == 1
    assert lease.release_count == 1
    assert guider_seam.call_log == ["sample"]


def test_guider_subclass_without_sample_override_acquires_once(guider_seam: _GuiderSeam) -> None:
    from comfy_extras.nodes_custom_sampler import Guider_Basic

    assert "sample" not in vars(Guider_Basic)
    lease = _FakeLease()
    sampling_lease.install_sampling_lease_hook()
    sampling_lease.set_gpu_sampling_lease(lease)

    _guider_sample(Guider_Basic)

    assert len(lease.acquire_calls) == 1
    assert lease.release_count == 1
    assert guider_seam.call_log == ["sample"]


def test_guider_subclass_whose_override_calls_super_acquires_once(guider_seam: _GuiderSeam) -> None:
    import comfy.samplers as samplers

    class _OverridingGuider(samplers.CFGGuider):
        def sample(self, noise: torch.Tensor, latent_image: torch.Tensor, *args: object, **kwargs: object) -> object:
            guider_seam.call_log.append("override")
            return super().sample(noise, latent_image, *args, **kwargs)

    lease = _FakeLease()
    sampling_lease.install_sampling_lease_hook()
    sampling_lease.set_gpu_sampling_lease(lease)

    _guider_sample(_OverridingGuider)

    assert len(lease.acquire_calls) == 1
    assert lease.release_count == 1
    assert guider_seam.call_log == ["override", "sample"]


def test_nested_guider_sample_acquires_once(guider_seam: _GuiderSeam) -> None:
    lease = _FakeLease()

    def _outer_sample_that_nests(
        self: object, noise: torch.Tensor, latent_image: torch.Tensor, *args: object, **kwargs: object
    ) -> torch.Tensor:
        guider_seam.call_log.append("outer")
        assert lease.held == 1
        if len(guider_seam.call_log) == 1:
            _guider_sample()
        return latent_image

    guider_seam.replace_outer_sample(_outer_sample_that_nests)
    sampling_lease.install_sampling_lease_hook()
    sampling_lease.set_gpu_sampling_lease(lease)

    _guider_sample()

    assert guider_seam.call_log == ["outer", "outer"]
    assert len(lease.acquire_calls) == 1
    assert lease.release_count == 1
    assert lease.held == 0


def test_hook_scope_clears_after_a_sample_so_the_next_one_acquires(guider_seam: _GuiderSeam) -> None:
    lease = _FakeLease()
    sampling_lease.install_sampling_lease_hook()
    sampling_lease.set_gpu_sampling_lease(lease)

    _guider_sample()
    _guider_sample()

    assert len(lease.acquire_calls) == 2
    assert lease.release_count == 2


def test_prefetch_reads_the_guiders_model(guider_seam: _GuiderSeam) -> None:
    import comfy.samplers as samplers

    sampling_lease.install_sampling_lease_hook()
    model_patcher = _FakeModelPatcher()

    samplers.CFGGuider(model_patcher).sample(torch.zeros(1, 4, 8, 8), _latent(), None, _sigmas())

    assert guider_seam.prefetched == [model_patcher.model]


def test_lease_released_even_when_sample_raises(guider_seam: _GuiderSeam) -> None:
    def _boom(self: object, *args: object, **kwargs: object) -> object:
        raise ValueError("sampling failed")

    guider_seam.replace_outer_sample(_boom)
    lease = _FakeLease()
    sampling_lease.install_sampling_lease_hook()
    sampling_lease.set_gpu_sampling_lease(lease)

    with pytest.raises(ValueError, match="sampling failed"):
        _guider_sample()

    assert lease.release_count == 1
    assert lease.held == 0
    # A failed sample must not leave the thread marked as inside the lease.
    with pytest.raises(ValueError, match="sampling failed"):
        _guider_sample()
    assert len(lease.acquire_calls) == 2


def test_acquire_timeout_samples_anyway_without_release(guider_seam: _GuiderSeam) -> None:
    # A lease that never grants (returns False) must not block the job and must not be released
    # (we never acquired it).
    lease = _FakeLease(acquire_returns=False)
    sampling_lease.install_sampling_lease_hook()
    sampling_lease.set_gpu_sampling_lease(lease)

    _guider_sample()
    assert guider_seam.call_log == ["sample"]
    assert lease.release_count == 0


def test_acquire_exception_samples_anyway(guider_seam: _GuiderSeam) -> None:
    lease = _FakeLease(raise_on_acquire=True)
    sampling_lease.install_sampling_lease_hook()
    sampling_lease.set_gpu_sampling_lease(lease)

    # A broken lease degrades to uncoordinated sampling rather than failing the job.
    _guider_sample()
    assert guider_seam.call_log == ["sample"]
    assert lease.release_count == 0


def test_clearing_lease_returns_to_passthrough(guider_seam: _GuiderSeam) -> None:
    lease = _FakeLease()
    sampling_lease.install_sampling_lease_hook()
    sampling_lease.set_gpu_sampling_lease(lease)
    _guider_sample()
    assert lease.acquire_calls

    sampling_lease.set_gpu_sampling_lease(None)
    _guider_sample()
    # No further acquires after clearing.
    assert len(lease.acquire_calls) == 1


def test_grant_figure_rebases_the_clamp_before_sampling(guider_seam: _GuiderSeam, rebases: list[float]) -> None:
    lease = _FigureLease(figure=12288.0)
    sampling_lease.install_sampling_lease_hook()
    sampling_lease.set_gpu_sampling_lease(lease)

    _guider_sample()

    assert rebases == [12288.0]
    assert guider_seam.call_log == ["rebase", "sample"], "the load inside sample must see the grant's figure"


def test_grant_without_a_figure_does_not_rebase(guider_seam: _GuiderSeam, rebases: list[float]) -> None:
    lease = _FigureLease(figure=None)
    sampling_lease.install_sampling_lease_hook()
    sampling_lease.set_gpu_sampling_lease(lease)

    _guider_sample()

    assert lease.take_calls == 1
    assert rebases == []


def test_timed_out_acquire_does_not_rebase(guider_seam: _GuiderSeam, rebases: list[float]) -> None:
    lease = _FigureLease(figure=12288.0, acquire_returns=False)
    sampling_lease.install_sampling_lease_hook()
    sampling_lease.set_gpu_sampling_lease(lease)

    _guider_sample()

    assert lease.take_calls == 0, "a timed-out acquire has no grant to read a figure from"
    assert rebases == []


def test_lease_without_the_figure_method_does_not_rebase(guider_seam: _GuiderSeam, rebases: list[float]) -> None:
    lease = _FakeLease()
    sampling_lease.install_sampling_lease_hook()
    sampling_lease.set_gpu_sampling_lease(lease)

    _guider_sample()

    assert rebases == []
    assert guider_seam.call_log == ["sample"]


def test_failed_rebase_still_samples(monkeypatch: pytest.MonkeyPatch, guider_seam: _GuiderSeam) -> None:
    def _boom(device_free_truth_mb: float) -> bool:
        raise RuntimeError("rebase boom")

    monkeypatch.setattr(comfy_patches, "rebase_free_memory_view_clamp", _boom)
    lease = _FigureLease(figure=12288.0)
    sampling_lease.install_sampling_lease_hook()
    sampling_lease.set_gpu_sampling_lease(lease)

    _guider_sample()
    assert guider_seam.call_log == ["sample"]
    assert lease.release_count == 1


def test_install_returns_false_when_comfy_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sampling_lease, "_installed", False)
    # A None parent package makes `from comfy import samplers` raise ImportError.
    monkeypatch.setitem(sys.modules, "comfy", None)
    monkeypatch.delitem(sys.modules, "comfy.samplers", raising=False)
    assert sampling_lease.install_sampling_lease_hook() is False
