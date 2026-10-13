"""GPU-free unit tests for the cross-process GPU sampling lease hook.

The hook patches ``comfy.samplers.CFGGuider.sample`` to hold a host-injected lease around the model's
load to VRAM and the denoising loop. These tests run the real ComfyUI sampling entry points down to
``CFGGuider.outer_sample``, which is replaced per test so no model loads and no GPU is touched.
"""

import sys
import types
from collections.abc import Callable, Iterator
from typing import Any

import pytest
import torch
from loguru import logger

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
    import comfy.sampler_helpers as sampler_helpers
    import comfy.samplers as samplers

    seam = _GuiderSeam(monkeypatch, samplers.CFGGuider)
    monkeypatch.setattr(samplers.CFGGuider, "sample", samplers.CFGGuider.sample)
    # The hook wraps estimate_memory on install, so re-binding it here removes the wrapper at teardown.
    monkeypatch.setattr(sampler_helpers, "estimate_memory", sampler_helpers.estimate_memory)

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
    monkeypatch.setattr(sampling_lease, "_stage_lease", None)
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


class _LoggingLease(_FigureLease):
    """A lease that writes its acquires and releases into the shared call log, under a label."""

    def __init__(
        self,
        call_log: list[str],
        label: str,
        *,
        figure: float | None = None,
        acquire_returns: bool = True,
        on_acquire: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(figure=figure, acquire_returns=acquire_returns)
        self._call_log = call_log
        self._label = label
        self._on_acquire = on_acquire

    def acquire(self, block: bool = True, timeout: float | None = None) -> bool:
        self._call_log.append(f"{self._label} acquire")
        if self._on_acquire is not None:
            self._on_acquire()
        return super().acquire(block, timeout)

    def release(self) -> None:
        self._call_log.append(f"{self._label} release")
        super().release()


class _StageUploads:
    """Each call ``_sample_under_lease`` made to ``_stage_sampling_model``, logged in the shared call log."""

    def __init__(self, call_log: list[str]) -> None:
        self.calls: list[tuple[object, object, object, object]] = []
        self._call_log = call_log

    def __call__(self, guider: object, noise: object, conds: object, model_options: object) -> None:
        self._call_log.append("stage upload")
        self.calls.append((guider, noise, conds, model_options))


@pytest.fixture
def stage_uploads(monkeypatch: pytest.MonkeyPatch, guider_seam: _GuiderSeam) -> _StageUploads:
    """Replace the early upload with a recorder, so no model loads."""
    uploads = _StageUploads(guider_seam.call_log)
    monkeypatch.setattr(sampling_lease, "_stage_sampling_model", uploads)
    return uploads


@pytest.fixture
def lease_warnings() -> Iterator[list[str]]:
    """Collect the warnings the lease module logs during the test."""
    messages: list[str] = []
    handler_id = logger.add(
        lambda message: messages.append(message.record["message"]),
        level="WARNING",
        filter=sampling_lease.__name__,
    )
    yield messages
    logger.remove(handler_id)


class _FakeClock:
    """A monotonic clock the test advances by hand."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _guider_with_conds() -> Any:
    import comfy.samplers as samplers

    guider = samplers.CFGGuider(_FakeModelPatcher())
    guider.set_conds([], [])
    return guider


def test_stage_lease_granted_uploads_before_the_sampling_acquire(
    monkeypatch: pytest.MonkeyPatch,
    guider_seam: _GuiderSeam,
    rebases: list[float],
    stage_uploads: _StageUploads,
) -> None:
    monkeypatch.setattr(
        weight_prefetch,
        "prefetch_module_weights_async",
        lambda module, **kwargs: guider_seam.call_log.append("prefetch"),
    )
    stage = _LoggingLease(guider_seam.call_log, "stage", figure=9000.0)
    lease = _LoggingLease(guider_seam.call_log, "sampling", figure=7000.0)
    sampling_lease.install_sampling_lease_hook()
    sampling_lease.set_gpu_sampling_lease(lease)
    sampling_lease.set_gpu_stage_lease(stage)
    guider = _guider_with_conds()
    noise = torch.zeros(1, 4, 8, 8)

    guider.sample(noise, _latent(), None, _sigmas())

    assert guider_seam.call_log == [
        "prefetch",
        "stage acquire",
        "rebase",
        "stage upload",
        "stage release",
        "sampling acquire",
        "rebase",
        "sample",
        "sampling release",
    ]
    assert rebases == [9000.0, 7000.0], "the upload sizes against the stage grant, the sample against its own"
    assert stage_uploads.calls == [(guider, noise, guider.original_conds, guider.model_options)]
    assert stage.held == 0
    assert lease.held == 0


def test_stage_acquire_timeout_falls_through_to_the_sampling_acquire(
    monkeypatch: pytest.MonkeyPatch,
    guider_seam: _GuiderSeam,
    stage_uploads: _StageUploads,
    lease_warnings: list[str],
) -> None:
    clock = _FakeClock()
    monkeypatch.setattr(sampling_lease, "monotonic", clock)

    def _wait_out_the_stage() -> None:
        clock.now += 20.0

    stage = _LoggingLease(guider_seam.call_log, "stage", acquire_returns=False, on_acquire=_wait_out_the_stage)
    lease = _LoggingLease(guider_seam.call_log, "sampling")
    sampling_lease.install_sampling_lease_hook()
    sampling_lease.set_gpu_sampling_lease(lease, acquire_timeout_seconds=30.0)
    sampling_lease.set_gpu_stage_lease(stage)

    _guider_sample()

    assert guider_seam.call_log == ["stage acquire", "sampling acquire", "sample", "sampling release"]
    assert stage.acquire_calls == [(True, 30.0)]
    assert lease.acquire_calls == [(True, 10.0)]
    assert stage.release_count == 0, "a stage acquire that timed out holds nothing to release"
    assert stage_uploads.calls == []
    assert lease_warnings == ["GPU stage lease acquire timed out; the model loads at sampling"]


def test_stage_acquire_exception_falls_through_to_the_sampling_acquire(
    guider_seam: _GuiderSeam,
    stage_uploads: _StageUploads,
    lease_warnings: list[str],
) -> None:
    stage = _FakeLease(raise_on_acquire=True)
    lease = _FakeLease()
    sampling_lease.install_sampling_lease_hook()
    sampling_lease.set_gpu_sampling_lease(lease)
    sampling_lease.set_gpu_stage_lease(stage)

    _guider_sample()

    assert guider_seam.call_log == ["sample"]
    assert len(lease.acquire_calls) == 1
    assert lease.release_count == 1
    assert stage.release_count == 0
    assert stage_uploads.calls == []
    assert len(lease_warnings) == 1


def test_no_stage_lease_keeps_the_single_acquire(guider_seam: _GuiderSeam, stage_uploads: _StageUploads) -> None:
    # The same row as test_lease_acquired_around_and_released_after, with a stage lease set and then cleared.
    stage = _FakeLease()
    lease = _FakeLease()
    sampling_lease.install_sampling_lease_hook()
    sampling_lease.set_gpu_sampling_lease(lease, acquire_timeout_seconds=30.0)
    sampling_lease.set_gpu_stage_lease(stage)
    sampling_lease.set_gpu_stage_lease(None)

    _guider_sample()

    assert lease.acquire_calls == [(True, 30.0)]
    assert lease.release_count == 1
    assert guider_seam.call_log == ["sample"]
    assert stage.acquire_calls == []
    assert stage_uploads.calls == []


def test_stage_lease_without_a_sampling_lease_does_nothing(
    guider_seam: _GuiderSeam, stage_uploads: _StageUploads
) -> None:
    stage = _FakeLease()
    sampling_lease.install_sampling_lease_hook()
    sampling_lease.set_gpu_stage_lease(stage)

    _guider_sample()

    assert stage.acquire_calls == []
    assert stage_uploads.calls == []
    assert guider_seam.call_log == ["sample"]


def _install_fake_comfy(
    monkeypatch: pytest.MonkeyPatch,
    *,
    load_models_gpu: Callable[..., None],
    additional_models: list[object],
    hook_models: list[object],
    estimate: tuple[int, int] = (1000, 400),
) -> list[tuple[str, tuple[object, ...]]]:
    """Put a fake ``comfy`` with ``sampler_helpers`` and ``model_management`` in ``sys.modules``.

    Returns:
        The helper calls in order, each with its positional arguments.
    """
    helper_calls: list[tuple[str, tuple[object, ...]]] = []

    def _get_additional_models(conds: object, dtype: object) -> tuple[list[object], int]:
        helper_calls.append(("get_additional_models", (conds, dtype)))
        return list(additional_models), 100

    def _get_additional_models_from_model_options(model_options: object) -> list[object]:
        helper_calls.append(("get_additional_models_from_model_options", (model_options,)))
        return list(hook_models)

    def _estimate_memory(model: object, noise_shape: object, conds: object) -> tuple[int, int]:
        helper_calls.append(("estimate_memory", (model, noise_shape, conds)))
        return estimate

    sampler_helpers = types.ModuleType("comfy.sampler_helpers")
    sampler_helpers.get_additional_models = _get_additional_models  # type: ignore[attr-defined]
    sampler_helpers.get_additional_models_from_model_options = (  # type: ignore[attr-defined]
        _get_additional_models_from_model_options
    )
    sampler_helpers.estimate_memory = _estimate_memory  # type: ignore[attr-defined]
    model_management = types.ModuleType("comfy.model_management")
    model_management.load_models_gpu = load_models_gpu  # type: ignore[attr-defined]
    # Read by the text encoder release ahead of the acquires; nothing is loaded in the fake.
    model_management.current_loaded_models = []  # type: ignore[attr-defined]
    comfy_package = types.ModuleType("comfy")
    comfy_package.sampler_helpers = sampler_helpers  # type: ignore[attr-defined]
    comfy_package.model_management = model_management  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "comfy", comfy_package)
    monkeypatch.setitem(sys.modules, "comfy.sampler_helpers", sampler_helpers)
    monkeypatch.setitem(sys.modules, "comfy.model_management", model_management)
    return helper_calls


class _StagedModelPatcher:
    """The ``ModelPatcher`` surface ``_prepare_sampling`` reads before ``load_models_gpu``."""

    def __init__(self, nested_models: list[object]) -> None:
        self._nested_models = nested_models

    def model_dtype(self) -> str:
        return "model-dtype"

    def get_nested_additional_models(self) -> list[object]:
        return list(self._nested_models)


class _StagedGuider:
    def __init__(self, model_patcher: _StagedModelPatcher) -> None:
        self.model_patcher = model_patcher


def test_stage_upload_makes_the_load_call_prepare_sampling_makes(monkeypatch: pytest.MonkeyPatch) -> None:
    load_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def _load_models_gpu(*args: object, **kwargs: object) -> None:
        load_calls.append((args, kwargs))

    helper_calls = _install_fake_comfy(
        monkeypatch,
        load_models_gpu=_load_models_gpu,
        additional_models=["controlnet"],
        hook_models=["hook model"],
    )
    model_patcher = _StagedModelPatcher(nested_models=["nested model"])
    noise = torch.zeros(2, 4, 8, 8)
    conds: dict[str, list[object]] = {"positive": [], "negative": []}
    model_options: dict[str, object] = {"transformer_options": {}}

    sampling_lease._stage_sampling_model(_StagedGuider(model_patcher), noise, conds, model_options)

    assert helper_calls == [
        ("get_additional_models", (conds, "model-dtype")),
        ("get_additional_models_from_model_options", (model_options,)),
        ("estimate_memory", (model_patcher, noise.shape, conds)),
    ]
    assert load_calls == [
        (
            ([model_patcher, "controlnet", "hook model", "nested model"],),
            {"memory_required": 1100, "minimum_memory_required": 500, "force_full_load": False},
        )
    ]


def test_stage_upload_failure_releases_the_stage_and_samples(
    monkeypatch: pytest.MonkeyPatch,
    guider_seam: _GuiderSeam,
    lease_warnings: list[str],
) -> None:
    def _load_models_gpu(*args: object, **kwargs: object) -> None:
        raise RuntimeError("upload boom")

    stage = _LoggingLease(guider_seam.call_log, "stage")
    lease = _LoggingLease(guider_seam.call_log, "sampling")
    sampling_lease.install_sampling_lease_hook()
    sampling_lease.set_gpu_sampling_lease(lease)
    sampling_lease.set_gpu_stage_lease(stage)
    guider = _guider_with_conds()

    # The fake comfy goes in after the guider is built: the hooked sample and outer_sample are already bound.
    _install_fake_comfy(monkeypatch, load_models_gpu=_load_models_gpu, additional_models=[], hook_models=[])
    monkeypatch.setattr(guider.model_patcher, "model_dtype", lambda: "model-dtype", raising=False)
    monkeypatch.setattr(guider.model_patcher, "get_nested_additional_models", list, raising=False)

    guider.sample(torch.zeros(1, 4, 8, 8), _latent(), None, _sigmas())

    assert guider_seam.call_log == [
        "stage acquire",
        "stage release",
        "sampling acquire",
        "sample",
        "sampling release",
    ]
    assert stage.held == 0
    assert lease.held == 0
    assert lease_warnings == [
        "Early sampling model upload failed (RuntimeError: upload boom); the model loads at sampling"
    ]


@pytest.mark.parametrize(
    ("stage_wait_seconds", "expected_sampling_acquire"),
    [
        pytest.param(25.0, (True, 5.0), id="remainder"),
        pytest.param(30.0, (False, None), id="exhausted-non-blocking"),
        pytest.param(45.0, (False, None), id="overrun-non-blocking"),
    ],
)
def test_sampling_acquire_gets_what_the_stage_left_of_the_deadline(
    monkeypatch: pytest.MonkeyPatch,
    guider_seam: _GuiderSeam,
    stage_uploads: _StageUploads,
    stage_wait_seconds: float,
    expected_sampling_acquire: tuple[bool, float | None],
) -> None:
    clock = _FakeClock()
    monkeypatch.setattr(sampling_lease, "monotonic", clock)

    def _consume_budget() -> None:
        clock.now += stage_wait_seconds

    stage = _LoggingLease(guider_seam.call_log, "stage", on_acquire=_consume_budget)
    lease = _FakeLease()
    sampling_lease.install_sampling_lease_hook()
    sampling_lease.set_gpu_sampling_lease(lease, acquire_timeout_seconds=30.0)
    sampling_lease.set_gpu_stage_lease(stage)

    _guider_sample()

    assert stage.acquire_calls == [(True, 30.0)]
    assert lease.acquire_calls == [expected_sampling_acquire]
    assert len(stage_uploads.calls) == 1
    assert guider_seam.call_log[-1] == "sample"


_GIB = 2**30
_RESERVE_MB = 3117.0
_RESERVE_BYTES = int(_RESERVE_MB * 2**20)


class _ReserveLease(_FakeLease):
    """A lease whose grants carry the host's activation reserve, one queued figure per take, as a worker's does."""

    def __init__(self, reserves: list[float | None], *, acquire_returns: bool = True) -> None:
        super().__init__(acquire_returns=acquire_returns)
        self._reserves = list(reserves)
        self.take_calls = 0

    def take_cleared_sampling_reserve_mb(self) -> float | None:
        self.take_calls += 1
        return self._reserves.pop(0) if self._reserves else None


@pytest.fixture
def reserve_bounds() -> Iterator[list[str]]:
    """Collect the DEBUG lines the lease module logs when it bounds a load by the grant's activation reserve."""
    messages: list[str] = []
    handler_id = logger.add(
        lambda message: messages.append(message.record["message"]),
        level="DEBUG",
        filter=lambda record: record["name"] == sampling_lease.__name__ and "activation reserve" in record["message"],
    )
    yield messages
    logger.remove(handler_id)


@pytest.fixture
def comfy_estimates(monkeypatch: pytest.MonkeyPatch, guider_seam: _GuiderSeam) -> list[tuple[float, float]]:
    """Fix ComfyUI's estimate at 2 GiB (1 GiB minimum) and record what ``prepare_sampling`` would read inside sample.

    The stand-in ``outer_sample`` calls ``comfy.sampler_helpers.estimate_memory`` twice, as ``_prepare_sampling``
    resolves it (a module attribute looked up at call time), so the recorded figures are the ones its load would get.
    """
    import comfy.sampler_helpers as sampler_helpers

    monkeypatch.setattr(sampler_helpers, "estimate_memory", lambda model, noise_shape, conds: (2 * _GIB, 1 * _GIB))
    estimates: list[tuple[float, float]] = []

    def _outer_sample(
        self: object, noise: torch.Tensor, latent_image: torch.Tensor, *args: object, **kwargs: object
    ) -> torch.Tensor:
        for _ in range(2):
            estimates.append(sampler_helpers.estimate_memory(None, noise.shape, {}))
        guider_seam.call_log.append("sample")
        return latent_image

    guider_seam.replace_outer_sample(_outer_sample)
    return estimates


def test_grant_reserve_bounds_the_estimate_prepare_sampling_forms(
    guider_seam: _GuiderSeam,
    comfy_estimates: list[tuple[float, float]],
    reserve_bounds: list[str],
) -> None:
    import comfy.sampler_helpers as sampler_helpers

    lease = _ReserveLease(reserves=[_RESERVE_MB])
    sampling_lease.install_sampling_lease_hook()
    sampling_lease.set_gpu_sampling_lease(lease)

    _guider_sample()

    assert comfy_estimates == [(_RESERVE_BYTES, _RESERVE_BYTES)] * 2
    assert lease.take_calls == 1
    assert len(reserve_bounds) == 1, "the bound is logged once per grant"
    assert "comfy_estimate_mb=2048" in reserve_bounds[0]
    assert "reserve_mb=3117" in reserve_bounds[0]
    assert sampler_helpers.estimate_memory(None, (1,), {}) == (2 * _GIB, 1 * _GIB), "the bound ends with the sample"


def test_grant_reserve_below_comfys_estimate_leaves_it(
    guider_seam: _GuiderSeam,
    comfy_estimates: list[tuple[float, float]],
) -> None:
    lease = _ReserveLease(reserves=[512.0])
    sampling_lease.install_sampling_lease_hook()
    sampling_lease.set_gpu_sampling_lease(lease)

    _guider_sample()

    assert comfy_estimates == [(2 * _GIB, 1 * _GIB)] * 2


def test_lease_without_the_reserve_method_leaves_comfys_estimate(
    guider_seam: _GuiderSeam,
    rebases: list[float],
    comfy_estimates: list[tuple[float, float]],
) -> None:
    lease = _FigureLease(figure=12288.0)
    sampling_lease.install_sampling_lease_hook()
    sampling_lease.set_gpu_sampling_lease(lease)

    _guider_sample()

    assert comfy_estimates == [(2 * _GIB, 1 * _GIB)] * 2
    assert rebases == [12288.0]


def test_grant_reserve_is_taken_once_per_grant(
    guider_seam: _GuiderSeam,
    comfy_estimates: list[tuple[float, float]],
) -> None:
    # The proxy serves the figure once per grant, and the second grant here carries none.
    lease = _ReserveLease(reserves=[_RESERVE_MB, None])
    sampling_lease.install_sampling_lease_hook()
    sampling_lease.set_gpu_sampling_lease(lease)

    _guider_sample()
    _guider_sample()

    assert lease.take_calls == 2
    assert comfy_estimates == [(_RESERVE_BYTES, _RESERVE_BYTES)] * 2 + [(2 * _GIB, 1 * _GIB)] * 2


def test_timed_out_acquire_takes_no_reserve(
    guider_seam: _GuiderSeam,
    comfy_estimates: list[tuple[float, float]],
) -> None:
    lease = _ReserveLease(reserves=[_RESERVE_MB], acquire_returns=False)
    sampling_lease.install_sampling_lease_hook()
    sampling_lease.set_gpu_sampling_lease(lease)

    _guider_sample()

    assert lease.take_calls == 0
    assert comfy_estimates == [(2 * _GIB, 1 * _GIB)] * 2


@pytest.mark.parametrize(
    ("comfy_estimate", "expected_memory_required", "expected_minimum_memory_required"),
    [
        pytest.param((2 * _GIB, 1 * _GIB), _RESERVE_BYTES + 100, _RESERVE_BYTES + 100, id="reserve-larger"),
        pytest.param((4 * _GIB, 7 * _GIB // 2), 4 * _GIB + 100, 7 * _GIB // 2 + 100, id="comfy-larger"),
    ],
)
def test_stage_grant_reserve_bounds_the_staged_load(
    monkeypatch: pytest.MonkeyPatch,
    guider_seam: _GuiderSeam,
    comfy_estimate: tuple[int, int],
    expected_memory_required: int,
    expected_minimum_memory_required: int,
) -> None:
    load_calls: list[dict[str, object]] = []

    def _load_models_gpu(*args: object, **kwargs: object) -> None:
        load_calls.append(kwargs)

    stage = _ReserveLease(reserves=[_RESERVE_MB])
    lease = _FakeLease()
    sampling_lease.install_sampling_lease_hook()
    sampling_lease.set_gpu_sampling_lease(lease)
    sampling_lease.set_gpu_stage_lease(stage)
    guider = _guider_with_conds()

    # The fake comfy goes in after the guider is built: the hooked sample and outer_sample are already bound.
    _install_fake_comfy(
        monkeypatch,
        load_models_gpu=_load_models_gpu,
        additional_models=[],
        hook_models=[],
        estimate=comfy_estimate,
    )
    monkeypatch.setattr(guider.model_patcher, "model_dtype", lambda: "model-dtype", raising=False)
    monkeypatch.setattr(guider.model_patcher, "get_nested_additional_models", list, raising=False)

    guider.sample(torch.zeros(1, 4, 8, 8), _latent(), None, _sigmas())

    assert stage.take_calls == 1
    assert load_calls == [
        {
            "memory_required": expected_memory_required,
            "minimum_memory_required": expected_minimum_memory_required,
            "force_full_load": False,
        }
    ]
    assert guider_seam.call_log == ["sample"]
