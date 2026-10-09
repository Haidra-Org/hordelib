"""Weight page-in helpers, the background prefetch's lifetime, and CPU-side weight retention."""

from __future__ import annotations

import gc
import threading
import time
import weakref
from collections.abc import Callable, Generator
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from hordelib.execution.component_cache import (
    ComponentCache,
    ComponentCacheEntry,
    ComponentCacheKey,
    ComponentSlotKind,
)
from hordelib.execution.cpu_weight_retention import (
    release_copied_back_weights,
    restore_cpu_origins,
    stash_cpu_origins,
)
from hordelib.execution.weight_prefetch import collect_cpu_weight_ranges, prefetch_ranges, touch_cpu_weights


def _module() -> torch.nn.Module:
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Linear(64, 64), torch.nn.Conv2d(3, 4, 3)).half()


def test_collect_cpu_weight_ranges_covers_every_weight_byte() -> None:
    module = _module()
    ranges = collect_cpu_weight_ranges(module)
    total = sum(length for _, length in ranges)
    expected = sum(p.numel() * p.element_size() for p in module.parameters())
    assert total >= expected
    assert all(length > 0 for _, length in ranges)


def test_prefetch_and_touch_are_no_ops_on_resident_private_memory() -> None:
    module = _module()
    assert prefetch_ranges(collect_cpu_weight_ranges(module)) in (True, False)
    touched = touch_cpu_weights(p.data for p in module.parameters())
    assert touched == sum(p.numel() * p.element_size() for p in module.parameters())


def test_touch_skips_non_contiguous_and_empty_tensors() -> None:
    base = torch.zeros(8, 8)
    assert touch_cpu_weights([base.t(), torch.empty(0)]) == 0


def test_stash_records_cpu_tensors_once() -> None:
    module = _module()
    first = stash_cpu_origins(module)
    assert first == len(list(module.parameters()))
    assert stash_cpu_origins(module) == 0


def test_restore_is_a_no_op_while_weights_are_still_on_cpu() -> None:
    module = _module()
    stash_cpu_origins(module)
    assert restore_cpu_origins(module) == (0, 0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_restore_returns_the_recorded_cpu_tensors_after_a_device_round_trip() -> None:
    module = _module()
    origins = {name: p.data for name, p in module.named_parameters()}
    stash_cpu_origins(module)
    module.to("cuda")
    # A patched key is left to the caller: it keeps its device tensor and is restored by the caller's backup.
    restored, restored_bytes = restore_cpu_origins(module, skip_keys={"0.bias"})
    expected_names = [name for name, _ in module.named_parameters() if name != "0.bias"]
    assert restored == len(expected_names)
    assert restored_bytes == sum(origins[name].numel() * origins[name].element_size() for name in expected_names)
    for name, param in module.named_parameters():
        if name == "0.bias":
            assert param.data.device.type == "cuda"
            continue
        assert param.data.device.type == "cpu"
        assert param.data.data_ptr() == origins[name].data_ptr()
        assert torch.equal(param.data, origins[name])
    module.to("cpu")
    assert restore_cpu_origins(module) == (0, 0)


def test_release_returns_copied_back_weights_to_their_origins_and_unpins_them() -> None:
    """A CPU copy of a recorded origin (what a partial unload leaves) is unpinned, then replaced by the origin."""
    module = _module()
    origins = {name: p.data for name, p in module.named_parameters()}
    stash_cpu_origins(module)
    for name, param in module.named_parameters():
        if name != "0.bias":
            param.data = param.data.clone()
    unpinned: list[str] = []
    released, released_bytes = release_copied_back_weights(module, skip_keys={"1.bias"}, unpin=unpinned.append)
    expected = [name for name, _ in module.named_parameters() if name not in ("0.bias", "1.bias")]
    assert released == len(expected)
    assert released_bytes == sum(origins[name].numel() * origins[name].element_size() for name in expected)
    assert unpinned == expected
    for name, param in module.named_parameters():
        if name == "1.bias":
            assert param.data.data_ptr() != origins[name].data_ptr()
            continue
        assert param.data.data_ptr() == origins[name].data_ptr()
    assert release_copied_back_weights(module, skip_keys={"1.bias"}) == (0, 0)


def test_release_leaves_a_weight_whose_shape_changed() -> None:
    module = _module()
    stash_cpu_origins(module)
    module[0].weight.data = torch.zeros(2, 2, dtype=torch.float16)
    assert release_copied_back_weights(module) == (0, 0)


def test_kill_switches_disable_both_mechanisms(monkeypatch: pytest.MonkeyPatch) -> None:
    from hordelib.execution import cpu_weight_retention, weight_prefetch

    monkeypatch.setenv("HORDELIB_DISABLE_WEIGHT_PREFETCH", "1")
    monkeypatch.setenv("HORDELIB_DISABLE_CPU_WEIGHT_RETENTION", "1")
    assert weight_prefetch.weight_prefetch_disabled()
    assert cpu_weight_retention.cpu_weight_retention_disabled()
    module = _module()
    assert stash_cpu_origins(module) == 0
    assert weight_prefetch.prefetch_module_weights_async(module) is None


class _WrapperWeight(torch.Tensor):
    """A minimal ``__tensor_flatten__`` wrapper whose dispatch path is forbidden.

    Mirrors a quantized weight: the wrapper reports a logical compute dtype while the real bytes live in an
    inner plain tensor. Any op reaching ``__torch_dispatch__`` would, in the real layout classes, fall back
    to materialising a dense dequantized copy, so the helpers must reach the inner tensor without ever
    dispatching through the wrapper.
    """

    @staticmethod
    def __new__(cls, qdata: torch.Tensor) -> _WrapperWeight:
        return torch.Tensor._make_wrapper_subclass(cls, qdata.shape, dtype=torch.bfloat16, device=qdata.device)

    def __init__(self, qdata: torch.Tensor) -> None:
        self._qdata = qdata

    def __tensor_flatten__(self) -> tuple[list[str], None]:
        return ["_qdata"], None

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        raise AssertionError(f"a weight page-in helper dispatched {func} through the wrapper")


def test_touch_unwraps_tensor_subclasses_without_dispatching() -> None:
    inner = torch.zeros(256, 256, dtype=torch.uint8)
    touched = touch_cpu_weights([_WrapperWeight(inner)])
    assert touched == inner.numel() * inner.element_size()


def test_collect_ranges_covers_a_wrapped_weight_via_its_inner_storage() -> None:
    inner = torch.zeros(256, 256, dtype=torch.uint8)
    module = torch.nn.Module()
    module.register_buffer("weight", _WrapperWeight(inner))
    ranges = collect_cpu_weight_ranges(module)
    total = sum(length for _, length in ranges)
    assert total >= inner.numel() * inner.element_size()


def test_touch_never_dequantizes_a_real_comfy_kitchen_fp8_weight(caplog: pytest.LogCaptureFixture) -> None:
    """Bump canary: page-touching a comfy_kitchen fp8 weight must not hit the dequantization fallback.

    The fallback logs "Unhandled op ..., dequantizing" and materialises a dense compute-dtype copy, which
    for a whole checkpoint is twice the file's bytes and tens of seconds of CPU per touch pass. A
    comfy_kitchen or torch bump that changes the wrapper protocol or the dispatch tables could
    reintroduce that silently; this pins the contract against the installed library.
    """
    pytest.importorskip("comfy_kitchen.tensor.fp8")
    base = pytest.importorskip("comfy_kitchen.tensor.base")

    weight = base.QuantizedTensor.from_float(torch.randn(128, 128), "TensorCoreFP8Layout")
    with caplog.at_level("DEBUG", logger="comfy_kitchen"):
        touched = touch_cpu_weights([weight])

    assert touched > 0
    assert not any("dequantizing" in record.message for record in caplog.records)


class _GatedTouch:
    """Blocks the background touch on its first tensor until released, counting the bytes it touches."""

    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()
        self.calls = 0
        self.touched = 0

    def __call__(self, tensors: Any) -> int:
        self.calls += 1
        self.entered.set()
        assert self.release.wait(timeout=30)
        touched = touch_cpu_weights(tensors)
        self.touched += touched
        return touched


@pytest.fixture
def gated_touch(monkeypatch: pytest.MonkeyPatch) -> Generator[_GatedTouch, None, None]:
    from hordelib.execution import weight_prefetch

    monkeypatch.delenv("HORDELIB_DISABLE_WEIGHT_PREFETCH", raising=False)
    gate = _GatedTouch()
    monkeypatch.setattr(weight_prefetch, "touch_cpu_weights", gate)
    yield gate
    gate.release.set()


def _wait_until(condition: Callable[[], bool], *, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        gc.collect()
        if condition():
            return True
        time.sleep(0.01)
    return condition()


def test_an_evicted_entry_is_collectable_while_its_prefetch_runs(gated_touch: _GatedTouch) -> None:
    """Eviction signals the entry's prefetch, and the module dies with the thread still mid-read.

    The thread holds the module weakly, so nothing waits on it. A strong hold kept an evicted model and its
    file mapping committed beside the replacement load for as long as the read took.
    """
    from hordelib.execution.weight_prefetch import prefetch_module_weights_async

    cache = ComponentCache(budget_mb=1000)
    module = _module()
    entry = ComponentCacheEntry(
        key=ComponentCacheKey(ComponentSlotKind.UNET, "model_a:unet"),
        payload=(SimpleNamespace(model=module), None, None),
        approx_ram_mb=1.0,
        source_ckpt_path="model_a.safetensors",
    )
    cache.put(entry)
    module_ref = weakref.ref(module)
    stop = entry.prefetch_stop
    thread = prefetch_module_weights_async(module, label="model_a:unet", stop=stop)
    assert thread is not None
    try:
        assert gated_touch.entered.wait(timeout=10)
        cache.evict_all()
        del module, entry

        assert stop.is_set()
        assert _wait_until(lambda: module_ref() is None)
        assert thread.is_alive()
    finally:
        gated_touch.release.set()
        thread.join(timeout=10)
    assert not thread.is_alive()
    assert gated_touch.calls == 1


def test_a_stopped_prefetch_ends_at_the_next_tensor(gated_touch: _GatedTouch) -> None:
    """A set stop flag ends the read even while the module is still alive."""
    from hordelib.execution.weight_prefetch import prefetch_module_weights_async

    module = _module()
    stop = threading.Event()
    thread = prefetch_module_weights_async(module, label="stopped", stop=stop)
    assert thread is not None
    assert gated_touch.entered.wait(timeout=10)
    stop.set()
    gated_touch.release.set()
    thread.join(timeout=10)

    assert not thread.is_alive()
    assert gated_touch.calls == 1
    assert len(list(module.parameters())) > 1


def test_an_unstopped_prefetch_touches_every_weight(gated_touch: _GatedTouch) -> None:
    """A live module with no stop request is read to the end, so the weakref costs no prefetch coverage."""
    from hordelib.execution.weight_prefetch import prefetch_module_weights_async

    module = _module()
    gated_touch.release.set()
    thread = prefetch_module_weights_async(module, label="complete")
    assert thread is not None
    thread.join(timeout=10)

    assert not thread.is_alive()
    assert gated_touch.touched == sum(p.numel() * p.element_size() for p in module.parameters())
