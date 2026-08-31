"""Weight page-in helpers and CPU-side weight retention across a device round trip."""

from __future__ import annotations

import pytest
import torch

from hordelib.execution.cpu_weight_retention import restore_cpu_origins, stash_cpu_origins
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
