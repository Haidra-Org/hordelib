"""Pins that an unload registers a quantized weight's recorded CPU origin on its module and copies nothing back.

ComfyUI's quantized ops register a new ``Parameter`` on every move, so the attribute record a plain weight keeps
is lost, and a ``.data`` restore on a comfy_kitchen ``QuantizedTensor`` would move only its outer metadata. The
origin ``Parameter`` is kept in a table on its owning module instead, and a move back to the CPU inside an
unload registers it again. The weights go to the ``meta`` device (and to CUDA where one is present). A copy out of
``meta`` raises, so an unload that completes from ``meta`` made no device-to-host copy of any weight.
"""

from __future__ import annotations

import pytest
import torch

from hordelib.execution import comfy_patches
from hordelib.execution.cpu_weight_retention import restoring_quantized_origins, stash_cpu_origins

_DEVICES = [
    "meta",
    pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")),
]


def _quantized_weight(rows: int = 16, columns: int = 16) -> torch.Tensor:
    base = pytest.importorskip("comfy_kitchen.tensor.base")
    pytest.importorskip("comfy_kitchen.tensor.fp8")
    return base.QuantizedTensor.from_float(torch.randn(rows, columns), "TensorCoreFP8Layout")


class _QuantizedOp(torch.nn.Module):
    """A module that moves its weights as ComfyUI's mixed-precision ops do, through the module global."""

    def _apply(self, fn, recurse=True):
        import comfy.ops

        return comfy.ops._quantized_apply(self, fn, recurse)


def _quantized_op(*, with_bias: bool = False) -> _QuantizedOp:
    module = _QuantizedOp()
    module.weight = torch.nn.Parameter(_quantized_weight(), requires_grad=False)
    if with_bias:
        module.bias = torch.nn.Parameter(torch.randn(16), requires_grad=False)
    return module


@pytest.fixture
def quantized_apply_hijacked(init_horde: None, monkeypatch: pytest.MonkeyPatch) -> None:
    import comfy.ops

    monkeypatch.setattr(comfy.ops, "_quantized_apply", comfy_patches._quantized_apply_hijack)


@pytest.mark.parametrize("device", _DEVICES)
def test_a_full_unload_registers_the_quantized_origin_without_a_copy(
    quantized_apply_hijacked: None,
    device: str,
) -> None:
    from comfy.model_patcher import ModelPatcher

    # A plain weight's ``.data`` cannot be pointed from ``meta`` at a CPU tensor, so the bias joins on CUDA only.
    with_bias = device != "meta"
    model = _quantized_op(with_bias=with_bias)
    origin_weight = model.weight
    cpu = torch.device("cpu")
    patcher = ModelPatcher(model, load_device=torch.device(device), offload_device=cpu)
    origin_bias_data = model.bias.data if with_bias else None
    assert stash_cpu_origins(model) == (2 if with_bias else 1)
    model.to(device)
    assert model.weight is not origin_weight
    assert model.weight.device.type == device

    with torch.inference_mode():
        comfy_patches._model_patcher_unpatch_model_hijack(patcher, cpu, True)

    assert model.weight is origin_weight
    assert model.weight._qdata.data_ptr() == origin_weight._qdata.data_ptr()
    if origin_bias_data is not None:
        # The plain bias keeps the attribute record and its ``.data`` restore.
        assert model.bias.data.data_ptr() == origin_bias_data.data_ptr()


@pytest.mark.parametrize("device", _DEVICES)
def test_a_partial_unload_registers_the_quantized_origin_without_a_copy(
    quantized_apply_hijacked: None,
    device: str,
) -> None:
    from comfy.model_patcher import ModelPatcher

    model = torch.nn.Sequential(_quantized_op(), _quantized_op())
    origins = [layer.weight for layer in model]
    cpu = torch.device("cpu")
    patcher = ModelPatcher(model, load_device=torch.device(device), offload_device=cpu)
    comfy_patches._model_patcher_load_hijack(patcher, torch.device(device), full_load=True)
    assert all(layer.weight.device.type == device for layer in model)

    # Outside inference mode: ComfyUI sizes each module through ``state_dict``, which raises for a quantized
    # weight under it before any move is made.
    freed = comfy_patches._model_patcher_partially_unload_hijack(patcher, cpu, 1)

    assert freed > 0
    restored = [layer for layer, origin in zip(model, origins, strict=True) if layer.weight is origin]
    assert restored
    # Every weight the partial unload moved is its origin. None is a copy.
    assert all(layer in restored or layer.weight.device.type == device for layer in model)
    # Whatever the original pinned is the origin itself, and its unpin releases the registration.
    import comfy.model_management

    if comfy.model_management.MAX_PINNED_MEMORY > 0:
        assert patcher.pinned
        assert all(layer.weight.is_pinned() for layer in restored)
    patcher.unpin_all_weights()
    assert not patcher.pinned
    assert not any(layer.weight.is_pinned() for layer in restored)


def test_a_weight_without_a_record_still_moves_normally(quantized_apply_hijacked: None) -> None:
    model = _quantized_op()
    model.to("meta")

    with restoring_quantized_origins(), pytest.raises(NotImplementedError, match="meta"):
        model.to("cpu")


def test_a_replaced_device_weight_is_moved_and_not_restored(quantized_apply_hijacked: None) -> None:
    """A weight a patch replaced on the device is not the origin's device copy, so the copy-back runs for it."""
    model = _quantized_op()
    stash_cpu_origins(model)
    model.to("meta")
    model.weight = torch.nn.Parameter(model.weight.to("meta"), requires_grad=False)

    with restoring_quantized_origins(), pytest.raises(NotImplementedError, match="meta"):
        model.to("cpu")


def test_a_move_outside_an_unload_is_unchanged(quantized_apply_hijacked: None) -> None:
    model = _quantized_op()
    stash_cpu_origins(model)
    model.to("meta")

    with pytest.raises(NotImplementedError, match="meta"):
        model.to("cpu")


def test_a_device_to_device_move_inside_an_unload_does_not_restore(quantized_apply_hijacked: None) -> None:
    model = _quantized_op()
    origin_weight = model.weight
    stash_cpu_origins(model)
    model.to("meta")

    with restoring_quantized_origins():
        model.to("meta")

    assert model.weight is not origin_weight
    assert model.weight.device.type == "meta"


def test_a_repeat_stash_keeps_the_record(quantized_apply_hijacked: None) -> None:
    model = _quantized_op()
    assert stash_cpu_origins(model) == 1
    assert stash_cpu_origins(model) == 0
