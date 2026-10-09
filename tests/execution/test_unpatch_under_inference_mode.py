"""Pins that unpatching and moving quantized weights inside ``torch.inference_mode`` does not raise.

ComfyUI's ``partially_load`` detaches the patcher when a load fails mid-sample (a CUDA OOM while patching a
LoRA weight), which reaches ``unpatch_model`` with the sampler's inference mode still active. The backup entry
is the model's own weight, a normal tensor, and ``comfy.utils.set_attr_param`` wraps it in a ``Parameter``
without cloning while inference mode is on. For a comfy_kitchen ``QuantizedTensor`` that wrap raises
``Cannot set version_counter for inference tensor``, which replaces the error that caused the unpatch.
The ``Module.to`` that ends the unpatch reaches ``comfy.ops._quantized_apply``, which wraps every parameter
the same way; its replacement leaves a parameter the move returned unchanged in place and carries the CPU
origin record across a rewrap. The patcher and modules live on the CPU, so no GPU work is done.
"""

from __future__ import annotations

import collections

import pytest
import torch

from hordelib.execution import comfy_patches
from hordelib.execution.cpu_weight_retention import _ORIGIN_ATTR, stash_cpu_origins

_BackupEntry = collections.namedtuple("_BackupEntry", ["weight", "inplace_update"])


def _quantized_weight(rows: int = 16, columns: int = 16):
    base = pytest.importorskip("comfy_kitchen.tensor.base")
    pytest.importorskip("comfy_kitchen.tensor.fp8")
    return base.QuantizedTensor.from_float(torch.randn(rows, columns), "TensorCoreFP8Layout")


def _patcher_with_fp8_backup():
    """A real ModelPatcher over a one-layer module whose weight is backed up as ComfyUI's LoRA patching does."""
    from comfy.model_patcher import ModelPatcher

    module = torch.nn.Module()
    weight = _quantized_weight()
    module.weight = torch.nn.Parameter(weight, requires_grad=False)
    cpu = torch.device("cpu")
    patcher = ModelPatcher(module, load_device=cpu, offload_device=cpu)
    # patch_weight_to_device stores the module's own weight when the backup needs no copy.
    patcher.backup["weight"] = _BackupEntry(module.weight, False)
    return patcher, weight.dequantize()


class _QuantizedOp(torch.nn.Module):
    """A module that moves its weights as ComfyUI's mixed-precision ops do, through the module global."""

    def _apply(self, fn, recurse=True):
        import comfy.ops

        return comfy.ops._quantized_apply(self, fn, recurse)


def _quantized_op() -> _QuantizedOp:
    module = _QuantizedOp()
    module.weight = torch.nn.Parameter(_quantized_weight(), requires_grad=False)
    return module


def test_an_unpatch_inside_inference_mode_restores_a_quantized_backup(init_horde: None) -> None:
    patcher, expected = _patcher_with_fp8_backup()

    with torch.inference_mode():
        comfy_patches._model_patcher_unpatch_model_hijack(patcher, None, True)

    assert not patcher.backup
    assert torch.equal(patcher.model.weight.dequantize(), expected)


def test_an_unpatch_outside_inference_mode_restores_the_backup_itself(init_horde: None) -> None:
    patcher, expected = _patcher_with_fp8_backup()
    backed_up = patcher.backup["weight"].weight

    comfy_patches._model_patcher_unpatch_model_hijack(patcher, None, True)

    assert not patcher.backup
    assert patcher.model.weight._qdata.data_ptr() == backed_up._qdata.data_ptr()
    assert torch.equal(patcher.model.weight.dequantize(), expected)


def test_a_no_op_quantized_move_inside_inference_mode_keeps_the_parameter(init_horde: None) -> None:
    module = _quantized_op()
    registered = module.weight

    with torch.inference_mode():
        comfy_patches._quantized_apply_hijack(module, lambda tensor: tensor.to("cpu"))

    assert module.weight is registered


def test_a_rewrap_carries_the_cpu_origin_record_of_a_plain_parameter_only(init_horde: None) -> None:
    module = _quantized_op()
    module.bias = torch.nn.Parameter(torch.randn(16), requires_grad=False)
    assert stash_cpu_origins(module) == 2
    bias_origin = getattr(module.bias, _ORIGIN_ATTR)
    previous_bias = module.bias
    previous_weight = module.weight

    comfy_patches._quantized_apply_hijack(module, lambda tensor: tensor.clone())

    assert module.bias is not previous_bias
    assert getattr(module.bias, _ORIGIN_ATTR) is bias_origin
    # Restoring assigns .data, which leaves a QuantizedTensor's payload in place, so its record is not carried.
    assert module.weight is not previous_weight
    assert getattr(module.weight, _ORIGIN_ATTR, None) is None


def test_a_failed_partial_load_inside_inference_mode_raises_the_out_of_memory_error(
    init_horde: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import comfy.ops
    from comfy.model_patcher import ModelPatcher

    monkeypatch.setattr(comfy.ops, "_quantized_apply", comfy_patches._quantized_apply_hijack)
    cpu = torch.device("cpu")
    patcher = ModelPatcher(_quantized_op(), load_device=cpu, offload_device=cpu)
    # Sized before the sampler starts, as model loading does; state_dict() inside inference mode raises.
    patcher.model_size()

    def _load_out_of_memory(*args, **kwargs) -> None:
        raise torch.OutOfMemoryError("CUDA out of memory")

    monkeypatch.setattr(patcher, "load", _load_out_of_memory)

    with torch.inference_mode(), pytest.raises(torch.OutOfMemoryError):
        patcher.partially_load(cpu)
