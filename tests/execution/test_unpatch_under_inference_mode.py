"""Pins that the ModelPatcher.unpatch_model hijack restores backed-up weights inside ``torch.inference_mode``.

ComfyUI's ``partially_load`` detaches the patcher when a load fails mid-sample (a CUDA OOM while patching a
LoRA weight), which reaches ``unpatch_model`` with the sampler's inference mode still active. The backup entry
is the model's own weight, a normal tensor, and ``comfy.utils.set_attr_param`` wraps it in a ``Parameter``
without cloning while inference mode is on. For a comfy_kitchen ``QuantizedTensor`` that wrap raises
``Cannot set version_counter for inference tensor``, which replaces the error that caused the unpatch.
The patcher and module live on the CPU, so no GPU work is done.
"""

from __future__ import annotations

import collections

import pytest
import torch

from hordelib.execution import comfy_patches

_BackupEntry = collections.namedtuple("_BackupEntry", ["weight", "inplace_update"])


def _patcher_with_fp8_backup():
    """A real ModelPatcher over a one-layer module whose weight is backed up as ComfyUI's LoRA patching does."""
    base = pytest.importorskip("comfy_kitchen.tensor.base")
    pytest.importorskip("comfy_kitchen.tensor.fp8")
    from comfy.model_patcher import ModelPatcher

    module = torch.nn.Module()
    weight = base.QuantizedTensor.from_float(torch.randn(16, 16), "TensorCoreFP8Layout")
    module.weight = torch.nn.Parameter(weight, requires_grad=False)
    cpu = torch.device("cpu")
    patcher = ModelPatcher(module, load_device=cpu, offload_device=cpu)
    # patch_weight_to_device stores the module's own weight when the backup needs no copy.
    patcher.backup["weight"] = _BackupEntry(module.weight, False)
    return patcher, weight.dequantize()


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
