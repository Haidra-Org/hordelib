"""Real-GPU test that unloading an fp8 checkpoint registers its CPU origins and copies no weight back.

A comfy_kitchen ``QuantizedTensor`` weight copied back from the device lands in new private, commit-charged
memory, the size of the whole model per unload. The unload instead registers each weight's recorded CPU origin
on its module (``cpu_weight_retention``). This loads Krea2-Turbo_fp8, runs one job with the UNet kept resident,
then unloads it partially and then fully, and asserts after each that the process's private bytes did not grow
by the size of the weights moved and that each moved weight's ``_qdata`` is its origin's storage.

Marked ``slow`` plus the checkpoint's model marker, and Windows only (``memory_info().private``). Run manually::

    uv run --no-sync pytest tests/test_quantized_origin_unload_gpu.py -m slow
"""

from __future__ import annotations

import gc
import sys

import psutil
import pytest
import torch

from hordelib.execution.cpu_weight_retention import _QUANTIZED_ORIGINS_ATTR, QuantizedOrigin
from hordelib.horde import HordeLib

_MB = 1024 * 1024
# The copy-back this guards against grows private memory by the whole moved size. Allocator and CUDA host
# bookkeeping move it by far less than a quarter of an fp8 UNet, or than the floor when little was moved.
_ALLOWED_GROWTH_FRACTION = 0.25
_ALLOWED_GROWTH_FLOOR_BYTES = 512 * _MB

pytestmark = [
    pytest.mark.slow,
    pytest.mark.default_krea2_turbo_model,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device"),
    pytest.mark.skipif(sys.platform != "win32", reason="reads Windows private bytes"),
]


def _private_bytes() -> int:
    gc.collect()
    return psutil.Process().memory_info().private


def _resident_quantized_weights() -> tuple[object, list[tuple[torch.nn.Module, str, QuantizedOrigin]]]:
    """Return the loaded patcher holding device copies of recorded quantized origins, with those records."""
    import comfy.model_management

    for loaded in comfy.model_management.current_loaded_models:
        patcher = loaded.model
        if patcher is None:
            continue
        records: list[tuple[torch.nn.Module, str, QuantizedOrigin]] = []
        for module in patcher.model.modules():
            table: dict[str, QuantizedOrigin] | None = module.__dict__.get(_QUANTIZED_ORIGINS_ATTR)
            if not table:
                continue
            for key, entry in table.items():
                if entry.is_device_twin(module._parameters[key]):
                    records.append((module, key, entry))
        if records:
            return patcher, records
    raise AssertionError("no loaded model holds a device copy of a recorded quantized CPU origin")


def _allowed_growth(moved_bytes: int) -> float:
    return max(moved_bytes * _ALLOWED_GROWTH_FRACTION, _ALLOWED_GROWTH_FLOOR_BYTES)


def _moved_back(records: list[tuple[torch.nn.Module, str, QuantizedOrigin]]) -> tuple[int, int]:
    """Return ``(count, bytes)`` of recorded weights now on the CPU, asserting each is its origin's storage."""
    count = 0
    moved_bytes = 0
    for module, key, entry in records:
        param = module._parameters[key]
        if param.device.type != "cpu":
            continue
        assert param is entry.parameter
        assert param._qdata.data_ptr() == entry.parameter._qdata.data_ptr()
        count += 1
        moved_bytes += entry.parameter.nbytes
    return count, moved_bytes


def test_an_fp8_unload_registers_cpu_origins_without_growing_private_memory(
    hordelib_instance: HordeLib,
    krea2_turbo_base_model_name: str,
) -> None:
    import comfy.model_management

    job = {
        "sampler_name": "er_sde",
        "cfg_scale": 1,
        "denoising_strength": 1.0,
        "seed": 1413,
        "height": 512,
        "width": 512,
        "karras": False,
        "tiling": False,
        "hires_fix": False,
        "clip_skip": 1,
        "control_type": None,
        "image_is_control": False,
        "return_control_map": False,
        "prompt": "a lighthouse on a cliff at dusk",
        "ddim_steps": 4,
        "n_iter": 1,
        "model": krea2_turbo_base_model_name,
    }
    result = hordelib_instance._inference(job, defer_vram_unload=True)
    assert result is not None

    patcher, records = _resident_quantized_weights()
    resident_bytes = sum(entry.parameter.nbytes for _, _, entry in records)
    print(f"{len(records)} quantized weight(s) resident on the device, {resident_bytes // _MB} MB")

    before_partial = _private_bytes()
    patcher.partially_unload(patcher.offload_device, memory_to_free=resident_bytes // 2)
    partial_count, partial_bytes = _moved_back(records)
    partial_growth = _private_bytes() - before_partial
    print(f"partial unload: {partial_count} weight(s), {partial_bytes // _MB} MB, private +{partial_growth // _MB} MB")
    assert partial_count > 0
    assert partial_growth < _allowed_growth(partial_bytes)

    before_full = _private_bytes()
    comfy.model_management.unload_all_models()
    full_count, full_bytes = _moved_back(records)
    full_growth = _private_bytes() - before_full
    print(f"full unload: {full_count} weight(s), {full_bytes // _MB} MB, private +{full_growth // _MB} MB")
    assert full_count == len(records)
    assert full_growth < _allowed_growth(full_bytes - partial_bytes)
