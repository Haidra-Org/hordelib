"""Pins that a move to the CPU inside an unload points a plain op weight at its recorded CPU origin and copies nothing.

ComfyUI's plain op classes (``comfy.ops.disable_weight_init`` and its ``manual_cast`` and ``fp8_ops`` subclasses)
move through ``Module._apply``. hordelib gives them an ``_apply`` that, inside ``restoring_cpu_origins``, points each
unpatched device weight at the CPU tensor it was loaded from before torch's move runs, so the move finds the weight
already on the CPU. A weight is observed inside ComfyUI's ``partially_unload``, before hordelib's backstop release
runs, so a copy the move made is visible even though the backstop would free it afterwards.
"""

from __future__ import annotations

import collections
from collections.abc import Callable

import pytest
import torch

from hordelib.execution import comfy_patches, cpu_weight_retention
from hordelib.execution.cpu_weight_retention import stash_cpu_origins

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")

_FEATURES = 256
_CPU = torch.device("cpu")
_CUDA = torch.device("cuda")


def _plain_model() -> torch.nn.Sequential:
    import comfy.ops

    layers = []
    for _ in range(2):
        layer = comfy.ops.disable_weight_init.Linear(_FEATURES, _FEATURES)
        layer.weight = torch.nn.Parameter(torch.randn(_FEATURES, _FEATURES), requires_grad=False)
        layer.bias = torch.nn.Parameter(torch.randn(_FEATURES), requires_grad=False)
        layers.append(layer)
    return torch.nn.Sequential(*layers)


def _origin_pointers(model: torch.nn.Module) -> dict[str, int]:
    return {name: param.data.data_ptr() for name, param in model.named_parameters()}


def _moved_on_origin(model: torch.nn.Module, origins: dict[str, int]) -> dict[str, bool]:
    """Return, for each parameter now on the CPU, whether it is its recorded origin's storage."""
    return {
        name: param.data.data_ptr() == origins[name]
        for name, param in model.named_parameters()
        if param.device.type == "cpu"
    }


def _observe_inside_partially_unload(
    monkeypatch: pytest.MonkeyPatch,
    model: torch.nn.Module,
    origins: dict[str, int],
) -> dict[str, bool]:
    """Wrap ComfyUI's partially_unload so the moved weights are read before hordelib's backstop release runs."""
    observed: dict[str, bool] = {}
    original: Callable = comfy_patches._originals["model_patcher_partially_unload"]

    def _recording_partially_unload(*args, **kwargs):
        freed = original(*args, **kwargs)
        observed.update(_moved_on_origin(model, origins))
        return freed

    monkeypatch.setitem(comfy_patches._originals, "model_patcher_partially_unload", _recording_partially_unload)
    return observed


def _loaded_patcher(model: torch.nn.Module):
    from comfy.model_patcher import ModelPatcher

    patcher = ModelPatcher(model, load_device=_CUDA, offload_device=_CPU)
    comfy_patches._model_patcher_load_hijack(patcher, _CUDA, full_load=True)
    assert all(param.device.type == "cuda" for param in model.parameters())
    return patcher


def test_a_partial_unload_points_plain_weights_at_their_origins(
    init_horde: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _plain_model()
    origins = _origin_pointers(model)
    patcher = _loaded_patcher(model)
    observed = _observe_inside_partially_unload(monkeypatch, model, origins)
    torch.cuda.synchronize()
    allocated_before = torch.cuda.memory_allocated()

    freed = comfy_patches._model_patcher_partially_unload_hijack(patcher, _CPU, 1)

    assert freed > 0
    assert observed
    # Every weight the move put on the CPU was its origin before the backstop release ran: no copy was made.
    assert all(observed.values()), observed
    assert _moved_on_origin(model, origins) == observed
    moved_bytes = sum(param.nbytes for name, param in model.named_parameters() if name in observed)
    torch.cuda.synchronize()
    assert allocated_before - torch.cuda.memory_allocated() >= moved_bytes
    import comfy.model_management

    if comfy.model_management.MAX_PINNED_MEMORY > 0:
        # ComfyUI pins the registered weight, which is the origin itself.
        assert patcher.pinned
    patcher.unpin_all_weights()
    assert not patcher.pinned


def test_a_backed_up_key_is_left_to_comfy_in_a_partial_unload(
    init_horde: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A key in ``patcher.backup`` at the start of the unload is restored by ComfyUI and moved by torch's copy."""
    model = _plain_model()
    origins = _origin_pointers(model)
    patcher = _loaded_patcher(model)
    Dimension = collections.namedtuple("Dimension", ["weight", "inplace_update"])  # ComfyUI's backup entry shape
    patcher.backup["0.weight"] = Dimension(model[0].weight.data.to(_CPU), True)
    observed = _observe_inside_partially_unload(monkeypatch, model, origins)

    comfy_patches._model_patcher_partially_unload_hijack(patcher, _CPU, 1)
    patcher.unpin_all_weights()

    assert observed["0.weight"] is False
    assert observed["0.bias"] is True


def test_a_skipped_key_is_moved_by_torch_inside_the_scope(init_horde: None) -> None:
    model = _plain_model()
    origins = _origin_pointers(model)
    stash_cpu_origins(model)
    model.to(_CUDA)

    with cpu_weight_retention.restoring_cpu_origins(model, skip_keys={"1.weight"}):
        model.to(_CPU)

    on_origin = _moved_on_origin(model, origins)
    assert on_origin.pop("1.weight") is False
    assert all(on_origin.values()), on_origin


def test_a_weight_cast_on_the_device_is_copied(init_horde: None) -> None:
    """A device tensor of another dtype is not the origin's twin, so the move copies it back."""
    model = _plain_model()
    origins = _origin_pointers(model)
    stash_cpu_origins(model)
    model.to(_CUDA, torch.float16)

    with cpu_weight_retention.restoring_cpu_origins():
        model.to(_CPU)

    assert not any(_moved_on_origin(model, origins).values())
    assert all(param.dtype == torch.float16 for param in model.parameters())


def test_a_move_that_casts_on_the_way_to_the_cpu_is_copied(init_horde: None) -> None:
    model = _plain_model()
    origins = _origin_pointers(model)
    stash_cpu_origins(model)
    model.to(_CUDA)

    with cpu_weight_retention.restoring_cpu_origins():
        model.to(_CPU, torch.float16)

    assert not any(_moved_on_origin(model, origins).values())
    assert all(param.dtype == torch.float16 for param in model.parameters())


def test_a_device_move_inside_the_scope_does_not_restore(init_horde: None) -> None:
    model = _plain_model()
    stash_cpu_origins(model)
    model.to(_CUDA)

    with cpu_weight_retention.restoring_cpu_origins():
        model.to(_CUDA)

    assert all(param.device.type == "cuda" for param in model.parameters())


def test_a_move_outside_the_scope_is_unchanged(init_horde: None) -> None:
    model = _plain_model()
    origins = _origin_pointers(model)
    stash_cpu_origins(model)
    model.to(_CUDA)

    model.to(_CPU)

    assert not any(_moved_on_origin(model, origins).values())


def test_a_weight_without_a_record_is_copied(init_horde: None) -> None:
    model = _plain_model()
    model.to(_CUDA)

    with cpu_weight_retention.restoring_cpu_origins():
        model.to(_CPU)

    assert all(param.device.type == "cpu" for param in model.parameters())


def test_every_plain_op_class_carries_the_hijack_and_mixed_precision_keeps_its_route(init_horde: None) -> None:
    import comfy.ops

    plain_classes = comfy_patches.plain_op_classes()
    assert comfy.ops.disable_weight_init.Linear in plain_classes
    assert comfy.ops.disable_weight_init.Embedding in plain_classes
    for op_class in plain_classes:
        assert op_class._apply is comfy_patches._plain_op_apply_hijack
    assert comfy.ops.manual_cast.Conv2d._apply is comfy_patches._plain_op_apply_hijack
    assert comfy.ops.fp8_ops.Linear._apply is comfy_patches._plain_op_apply_hijack
    mixed_precision = comfy.ops.mixed_precision_ops({})
    assert mixed_precision.Linear._apply is not comfy_patches._plain_op_apply_hijack
