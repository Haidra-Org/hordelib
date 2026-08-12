"""Pins the component restore contract (``hordelib.execution.component_restore``).

Everything here runs on CPU against toy modules, so no GPU and no checkpoint are involved. Two
properties matter and neither is obvious from reading ComfyUI:

``ModelPatcher.clone`` shares the underlying module and the patch ``backup`` dict by reference, so a
LoRA clone patches the same weights the cached base holds, and ``partially_unload`` empties that shared
backup while leaving a ``LowVramPatch`` closing over the clone's patches on the shared module. The
residue tests build exactly that state and assert restoration clears all of it.

Tiling resets ``Conv2d.padding_mode``, and resetting to a hardcoded ``"zeros"`` is wrong for any
architecture whose convolutions were constructed otherwise (the pinned tree has some built with
``"replicate"``). The padding tests pin that a non-default original survives a tiling round trip.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

from hordelib.execution.component_restore import (
    capture_pristine_state,
    has_patch_residue,
    restore_component,
    restore_payload,
    restore_pristine_padding,
)

# The pinned ComfyUI checkout is put on the path directly rather than through hordelib.initialise:
# these cases need only ModelPatcher and comfy.ops, and the session init pulls in model downloads
# and a GPU that nothing here exercises.
_COMFY_PATH = Path(__file__).resolve().parents[2] / "ComfyUI"
if _COMFY_PATH.is_dir() and str(_COMFY_PATH) not in sys.path:
    sys.path.insert(0, str(_COMFY_PATH))

pytest.importorskip("comfy.model_patcher", reason="needs the pinned ComfyUI checkout")

import comfy.model_patcher as model_patcher
import comfy.ops

_DIM = 8
_LAYERS = 4
_CPU = torch.device("cpu")


class _ToyModel(torch.nn.Module):
    """The minimum surface ``ModelPatcher`` touches, over convolution-free linear layers.

    ``comfy.ops.manual_cast.Linear`` carries ``comfy_cast_weights``, which is what makes a module
    eligible for the lowvram offload branch inside ``partially_unload``. A plain ``torch.nn.Linear``
    would take the branch that leaves weights patched, so the residue this pins would never form.
    """

    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [comfy.ops.manual_cast.Linear(_DIM, _DIM, bias=False) for _ in range(_LAYERS)],
        )
        with torch.no_grad():
            for layer in self.layers:
                layer.weight.copy_(torch.eye(_DIM))
        self.device = _CPU
        self.model_lowvram = False
        self.lowvram_patch_counter = 0
        self.model_loaded_weight_memory = self.model_size()
        self.model_offload_buffer_memory = 0
        self.current_weight_patches_uuid = None

    def model_size(self) -> int:
        return _LAYERS * _DIM * _DIM * 4

    def is_dynamic(self) -> bool:
        return False

    def loaded_size(self) -> int:
        return self.model_loaded_weight_memory


_PRISTINE_WEIGHT_SUM = float(_LAYERS * _DIM)


def _weight_sum(toy: _ToyModel) -> float:
    return sum(float(layer.weight.detach().abs().sum()) for layer in toy.layers)


def _patched_base_with_residue() -> tuple[_ToyModel, model_patcher.ModelPatcher]:
    """Return a base patcher whose shared module carries a LoRA clone's leftover patch state."""
    toy = _ToyModel()
    base = model_patcher.ModelPatcher(toy, load_device=_CPU, offload_device=_CPU, size=toy.model_size())
    base.patch_model(device_to=_CPU, load_weights=True)

    lora = model_patcher.ModelPatcher.clone(base)
    lora.add_patches(
        {f"layers.{index}.weight": (torch.full((_DIM, _DIM), 0.25),) for index in range(_LAYERS)},
        1.0,
    )
    lora.patch_model(device_to=_CPU, load_weights=True)
    lora.partially_unload(_CPU, memory_to_free=toy.model_size())
    return toy, base


def test_a_clone_shares_the_base_module_and_its_backup() -> None:
    """The premise the contract exists for: cloning does not isolate the parent's weights."""
    toy = _ToyModel()
    base = model_patcher.ModelPatcher(toy, load_device=_CPU, offload_device=_CPU, size=toy.model_size())
    clone = model_patcher.ModelPatcher.clone(base)

    assert clone.model is base.model
    assert clone.backup is base.backup


def test_restore_clears_a_clones_patch_residue_from_the_shared_module() -> None:
    """Every artefact a LoRA clone's partial unload leaves on the shared module is cleared."""
    toy, base = _patched_base_with_residue()
    assert any(layer.weight_function for layer in toy.layers), "no residue formed; the test proves nothing"

    restore_component(base)

    assert _weight_sum(toy) == pytest.approx(_PRISTINE_WEIGHT_SUM)
    assert all(layer.weight_function == [] for layer in toy.layers)
    assert all(layer.bias_function == [] for layer in toy.layers)
    assert toy.model_lowvram is False
    assert not any(hasattr(layer, "comfy_patched_weights") for layer in toy.layers)
    assert toy.current_weight_patches_uuid is None


def test_restore_reports_the_device_bytes_given_up() -> None:
    """The return value is the weight memory the component stops claiming, which the reclaim path reports.

    It comes from ``model_loaded_weight_memory`` read before unpatching, so a component holding no loaded
    weights reports zero even when the restore cleared real residue. Zero is also the falsey answer for a
    caller that only wants to know whether anything happened.
    """
    toy, base = _patched_base_with_residue()
    toy.model_loaded_weight_memory = 4096

    assert restore_component(base) == 4096
    assert restore_component(None) == 0
    assert restore_component(object()) == 0


def test_restore_never_raises_on_an_unrestorable_component() -> None:
    """A component the contract cannot restore degrades to a no-op rather than faulting the job."""

    class _Hostile:
        def unpatch_model(self, device_to=None, unpatch_weights=True):
            raise RuntimeError("no")

    assert restore_component(_Hostile()) == 0
    assert restore_payload(("a", 1, None)) == 0


def test_restore_payload_covers_every_slot_shape() -> None:
    """Each load path stores a different tuple shape, so every slot is offered to the restorer."""
    toy_checkpoint, checkpoint_model = _patched_base_with_residue()
    toy_bare, bare_component = _patched_base_with_residue()
    toy_vae, standalone_vae = _patched_base_with_residue()
    for toy in (toy_checkpoint, toy_bare, toy_vae):
        toy.model_loaded_weight_memory = 2048

    assert restore_payload((checkpoint_model, None, None, None)) == 2048
    assert restore_payload((bare_component, None, None)) == 2048
    assert restore_payload((None, None, standalone_vae, None)) == 2048


def test_restore_payload_does_not_stop_at_the_first_restored_slot() -> None:
    """Reducing with a short-circuiting ``any`` would leave later slots patched."""
    toy_one, first = _patched_base_with_residue()
    toy_two, second = _patched_base_with_residue()

    restore_payload((first, second, None))

    assert _weight_sum(toy_one) == pytest.approx(_PRISTINE_WEIGHT_SUM)
    assert _weight_sum(toy_two) == pytest.approx(_PRISTINE_WEIGHT_SUM)
    assert all(layer.weight_function == [] for layer in toy_two.layers)


def test_residue_probe_sees_another_patchers_weights_on_the_shared_module() -> None:
    """A base whose shared module holds a LoRA clone's patches reports residue, and a restore clears it."""
    _, base = _patched_base_with_residue()

    assert has_patch_residue((base, None, None, None)) is True

    restore_component(base)

    assert has_patch_residue((base, None, None, None)) is False


def test_residue_probe_is_false_for_a_patchers_own_weights() -> None:
    """Holding the patch set the loading patcher itself asked for is not residue."""
    toy = _ToyModel()
    base = model_patcher.ModelPatcher(toy, load_device=_CPU, offload_device=_CPU, size=toy.model_size())
    base.patch_model(device_to=_CPU, load_weights=True)

    assert toy.current_weight_patches_uuid == base.patches_uuid
    assert has_patch_residue((base, None, None)) is False


def test_residue_probe_is_false_for_an_unloaded_patcher() -> None:
    """A module with no patch set applied records None, which is the clean answer."""
    toy = _ToyModel()
    base = model_patcher.ModelPatcher(toy, load_device=_CPU, offload_device=_CPU, size=toy.model_size())

    assert toy.current_weight_patches_uuid is None
    assert has_patch_residue((base, None, None)) is False


def test_residue_probe_answers_false_for_shapes_it_does_not_recognise() -> None:
    """An unfamiliar payload is reported clean rather than raising into a residency report."""

    class _Hostile:
        @property
        def patcher(self):
            raise RuntimeError("no")

    assert has_patch_residue(None) is False
    assert has_patch_residue((None, None, None)) is False
    assert has_patch_residue(("a", 1, object())) is False
    assert has_patch_residue(object()) is False
    assert has_patch_residue((_Hostile(), None, None)) is False


class _ConvModel(torch.nn.Module):
    """A module whose convolutions are built with a padding mode that is not torch's default."""

    def __init__(self, padding_mode: str) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 4, kernel_size=3, padding=1, padding_mode=padding_mode)
        self.nested = torch.nn.Sequential(
            torch.nn.Conv2d(4, 4, kernel_size=3, padding=1, padding_mode=padding_mode),
        )


class _VaeLike:
    """Stands in for a comfy VAE, which exposes its module as ``first_stage_model``."""

    def __init__(self, padding_mode: str) -> None:
        self.first_stage_model = _ConvModel(padding_mode)


@pytest.mark.parametrize("original", ["zeros", "replicate", "reflect"])
def test_capture_and_restore_returns_the_as_constructed_padding(original: str) -> None:
    """A tiling round trip returns each convolution to the padding it was built with.

    Parametrised over a non-default mode because resetting to a hardcoded ``"zeros"`` passes for
    ``zeros`` and silently changes what the other architectures decode.
    """
    component = _ConvModel(original)
    capture_pristine_state(component)

    for module in component.modules():
        if isinstance(module, torch.nn.Conv2d):
            module.padding_mode = "circular"

    restore_pristine_padding(component)

    assert component.conv.padding_mode == original
    assert component.nested[0].padding_mode == original


def test_padding_capture_reaches_a_vae_through_first_stage_model() -> None:
    """A VAE's convolutions are reachable for capture and reset, which is where tiling applies them."""
    vae = _VaeLike("replicate")
    capture_pristine_state(vae)
    vae.first_stage_model.conv.padding_mode = "circular"

    restore_pristine_padding(vae)

    assert vae.first_stage_model.conv.padding_mode == "replicate"


def test_capture_is_idempotent_and_never_records_a_tiled_state() -> None:
    """Re-capturing after tiling keeps the original, so a repeated capture cannot launder the tiled mode."""
    component = _ConvModel("replicate")
    capture_pristine_state(component)

    component.conv.padding_mode = "circular"
    capture_pristine_state(component)
    restore_pristine_padding(component)

    assert component.conv.padding_mode == "replicate"


def test_an_uncaptured_convolution_is_left_alone() -> None:
    """With nothing recorded, the current mode is the best account of the original, so it is not guessed at."""
    component = _ConvModel("reflect")

    restore_pristine_padding(component)

    assert component.conv.padding_mode == "reflect"


def test_residue_is_cleared_even_when_no_device_bytes_are_released() -> None:
    """A component holding no loaded weights still gets its residue cleared; only the byte count is zero.

    Reporting bytes rather than a flag would otherwise read as "nothing happened" for exactly the case
    the contract exists to handle: a patched component that has already been offloaded.
    """
    toy, base = _patched_base_with_residue()
    toy.model_loaded_weight_memory = 0
    assert any(layer.weight_function for layer in toy.layers), "no residue formed; the test proves nothing"

    assert restore_component(base) == 0

    assert all(layer.weight_function == [] for layer in toy.layers)
    assert toy.model_lowvram is False
