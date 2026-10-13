"""Unit tests for ``hordelib.execution.upscale_dtype``.

Small randomly initialised DAT and ESRGAN models are built through spandrel on the CPU, so nothing here needs
model weights, a GPU or an initialised ComfyUI. The upscale node cannot be imported without ComfyUI, so the
tile path is exercised through the function the node hands to ComfyUI's tiler,
:func:`hordelib.execution.upscale_dtype.prepare_tile_upscaler`.
"""

from __future__ import annotations

import importlib
from typing import Protocol

import pytest
import torch

from hordelib.execution import upscale_dtype
from hordelib.execution.upscale_dtype import (
    BFLOAT16_APPROVED_ARCHITECTURE_IDS,
    UpscaleModelDescriptor,
    prepare_tile_upscaler,
    select_upscale_dtype,
    upscale_bfloat16_disabled,
    upscale_memory_element_size,
)

_TILE_SIZE = (40, 24)
"""A tile that is not the DAT test model's native size, so its shifted blocks build masks in the tile's dtype."""

_BFLOAT16_MAX_ABSOLUTE_DIFFERENCE = 0.02
"""Largest allowed difference between the bfloat16 and float32 outputs, on a [0, 1] scale.

bfloat16 keeps 8 significant bits, a relative rounding step of about 0.4%, and the test model's few layers
accumulate it to about 0.004 at the output. The bound is five times that, about 5 of 255 levels, and far below
the 0.3 or more that a wrong-dtype mask or a mis-cast weight produces.
"""


class _Descriptor(UpscaleModelDescriptor, Protocol):
    """The spandrel descriptor as the tests use it: the part hordelib reads, plus the module itself."""

    @property
    def model(self) -> torch.nn.Module:
        """The module the descriptor wraps."""
        ...


def _load(model: torch.nn.Module) -> _Descriptor:
    """Create the descriptor spandrel's loader builds for ``model``, as the upscale model loader does."""
    spandrel = importlib.import_module("spandrel")
    descriptor: _Descriptor = spandrel.ModelLoader().load_from_state_dict(model.state_dict()).eval()
    assert isinstance(descriptor, spandrel.ImageModelDescriptor)
    return descriptor


def _dat_descriptor() -> _Descriptor:
    """Create a small randomly initialised DAT with two shifted-window blocks; every call has the same weights."""
    torch.manual_seed(0)
    return _load(
        importlib.import_module("spandrel.architectures.DAT").DAT(
            img_size=16,
            in_chans=3,
            embed_dim=32,
            split_size=[2, 4],
            depth=[4, 2],
            num_heads=[2, 2],
            expansion_factor=2.0,
            upscale=2,
        ),
    )


def _esrgan_descriptor() -> _Descriptor:
    """Create a small randomly initialised ESRGAN, an architecture that declares bfloat16 support."""
    torch.manual_seed(0)
    return _load(
        importlib.import_module("spandrel.architectures.ESRGAN").ESRGAN(
            in_nc=3,
            out_nc=3,
            num_filters=8,
            num_blocks=1,
            scale=2,
        ),
    )


def _tile() -> torch.Tensor:
    generator = torch.Generator().manual_seed(1234)
    return torch.rand((1, 3, *_TILE_SIZE), generator=generator)


def _record_model_input_dtypes(descriptor: _Descriptor) -> list[torch.dtype]:
    """Record the dtype of every tensor the descriptor's module is called with."""
    seen: list[torch.dtype] = []

    def record(module: torch.nn.Module, arguments: tuple[torch.Tensor, ...]) -> None:
        seen.append(arguments[0].dtype)

    descriptor.model.register_forward_pre_hook(record)
    return seen


def _weight_addresses(descriptor: _Descriptor) -> list[int]:
    """Return where each weight's memory is; a cast allocates new memory, so an untouched model keeps these."""
    return [parameter.data_ptr() for parameter in descriptor.model.parameters()]


class TestDtypeRule:
    """bfloat16 only for an approved architecture whose descriptor declares support, with the switch on."""

    def test_dat_that_declares_support_runs_in_bfloat16(self) -> None:
        descriptor = _dat_descriptor()
        assert descriptor.supports_bfloat16

        assert select_upscale_dtype(descriptor, bfloat16_enabled=True) == torch.bfloat16

    def test_dat_that_does_not_declare_support_runs_in_float32(self) -> None:
        descriptor = _dat_descriptor()
        descriptor.supports_bfloat16 = False

        assert select_upscale_dtype(descriptor, bfloat16_enabled=True) == torch.float32

    def test_other_architecture_that_declares_support_runs_in_float32(self) -> None:
        descriptor = _esrgan_descriptor()
        assert descriptor.supports_bfloat16
        assert descriptor.architecture.id not in BFLOAT16_APPROVED_ARCHITECTURE_IDS

        assert select_upscale_dtype(descriptor, bfloat16_enabled=True) == torch.float32

    def test_switch_off_runs_dat_in_float32(self) -> None:
        descriptor = _dat_descriptor()

        assert select_upscale_dtype(descriptor, bfloat16_enabled=False) == torch.float32

    def test_architecture_is_read_from_the_descriptor(self) -> None:
        assert _dat_descriptor().architecture.id in BFLOAT16_APPROVED_ARCHITECTURE_IDS


class TestKillSwitch:
    """The environment variable turns the bfloat16 path off for the process; unset, the path is on."""

    def test_unset_leaves_bfloat16_on(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(upscale_dtype.DISABLE_ENV_VAR, raising=False)

        assert not upscale_bfloat16_disabled()

    @pytest.mark.parametrize("value", ["1", "true", "YES", " on "])
    def test_truthy_value_turns_bfloat16_off(self, monkeypatch: pytest.MonkeyPatch, value: str) -> None:
        monkeypatch.setenv(upscale_dtype.DISABLE_ENV_VAR, value)

        assert upscale_bfloat16_disabled()

    @pytest.mark.parametrize("value", ["", "0", "false"])
    def test_other_value_leaves_bfloat16_on(self, monkeypatch: pytest.MonkeyPatch, value: str) -> None:
        monkeypatch.setenv(upscale_dtype.DISABLE_ENV_VAR, value)

        assert not upscale_bfloat16_disabled()


class TestDatTilePath:
    """A DAT tile runs through the model in bfloat16 and comes back float32."""

    def test_model_sees_bfloat16_and_returns_a_close_float32_image(self) -> None:
        reference = _dat_descriptor()(_tile())
        descriptor = _dat_descriptor()
        dtype = select_upscale_dtype(descriptor, bfloat16_enabled=True)
        model_input_dtypes = _record_model_input_dtypes(descriptor)

        upscaled = prepare_tile_upscaler(descriptor, dtype)(_tile())

        assert descriptor.dtype == torch.bfloat16
        assert model_input_dtypes == [torch.bfloat16]
        assert upscaled.dtype == torch.float32
        assert upscaled.shape == reference.shape
        assert bool(torch.isfinite(upscaled).all())
        assert float(upscaled.min()) >= 0.0
        assert float(upscaled.max()) <= 1.0
        assert float((upscaled - reference).abs().max()) < _BFLOAT16_MAX_ABSOLUTE_DIFFERENCE
        assert not torch.equal(upscaled, reference)

    def test_switch_off_is_the_float32_call(self) -> None:
        reference = _dat_descriptor()(_tile())
        descriptor = _dat_descriptor()
        dtype = select_upscale_dtype(descriptor, bfloat16_enabled=False)
        model_input_dtypes = _record_model_input_dtypes(descriptor)

        upscaled = prepare_tile_upscaler(descriptor, dtype)(_tile())

        assert descriptor.dtype == torch.float32
        assert model_input_dtypes == [torch.float32]
        assert torch.equal(upscaled, reference)


class TestModelState:
    """The model is cast to the dtype the run needs whichever dtype it arrives in."""

    def test_model_already_in_bfloat16_is_used_as_it_is(self) -> None:
        descriptor = _dat_descriptor()
        first = prepare_tile_upscaler(descriptor, torch.bfloat16)(_tile())
        weights_before = _weight_addresses(descriptor)

        second = prepare_tile_upscaler(descriptor, torch.bfloat16)(_tile())

        assert _weight_addresses(descriptor) == weights_before
        assert torch.equal(second, first)

    def test_bfloat16_model_is_cast_back_for_a_float32_run(self) -> None:
        descriptor = _dat_descriptor()
        prepare_tile_upscaler(descriptor, torch.bfloat16)(_tile())
        model_input_dtypes = _record_model_input_dtypes(descriptor)

        upscaled = prepare_tile_upscaler(descriptor, torch.float32)(_tile())

        assert descriptor.dtype == torch.float32
        assert model_input_dtypes == [torch.float32]
        assert upscaled.dtype == torch.float32
        assert bool(torch.isfinite(upscaled).all())

    def test_float32_model_is_cast_for_a_bfloat16_run_after_a_float32_run(self) -> None:
        descriptor = _dat_descriptor()
        prepare_tile_upscaler(descriptor, torch.float32)(_tile())
        model_input_dtypes = _record_model_input_dtypes(descriptor)

        upscaled = prepare_tile_upscaler(descriptor, torch.bfloat16)(_tile())

        assert descriptor.dtype == torch.bfloat16
        assert model_input_dtypes == [torch.bfloat16]
        assert upscaled.dtype == torch.float32


class TestOtherArchitectureIsUnchanged:
    """An architecture outside the approved set gets ComfyUI's float32 call, value for value."""

    def test_tile_output_equals_the_stock_call(self) -> None:
        tile = _tile()
        stock_output = _esrgan_descriptor()(tile.float())
        descriptor = _esrgan_descriptor()
        weights_before = _weight_addresses(descriptor)
        dtype = select_upscale_dtype(descriptor, bfloat16_enabled=True)
        model_input_dtypes = _record_model_input_dtypes(descriptor)

        upscaled = prepare_tile_upscaler(descriptor, dtype)(tile)

        assert _weight_addresses(descriptor) == weights_before
        assert model_input_dtypes == [torch.float32]
        assert upscaled.dtype == stock_output.dtype
        assert torch.equal(upscaled, stock_output)


class TestMemoryElementSize:
    """The memory estimate's element size follows the dtype the tiles run in."""

    def test_float32_run_keeps_the_image_element_size(self) -> None:
        image = torch.zeros((1, 8, 8, 3), dtype=torch.float32)

        assert upscale_memory_element_size(image, torch.float32) == image.element_size()

    def test_bfloat16_run_halves_it(self) -> None:
        image = torch.zeros((1, 8, 8, 3), dtype=torch.float32)

        assert upscale_memory_element_size(image, torch.bfloat16) * 2 == image.element_size()
