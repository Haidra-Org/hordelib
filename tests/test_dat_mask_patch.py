"""Unit tests for ``hordelib.execution.dat_mask_patch``.

A small randomly initialised DAT is built through spandrel's architecture class on the CPU, so nothing here
needs model weights, a GPU or an initialised ComfyUI.
"""

from __future__ import annotations

import functools
import importlib
from collections.abc import Callable, Iterator
from types import ModuleType

import pytest
import torch
from loguru import logger

from hordelib.execution import dat_mask_patch
from hordelib.execution.dat_mask_patch import (
    apply_dat_mask_patch,
    is_dat_mask_patch_applied,
    remove_dat_mask_patch,
)

_NATIVE_SIZE = 16
"""The ``img_size`` the test model is built for; its shifted blocks register masks for this size."""

_SQUARE_TILE = (32, 32)
_STRIP_TILE = (48, 16)
_UNPADDED_TILE = (18, 22)
"""A tile that is not a multiple of the window, so the block pads it before masking."""
_NATIVE_TILE = (_NATIVE_SIZE, _NATIVE_SIZE)

_SHIFTED_BLOCK_COUNT = 2
"""Shifted-window blocks in the test model: block 2 of the first group and block 0 of the second."""

MaskCall = tuple[int, int]


@pytest.fixture
def dat_architecture() -> ModuleType:
    """Return spandrel's DAT architecture module."""
    return importlib.import_module(dat_mask_patch.DAT_ARCHITECTURE_MODULE)


@pytest.fixture
def stock_mask_calls(monkeypatch: pytest.MonkeyPatch, dat_architecture: ModuleType) -> list[MaskCall]:
    """Record every run of spandrel's CPU mask construction as its ``(height, width)``.

    The recorder wraps the stock method before any patch is applied, so a patch that captures the method as
    its original captures the recorder with it.
    """
    attention_class = dat_architecture.Adaptive_Spatial_Attention
    stock_calculate_mask = attention_class.calculate_mask
    calls: list[MaskCall] = []

    @functools.wraps(stock_calculate_mask)
    def recording_calculate_mask(
        self: torch.nn.Module,
        H: int,
        W: int,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        calls.append((H, W))
        return stock_calculate_mask(self, H, W, dtype=dtype)

    monkeypatch.setattr(attention_class, "calculate_mask", recording_calculate_mask)
    return calls


@pytest.fixture
def unpatched(stock_mask_calls: list[MaskCall]) -> Iterator[None]:
    """Leave spandrel as the test found it, whatever the test applied."""
    assert not is_dat_mask_patch_applied()
    yield
    remove_dat_mask_patch()


@pytest.fixture
def warnings_logged() -> Iterator[list[str]]:
    """Collect the warning-level log messages emitted during the test."""
    messages: list[str] = []
    sink_id = logger.add(lambda message: messages.append(str(message)), level="WARNING", format="{message}")
    yield messages
    logger.remove(sink_id)


def _build_model(dat_architecture: ModuleType) -> torch.nn.Module:
    """Create a small randomly initialised DAT with three spatial blocks, two of them shifted."""
    torch.manual_seed(0)
    model = dat_architecture.DAT(
        img_size=_NATIVE_SIZE,
        in_chans=3,
        embed_dim=32,
        split_size=[2, 4],
        depth=[4, 2],
        num_heads=[2, 2],
        expansion_factor=2.0,
        upscale=2,
    )
    return model.eval()


def _tile(size: tuple[int, int]) -> torch.Tensor:
    generator = torch.Generator().manual_seed(size[0] * 1000 + size[1])
    return torch.rand((1, 3, size[0], size[1]), generator=generator)


@torch.inference_mode()
def _upscale(model: torch.nn.Module, size: tuple[int, int]) -> torch.Tensor:
    return model(_tile(size))


class TestOutputIsUnchanged:
    """The patch changes where the masks are built and nothing about the image."""

    @pytest.mark.parametrize(
        "size",
        [_SQUARE_TILE, _STRIP_TILE, _UNPADDED_TILE, _NATIVE_TILE],
        ids=["square", "strip", "unpadded", "native"],
    )
    def test_patched_output_equals_stock_output(
        self,
        dat_architecture: ModuleType,
        unpatched: None,
        size: tuple[int, int],
    ) -> None:
        model = _build_model(dat_architecture)
        stock_output = _upscale(model, size)

        assert apply_dat_mask_patch()
        patched_output = _upscale(model, size)

        assert patched_output.dtype == stock_output.dtype
        assert torch.equal(patched_output, stock_output)

    def test_registered_masks_are_built_by_stock_code(
        self,
        dat_architecture: ModuleType,
        unpatched: None,
        stock_mask_calls: list[MaskCall],
    ) -> None:
        stock_model = _build_model(dat_architecture)
        assert apply_dat_mask_patch()
        stock_mask_calls.clear()

        patched_model = _build_model(dat_architecture)

        assert stock_mask_calls == [_NATIVE_TILE] * _SHIFTED_BLOCK_COUNT
        stock_buffers = dict(stock_model.named_buffers())
        patched_buffers = dict(patched_model.named_buffers())
        mask_names = [name for name in stock_buffers if "attn_mask" in name]
        assert len(mask_names) == 2 * _SHIFTED_BLOCK_COUNT
        for mask_name in mask_names:
            assert torch.equal(patched_buffers[mask_name], stock_buffers[mask_name])


class TestMaskConstruction:
    """Stock spandrel builds the masks on the CPU in every shifted block of every forward pass."""

    def test_stock_builds_masks_once_per_shifted_block(
        self,
        dat_architecture: ModuleType,
        unpatched: None,
        stock_mask_calls: list[MaskCall],
    ) -> None:
        model = _build_model(dat_architecture)
        stock_mask_calls.clear()

        _upscale(model, _SQUARE_TILE)

        assert stock_mask_calls == [_SQUARE_TILE] * _SHIFTED_BLOCK_COUNT

    @pytest.mark.parametrize("size", [_SQUARE_TILE, _STRIP_TILE, _UNPADDED_TILE], ids=["square", "strip", "unpadded"])
    def test_patched_forward_never_runs_the_cpu_construction(
        self,
        dat_architecture: ModuleType,
        unpatched: None,
        stock_mask_calls: list[MaskCall],
        size: tuple[int, int],
    ) -> None:
        model = _build_model(dat_architecture)
        apply_dat_mask_patch()
        stock_mask_calls.clear()

        _upscale(model, size)
        _upscale(model, size)

        assert stock_mask_calls == []

    def test_native_size_forward_uses_the_registered_masks(
        self,
        dat_architecture: ModuleType,
        unpatched: None,
        stock_mask_calls: list[MaskCall],
    ) -> None:
        model = _build_model(dat_architecture)
        assert apply_dat_mask_patch()
        patched_calculate_mask = dat_architecture.Adaptive_Spatial_Attention.calculate_mask
        patched_calls: list[MaskCall] = []

        def recording_patched_calculate_mask(
            self: torch.nn.Module,
            H: int,
            W: int,
            dtype: torch.dtype | None = None,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            patched_calls.append((H, W))
            return patched_calculate_mask(self, H, W, dtype=dtype)

        dat_architecture.Adaptive_Spatial_Attention.calculate_mask = recording_patched_calculate_mask
        try:
            _upscale(model, _NATIVE_TILE)
        finally:
            dat_architecture.Adaptive_Spatial_Attention.calculate_mask = patched_calculate_mask

        assert patched_calls == []

    def test_built_masks_take_the_requested_dtype(self, dat_architecture: ModuleType, unpatched: None) -> None:
        model = _build_model(dat_architecture)
        shifted_block = next(
            module
            for module in model.modules()
            if isinstance(module, dat_architecture.Adaptive_Spatial_Attention) and module.attn_mask_0 is not None
        )
        stock_masks = shifted_block.calculate_mask(*_STRIP_TILE, dtype=torch.bfloat16)
        assert apply_dat_mask_patch()

        patched_masks = shifted_block.calculate_mask(*_STRIP_TILE, dtype=torch.bfloat16)

        for patched_mask, stock_mask in zip(patched_masks, stock_masks, strict=True):
            assert patched_mask.dtype == torch.bfloat16
            assert patched_mask.device == shifted_block.qkv.weight.device
            assert torch.equal(patched_mask, stock_mask)


class TestPatchLifecycle:
    """Applying is idempotent and removal restores the method that was there."""

    def test_apply_twice_then_remove(self, dat_architecture: ModuleType, unpatched: None) -> None:
        attention_class = dat_architecture.Adaptive_Spatial_Attention
        method_before = attention_class.calculate_mask

        assert apply_dat_mask_patch()
        method_patched = attention_class.calculate_mask
        assert method_patched is not method_before
        assert is_dat_mask_patch_applied()

        assert apply_dat_mask_patch()
        assert attention_class.calculate_mask is method_patched

        assert remove_dat_mask_patch()
        assert attention_class.calculate_mask is method_before
        assert not is_dat_mask_patch_applied()
        assert not remove_dat_mask_patch()


def _mask_method_with_device_parameter(
    self: torch.nn.Module,
    H: int,
    W: int,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """A mask method whose signature has grown a parameter the patch does not know."""
    raise AssertionError("the patch must not call a method it does not recognise")


def _mask_method_with_different_values(
    stock_calculate_mask: Callable[..., tuple[torch.Tensor, torch.Tensor]],
) -> Callable[..., tuple[torch.Tensor, torch.Tensor]]:
    """Return a mask method with the stock signature whose masks use a different fill value."""

    @functools.wraps(stock_calculate_mask)
    def halved_calculate_mask(
        self: torch.nn.Module,
        H: int,
        W: int,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mask_0, mask_1 = stock_calculate_mask(self, H, W, dtype=dtype)
        return mask_0 / 2, mask_1 / 2

    return halved_calculate_mask


class TestFailSafe:
    """A spandrel that does not look like the one the patch was written for is left untouched."""

    def test_missing_method_leaves_spandrel_untouched_and_warns(
        self,
        monkeypatch: pytest.MonkeyPatch,
        dat_architecture: ModuleType,
        unpatched: None,
        warnings_logged: list[str],
    ) -> None:
        attention_class = dat_architecture.Adaptive_Spatial_Attention
        monkeypatch.delattr(attention_class, "calculate_mask")
        class_attributes_before = dict(vars(attention_class))

        assert not apply_dat_mask_patch()

        assert dict(vars(attention_class)) == class_attributes_before
        assert not hasattr(attention_class, "calculate_mask")
        assert not is_dat_mask_patch_applied()
        assert len(warnings_logged) == 1
        assert "spandrel is left untouched" in warnings_logged[0]

    def test_missing_class_leaves_spandrel_untouched_and_warns(
        self,
        monkeypatch: pytest.MonkeyPatch,
        dat_architecture: ModuleType,
        unpatched: None,
        warnings_logged: list[str],
    ) -> None:
        monkeypatch.delattr(dat_architecture, "Adaptive_Spatial_Attention")

        assert not apply_dat_mask_patch()

        assert not is_dat_mask_patch_applied()
        assert len(warnings_logged) == 1
        assert "spandrel is left untouched" in warnings_logged[0]

    def test_missing_module_leaves_spandrel_untouched_and_warns(
        self,
        monkeypatch: pytest.MonkeyPatch,
        unpatched: None,
        warnings_logged: list[str],
    ) -> None:
        monkeypatch.setattr(dat_mask_patch, "DAT_ARCHITECTURE_MODULE", "spandrel.architectures.DAT.no_such_module")

        assert not apply_dat_mask_patch()

        assert not is_dat_mask_patch_applied()
        assert len(warnings_logged) == 1
        assert "spandrel is left untouched" in warnings_logged[0]

    def test_changed_signature_leaves_spandrel_untouched_and_warns(
        self,
        monkeypatch: pytest.MonkeyPatch,
        dat_architecture: ModuleType,
        unpatched: None,
        warnings_logged: list[str],
    ) -> None:
        attention_class = dat_architecture.Adaptive_Spatial_Attention
        monkeypatch.setattr(attention_class, "calculate_mask", _mask_method_with_device_parameter)

        assert not apply_dat_mask_patch()

        assert attention_class.calculate_mask is _mask_method_with_device_parameter
        assert not is_dat_mask_patch_applied()
        assert len(warnings_logged) == 1
        assert "spandrel is left untouched" in warnings_logged[0]

    def test_changed_mask_values_leave_spandrel_untouched_and_warn(
        self,
        monkeypatch: pytest.MonkeyPatch,
        dat_architecture: ModuleType,
        unpatched: None,
        warnings_logged: list[str],
    ) -> None:
        attention_class = dat_architecture.Adaptive_Spatial_Attention
        changed_method = _mask_method_with_different_values(attention_class.calculate_mask)
        monkeypatch.setattr(attention_class, "calculate_mask", changed_method)

        assert not apply_dat_mask_patch()

        assert attention_class.calculate_mask is changed_method
        assert not is_dat_mask_patch_applied()
        assert len(warnings_logged) == 1
        assert "spandrel is left untouched" in warnings_logged[0]

    def test_forward_that_no_longer_rebuilds_masks_leaves_spandrel_untouched_and_warns(
        self,
        monkeypatch: pytest.MonkeyPatch,
        dat_architecture: ModuleType,
        unpatched: None,
        warnings_logged: list[str],
    ) -> None:
        attention_class = dat_architecture.Adaptive_Spatial_Attention
        method_before = attention_class.calculate_mask
        stock_forward = attention_class.forward

        def forward_with_own_masks(self: torch.nn.Module, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
            # An instance attribute shadows the class method, so the masks no longer come from the method a
            # patch of the class would replace.
            vars(self)["calculate_mask"] = functools.partial(method_before, self)
            return stock_forward(self, x, H, W)

        monkeypatch.setattr(attention_class, "forward", forward_with_own_masks)

        assert not apply_dat_mask_patch()

        assert attention_class.calculate_mask is method_before
        assert not is_dat_mask_patch_applied()
        assert len(warnings_logged) == 1
        assert "spandrel is left untouched" in warnings_logged[0]
