"""Run-time patch for spandrel's DAT shifted-window attention masks.

A DAT shifted-window block holds registered masks for the one size the model was built for. For any other
input size it builds two ``(windows, tokens, tokens)`` masks from CPU tensors on every forward pass and copies
them to the model's device. A tiled upscale feeds it tiles of other sizes, so that construction and copy run in
every shifted block of every tile and dominate the upscale's wall time.

The patch replaces the block's mask method with one that builds the same masks on the block's own device. The
output is bit-identical. The masks are local to the forward pass that asked for them, so the patch holds no
memory after an upscale and has nothing to release when the model is unloaded. The cost is that both masks of a
block are on the device at once, where stock spandrel copies them over one at a time.

Public members:

- :func:`apply_dat_mask_patch` installs the patch, or leaves spandrel untouched and logs one warning when the
  installed spandrel does not behave like the one the patch was written for.
- :func:`remove_dat_mask_patch` restores spandrel's method.
- :func:`is_dat_mask_patch_applied` reports whether the patch is installed.
- :data:`DAT_ARCHITECTURE_MODULE` is the spandrel module the patch targets.

The module imports spandrel only inside :func:`apply_dat_mask_patch`. Only the DAT attention class is touched,
and a call for the model's native size (the registered-buffer path) is passed to spandrel's method unchanged.
"""

from __future__ import annotations

import dataclasses
import importlib
import inspect
from collections.abc import Callable, Sequence
from typing import Protocol

import torch
from loguru import logger

__all__ = [
    "DAT_ARCHITECTURE_MODULE",
    "apply_dat_mask_patch",
    "is_dat_mask_patch_applied",
    "remove_dat_mask_patch",
]

DAT_ARCHITECTURE_MODULE: str = "spandrel.architectures.DAT.__arch.DAT"
"""Import path of the spandrel module that defines the DAT attention blocks."""

_ATTENTION_CLASS_NAME: str = "Adaptive_Spatial_Attention"
_MASK_METHOD_NAME: str = "calculate_mask"
_MASK_METHOD_PARAMETERS: tuple[str, ...] = ("self", "H", "W", "dtype")

_MASKED_ATTENTION_BIAS: float = -100.0
"""The bias spandrel adds to attention between tokens of different shift regions."""

MaskPair = tuple[torch.Tensor, torch.Tensor]
MaskMethod = Callable[..., MaskPair]


class _UnexpectedSpandrelError(Exception):
    """Raised when the installed spandrel differs from the one the patch was written for."""


class _ShiftedWindowBlock(Protocol):
    """The attributes of spandrel's attention block that the replacement method reads."""

    patches_resolution: int
    split_size: Sequence[int]
    shift_size: Sequence[int]
    qkv: torch.nn.Linear


@dataclasses.dataclass(frozen=True)
class _InstalledPatch:
    """Represents the patch as installed: the class it changed and both versions of the method."""

    attention_class: type[torch.nn.Module]
    stock_method: MaskMethod
    replacement_method: MaskMethod


@dataclasses.dataclass(frozen=True)
class _RecordedMaskCall:
    """Represents one call spandrel made to its mask method while being probed."""

    height: int
    width: int
    dtype: torch.dtype | None
    masks: MaskPair


class _ProbeBlock:
    """The small shifted-window block the patch builds to observe spandrel's behaviour before changing it."""

    # The narrowest width at which none of the block's layers is built with zero elements.
    DIM: int = 32
    NUM_HEADS: int = 2
    NATIVE_SIZE: int = 8
    SPLIT_SIZE: tuple[int, int] = (2, 4)
    SHIFT_SIZE: tuple[int, int] = (1, 2)
    # Block 2 of residual group 0 is the first block spandrel makes a shifted-window block.
    RESIDUAL_GROUP_INDEX: int = 0
    BLOCK_INDEX: int = 2
    FORWARD_HEIGHT: int = 16
    FORWARD_WIDTH: int = 8
    DIRECT_HEIGHT: int = 12
    DIRECT_WIDTH: int = 16


_installed_patch: _InstalledPatch | None = None


def _shifted_window_mask(
    *,
    height: int,
    width: int,
    window_height: int,
    window_width: int,
    shift_height: int,
    shift_width: int,
    dtype: torch.dtype | None,
    device: torch.device,
) -> torch.Tensor:
    """Create one shifted-window attention mask on ``device``.

    The image is cut into nine regions by the window and shift sizes, as spandrel does, and a pair of tokens in
    one window is masked when the two lie in different regions. spandrel subtracts the region indices as floats
    and rewrites the result twice. Comparing the indices gives the same mask with one boolean temporary and no
    intermediate the size of the mask.

    Args:
        height: Padded tile height in tokens, a multiple of ``window_height``.
        width: Padded tile width in tokens, a multiple of ``window_width``.
        window_height: Attention window height.
        window_width: Attention window width.
        shift_height: Vertical shift of the window grid.
        shift_width: Horizontal shift of the window grid.
        dtype: Floating dtype of the mask, or None for torch's default.
        device: Device the mask is created on.

    Returns:
        A ``(windows, tokens, tokens)`` tensor holding 0 for a visible pair and the masking bias otherwise.
    """
    height_slices = (
        slice(0, -window_height),
        slice(-window_height, -shift_height),
        slice(-shift_height, None),
    )
    width_slices = (
        slice(0, -window_width),
        slice(-window_width, -shift_width),
        slice(-shift_width, None),
    )
    region_of_token = torch.zeros((height, width), dtype=torch.uint8, device=device)
    region_index = 0
    for height_slice in height_slices:
        for width_slice in width_slices:
            region_of_token[height_slice, width_slice] = region_index
            region_index += 1

    region_of_window_token = (
        region_of_token.view(height // window_height, window_height, width // window_width, window_width)
        .permute(0, 2, 1, 3)
        .reshape(-1, window_height * window_width)
    )
    crosses_region = region_of_window_token.unsqueeze(1) != region_of_window_token.unsqueeze(2)
    mask = torch.zeros(crosses_region.shape, dtype=dtype, device=device)
    return mask.masked_fill_(crosses_region, _MASKED_ATTENTION_BIAS)


def _make_replacement_method(stock_method: MaskMethod) -> MaskMethod:
    """Create the mask method installed in place of ``stock_method``."""

    def calculate_mask_on_block_device(
        self: _ShiftedWindowBlock,
        H: int,
        W: int,
        dtype: torch.dtype | None = None,
    ) -> MaskPair:
        # The names and order of the parameters are spandrel's; its forward pass passes dtype by keyword.
        # A native-size call is the constructor filling the registered buffers, which stay stock-built.
        if H == self.patches_resolution and W == self.patches_resolution:
            return stock_method(self, H, W, dtype=dtype)

        device = self.qkv.weight.device
        window_height, window_width = self.split_size
        shift_height, shift_width = self.shift_size
        # The block's second attention branch uses the transposed window.
        return (
            _shifted_window_mask(
                height=H,
                width=W,
                window_height=window_height,
                window_width=window_width,
                shift_height=shift_height,
                shift_width=shift_width,
                dtype=dtype,
                device=device,
            ),
            _shifted_window_mask(
                height=H,
                width=W,
                window_height=window_width,
                window_width=window_height,
                shift_height=shift_width,
                shift_width=shift_height,
                dtype=dtype,
                device=device,
            ),
        )

    return calculate_mask_on_block_device


def _validate_mask_method_signature(stock_method: MaskMethod) -> None:
    """Validate that spandrel's mask method takes the parameters the replacement takes.

    Raises:
        _UnexpectedSpandrelError: The parameters differ, as they would if the method had gained a device.
    """
    parameters = inspect.signature(stock_method).parameters
    if tuple(parameters) != _MASK_METHOD_PARAMETERS:
        raise _UnexpectedSpandrelError(f"{_MASK_METHOD_NAME} takes {tuple(parameters)}")
    if parameters["dtype"].default is not None:
        raise _UnexpectedSpandrelError(f"{_MASK_METHOD_NAME} no longer defaults dtype to None")


def _masks_match(candidate: MaskPair, reference: MaskPair) -> bool:
    """Return whether two mask pairs are equal in dtype, shape and every value."""
    if len(candidate) != len(reference):
        return False
    return all(
        candidate_mask.dtype == reference_mask.dtype and torch.equal(candidate_mask, reference_mask)
        for candidate_mask, reference_mask in zip(candidate, reference, strict=True)
    )


def _validate_spandrel_behaviour(
    *,
    attention_class: type[torch.nn.Module],
    stock_method: MaskMethod,
    replacement_method: MaskMethod,
) -> None:
    """Validate, on a small real block, that spandrel builds masks where and how the replacement assumes.

    A version number says nothing about a renamed attribute, a changed constructor or a forward pass that no
    longer rebuilds its masks, so the block is built and run. It must register its native-size masks through
    the mask method, call that method once for a forward pass at another size, and return masks the replacement
    reproduces exactly.

    Raises:
        _UnexpectedSpandrelError: The block does not behave as the replacement assumes.
    """
    recorded_calls: list[_RecordedMaskCall] = []

    class _RecordingAttention(attention_class):  # type: ignore[valid-type, misc]
        def calculate_mask(self, H: int, W: int, dtype: torch.dtype | None = None) -> MaskPair:
            masks = stock_method(self, H, W, dtype=dtype)
            recorded_calls.append(_RecordedMaskCall(height=H, width=W, dtype=dtype, masks=masks))
            return masks

    block = _RecordingAttention(
        dim=_ProbeBlock.DIM,
        num_heads=_ProbeBlock.NUM_HEADS,
        reso=_ProbeBlock.NATIVE_SIZE,
        split_size=list(_ProbeBlock.SPLIT_SIZE),
        shift_size=list(_ProbeBlock.SHIFT_SIZE),
        rg_idx=_ProbeBlock.RESIDUAL_GROUP_INDEX,
        b_idx=_ProbeBlock.BLOCK_INDEX,
    ).eval()

    native_calls = [(call.height, call.width, call.dtype) for call in recorded_calls]
    if native_calls != [(_ProbeBlock.NATIVE_SIZE, _ProbeBlock.NATIVE_SIZE, None)]:
        raise _UnexpectedSpandrelError(f"the constructor's mask calls were {native_calls}")
    if block.attn_mask_0 is None or block.attn_mask_1 is None:
        raise _UnexpectedSpandrelError("the probe block registered no masks")

    recorded_calls.clear()
    tokens = torch.zeros((1, _ProbeBlock.FORWARD_HEIGHT * _ProbeBlock.FORWARD_WIDTH, _ProbeBlock.DIM))
    with torch.no_grad():
        block(tokens, _ProbeBlock.FORWARD_HEIGHT, _ProbeBlock.FORWARD_WIDTH)

    forward_calls = [(call.height, call.width, call.dtype) for call in recorded_calls]
    if forward_calls != [(_ProbeBlock.FORWARD_HEIGHT, _ProbeBlock.FORWARD_WIDTH, tokens.dtype)]:
        raise _UnexpectedSpandrelError(f"a forward pass at a non-native size made the mask calls {forward_calls}")

    forward_masks = recorded_calls[0].masks
    replacement_forward_masks = replacement_method(
        block,
        _ProbeBlock.FORWARD_HEIGHT,
        _ProbeBlock.FORWARD_WIDTH,
        dtype=tokens.dtype,
    )
    if not _masks_match(replacement_forward_masks, forward_masks):
        raise _UnexpectedSpandrelError("the masks spandrel builds are not the ones the patch builds")

    direct_masks = stock_method(block, _ProbeBlock.DIRECT_HEIGHT, _ProbeBlock.DIRECT_WIDTH, dtype=None)
    replacement_direct_masks = replacement_method(
        block,
        _ProbeBlock.DIRECT_HEIGHT,
        _ProbeBlock.DIRECT_WIDTH,
        dtype=None,
    )
    if not _masks_match(replacement_direct_masks, direct_masks):
        raise _UnexpectedSpandrelError("the masks spandrel builds are not the ones the patch builds")


def apply_dat_mask_patch() -> bool:
    """Install the patch so DAT builds its attention masks on the model's device.

    The patch is installed only when the installed spandrel has the attention class and mask method the patch
    targets, with the expected signature, and a small real block behaves as the replacement assumes. Otherwise
    spandrel is left untouched and one warning is logged. Calling this again while the patch is installed
    changes nothing.

    Returns:
        Whether the patch is installed when the call returns.

    Side Effects:
        Imports spandrel and replaces a method on its DAT attention class.
    """
    global _installed_patch

    if is_dat_mask_patch_applied():
        return True

    try:
        architecture_module = importlib.import_module(DAT_ARCHITECTURE_MODULE)
        attention_class = vars(architecture_module).get(_ATTENTION_CLASS_NAME)
        if not isinstance(attention_class, type) or not issubclass(attention_class, torch.nn.Module):
            raise _UnexpectedSpandrelError(f"{_ATTENTION_CLASS_NAME} is not a module class")
        stock_method = vars(attention_class).get(_MASK_METHOD_NAME)
        if not inspect.isfunction(stock_method):
            raise _UnexpectedSpandrelError(f"{_ATTENTION_CLASS_NAME}.{_MASK_METHOD_NAME} is not a method")
        _validate_mask_method_signature(stock_method)
        replacement_method = _make_replacement_method(stock_method)
        _validate_spandrel_behaviour(
            attention_class=attention_class,
            stock_method=stock_method,
            replacement_method=replacement_method,
        )
    except Exception as error:
        # Any failure here means the installed spandrel is not the code the replacement was derived from, and
        # an upscale on stock spandrel is slow but correct.
        logger.warning(
            "DAT attention masks stay on spandrel's CPU path; spandrel is left untouched: {}: {}",
            type(error).__name__,
            error,
        )
        return False

    setattr(attention_class, _MASK_METHOD_NAME, replacement_method)
    _installed_patch = _InstalledPatch(
        attention_class=attention_class,
        stock_method=stock_method,
        replacement_method=replacement_method,
    )
    logger.debug("Applied spandrel patch: target={}.{}", _ATTENTION_CLASS_NAME, _MASK_METHOD_NAME)
    return True


def remove_dat_mask_patch() -> bool:
    """Restore spandrel's mask method.

    Returns:
        Whether a patch was installed and has been removed.
    """
    global _installed_patch

    installed_patch = _installed_patch
    if installed_patch is None:
        return False
    if is_dat_mask_patch_applied():
        setattr(installed_patch.attention_class, _MASK_METHOD_NAME, installed_patch.stock_method)
    _installed_patch = None
    return True


def is_dat_mask_patch_applied() -> bool:
    """Return whether spandrel's DAT attention class currently carries the replacement method."""
    installed_patch = _installed_patch
    if installed_patch is None:
        return False
    return vars(installed_patch.attention_class).get(_MASK_METHOD_NAME) is installed_patch.replacement_method
