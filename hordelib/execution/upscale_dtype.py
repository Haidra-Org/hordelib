"""Choice of the dtype an upscale model runs its tiles in.

ComfyUI's upscale node calls every model in float32 and never reads the reduced-precision support the
model's spandrel descriptor declares. hordelib's upscale node
(``hordelib/nodes/node_image_upscale_with_model.py``) runs an approved architecture in bfloat16, which halves
both the time and the peak VRAM of the upscale, and hands on a float32 image as before. Every other
architecture keeps the float32 call.

Public members:

- :data:`BFLOAT16_APPROVED_ARCHITECTURE_IDS` holds the spandrel architecture ids allowed to run in bfloat16.
- :func:`select_upscale_dtype` is the rule: the dtype one loaded model runs in.
- :func:`prepare_tile_upscaler` casts the model to that dtype and returns the function ComfyUI's tiler calls
  for each tile.
- :func:`upscale_memory_element_size` is the bytes per element the node's memory estimate is sized with.
- :func:`upscale_bfloat16_disabled` reads the kill switch :data:`DISABLE_ENV_VAR`.
- :class:`UpscaleModelDescriptor` is the part of a spandrel image model descriptor these functions read.

The module imports neither ComfyUI nor spandrel, so the rule is testable without either being initialised.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Protocol

import torch

__all__ = [
    "BFLOAT16_APPROVED_ARCHITECTURE_IDS",
    "DISABLE_ENV_VAR",
    "TileUpscaler",
    "UpscaleArchitecture",
    "UpscaleModelDescriptor",
    "prepare_tile_upscaler",
    "select_upscale_dtype",
    "upscale_bfloat16_disabled",
    "upscale_memory_element_size",
]

# Each entry was approved from a visual comparison of its float32 and bfloat16 outputs across content types
# (faces, flat-shaded line art, fine texture, dark low-contrast scenes, hard edges and lettering, a
# non-square frame). A descriptor declaring ``supports_bfloat16`` says the arithmetic is stable, which is
# necessary and says nothing about how the picture looks, so an architecture is added here only after the
# same comparison.
BFLOAT16_APPROVED_ARCHITECTURE_IDS: frozenset[str] = frozenset({"DAT"})
"""spandrel architecture ids (``descriptor.architecture.id``) whose upscales run in bfloat16."""

DISABLE_ENV_VAR: str = "HORDELIB_DISABLE_UPSCALE_BFLOAT16"
"""Environment variable that, set to a truthy value, runs every upscale in float32."""

_TRUTHY_VALUES: frozenset[str] = frozenset({"1", "true", "yes", "on"})

TileUpscaler = Callable[[torch.Tensor], torch.Tensor]
"""Takes one ``(1, channels, height, width)`` tile and returns its upscaled float32 tile."""


class UpscaleArchitecture(Protocol):
    """The part of a spandrel ``Architecture`` the dtype rule reads."""

    @property
    def id(self) -> str:
        """spandrel's identifier of the architecture, for example ``DAT`` or ``ESRGAN``."""
        ...


class UpscaleModelDescriptor(Protocol):
    """The part of a spandrel ``ImageModelDescriptor`` the dtype rule and the tile function use."""

    supports_bfloat16: bool
    """Whether spandrel declares the architecture numerically stable in bfloat16."""

    @property
    def architecture(self) -> UpscaleArchitecture:
        """The architecture spandrel detected from the model's state dict."""
        ...

    @property
    def dtype(self) -> torch.dtype:
        """The dtype of the model's weights."""
        ...

    def to(self, dtype: torch.dtype, /) -> object:
        """Mutate the model's weights to ``dtype``."""
        ...

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Return the upscaled image, in the dtype and on the device of ``image``, clamped to [0, 1]."""
        ...


def upscale_bfloat16_disabled() -> bool:
    """Return whether the kill-switch env var turns the bfloat16 upscale path off (default: on).

    The variable is read on every call, so a change applies to the next upscale in the process.
    """
    return os.environ.get(DISABLE_ENV_VAR, "").strip().lower() in _TRUTHY_VALUES


def select_upscale_dtype(descriptor: UpscaleModelDescriptor, *, bfloat16_enabled: bool) -> torch.dtype:
    """Return the dtype the model and its input tiles run in.

    bfloat16 if and only if the bfloat16 path is enabled, the model's architecture is one of
    :data:`BFLOAT16_APPROVED_ARCHITECTURE_IDS` and its descriptor declares ``supports_bfloat16``. The
    architecture is read from the descriptor spandrel built, never from the model's file name.

    Args:
        descriptor: The loaded model's spandrel descriptor.
        bfloat16_enabled: Whether the bfloat16 path is turned on for this run.

    Returns:
        ``torch.bfloat16`` or ``torch.float32``.
    """
    architecture_is_approved = descriptor.architecture.id in BFLOAT16_APPROVED_ARCHITECTURE_IDS
    if bfloat16_enabled and architecture_is_approved and descriptor.supports_bfloat16:
        return torch.bfloat16
    return torch.float32


def upscale_memory_element_size(image: torch.Tensor, dtype: torch.dtype) -> int:
    """Return the bytes per element to size the upscale's memory estimate with.

    ComfyUI's estimate scales with the input image's element size, which stands for the float32 the model
    runs in. A float32 run keeps that term as it is. A run in another dtype uses that dtype's size, so a
    bfloat16 run asks for half, as its measured peak is.

    Args:
        image: The input image as the node received it.
        dtype: The dtype from :func:`select_upscale_dtype`.
    """
    if dtype == torch.float32:
        return image.element_size()
    return dtype.itemsize


def prepare_tile_upscaler(descriptor: UpscaleModelDescriptor, dtype: torch.dtype) -> TileUpscaler:
    """Mutate the model to ``dtype`` and return the function that upscales one tile in it.

    The model is cast only when its current dtype differs, so a model that arrives in either dtype leaves
    in the dtype this run needs, and one already there is not touched. The cast is in place and lossy in one
    direction: a model that ran in bfloat16 and is cast back keeps weights rounded to bfloat16 precision.

    Each tile is cast to ``dtype`` on its way in and to float32 on its way out, so the tiler blends float32
    tiles whatever the model ran in. For float32 this is ComfyUI's ``upscale_model(tile.float())``.

    Args:
        descriptor: The loaded model's spandrel descriptor, on the device the tiles will be on.
        dtype: The dtype from :func:`select_upscale_dtype`.

    Returns:
        The function to pass to ComfyUI's tiler.

    Raises:
        spandrel.UnsupportedDtypeError: The descriptor does not support ``dtype``.
    """
    if descriptor.dtype != dtype:
        descriptor.to(dtype)

    def upscale_tile(tile: torch.Tensor) -> torch.Tensor:
        return descriptor(tile.to(dtype)).float()

    return upscale_tile
