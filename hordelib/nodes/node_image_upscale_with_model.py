# node_image_upscale_with_model.py
# ComfyUI's ImageUpscaleWithModel with the tile dtype chosen per model.
#
# The stock node calls every upscale model in float32. This node runs an architecture approved for it in
# bfloat16 (see hordelib.execution.upscale_dtype) and is otherwise the stock node step for step: the model
# load, tile 512 and overlap 32, blending on the intermediate device, the out-of-memory retry that halves
# the tile down to 128, progress reporting, and the clamp, layout and dtype of the returned image. The
# pipeline graphs name the stock node and the replacement table in hordelib.pipeline.graph substitutes this
# one, so a change to the stock node's execute needs the same change here.
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, override

import comfy.model_management
import comfy.utils
import torch
from comfy_api.latest import ComfyExtension, io
from loguru import logger

from hordelib.execution.upscale_dtype import (
    UpscaleModelDescriptor,
    prepare_tile_upscaler,
    select_upscale_dtype,
    upscale_bfloat16_disabled,
    upscale_memory_element_size,
)

if TYPE_CHECKING:
    from comfy.model_patcher import ModelPatcher


class _LoadedUpscaleModel(UpscaleModelDescriptor, Protocol):
    """The descriptor as ComfyUI's upscale model loader hands it on."""

    patcher: ModelPatcher
    """The model patcher the loader attaches, through which ComfyUI's model management loads the model."""

    @property
    def scale(self) -> int:
        """The factor the model enlarges each side by."""
        ...


_INITIAL_TILE_SIZE: int = 512
_MINIMUM_TILE_SIZE: int = 128
"""An out-of-memory tile is halved and the image restarted until the tile would be smaller than this."""
_TILE_OVERLAP: int = 32

_TILE_ELEMENTS: int = 512 * 512 * 3
_MEMORY_PER_TILE_BYTE: float = 384.0
"""ComfyUI's estimate of the working memory an upscale model needs per byte of a 512x512 tile and unit scale."""


class HordeImageUpscaleWithModel(io.ComfyNode):
    """Upscale an image with a spandrel model, in the dtype approved for the model's architecture."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        """Define the stock node's sockets: an upscale model and an image in, an image out."""
        return io.Schema(
            node_id="HordeImageUpscaleWithModel",
            display_name="Horde Upscale Image (using Model)",
            category="image/upscaling",
            inputs=[
                io.UpscaleModel.Input("upscale_model"),
                io.Image.Input("image"),
            ],
            outputs=[
                io.Image.Output(),
            ],
        )

    @classmethod
    def execute(cls, upscale_model: _LoadedUpscaleModel, image: torch.Tensor) -> io.NodeOutput:
        """Upscale ``image`` tile by tile and return it in the intermediate dtype, clamped to [0, 1].

        Args:
            upscale_model: The descriptor from the upscale model loader, carrying its model patcher.
            image: A ``(batch, height, width, channels)`` image. A fourth channel is treated as alpha and
                resized separately.

        Side Effects:
            Loads the model on its device and casts it in place to the dtype this run uses.
        """
        device = upscale_model.patcher.load_device

        alpha = None
        if image.shape[-1] == 4:
            alpha = image[..., 3:4]
            image = image[..., :3]

        upscale_dtype = select_upscale_dtype(upscale_model, bfloat16_enabled=not upscale_bfloat16_disabled())

        memory_required = (
            _TILE_ELEMENTS
            * upscale_memory_element_size(image, upscale_dtype)
            * max(upscale_model.scale, 1.0)
            * _MEMORY_PER_TILE_BYTE
        )
        memory_required += image.nelement() * image.element_size()
        comfy.model_management.load_models_gpu(
            [upscale_model.patcher],
            memory_required=memory_required,
            force_full_load=True,
        )

        # The cast follows the load, so the weights are cast on the device the tiles run on, from whichever
        # dtype the load left them in.
        upscale_tile = prepare_tile_upscaler(upscale_model, upscale_dtype)
        logger.debug(
            "image.upscale_with_model: architecture={}, dtype={}",
            upscale_model.architecture.id,
            upscale_dtype,
        )

        in_img = image.movedim(-1, -3).to(device)

        tile = _INITIAL_TILE_SIZE
        output_device = comfy.model_management.intermediate_device()

        while True:
            try:
                steps = in_img.shape[0] * comfy.utils.get_tiled_scale_steps(
                    in_img.shape[3],
                    in_img.shape[2],
                    tile_x=tile,
                    tile_y=tile,
                    overlap=_TILE_OVERLAP,
                )
                pbar = comfy.utils.ProgressBar(steps)
                upscaled = comfy.utils.tiled_scale(
                    in_img,
                    upscale_tile,
                    tile_x=tile,
                    tile_y=tile,
                    overlap=_TILE_OVERLAP,
                    upscale_amount=upscale_model.scale,
                    pbar=pbar,
                    output_device=output_device,
                )
                break
            except Exception as error:
                # Only an out-of-memory error is retried, as in the stock node; anything else is re-raised here.
                comfy.model_management.raise_non_oom(error)
                tile //= 2
                if tile < _MINIMUM_TILE_SIZE:
                    raise

        upscaled = torch.clamp(upscaled.movedim(-3, -1), min=0, max=1.0).to(
            comfy.model_management.intermediate_dtype(),
        )

        if alpha is not None:
            alpha = comfy.utils.common_upscale(
                alpha.movedim(-1, -3).to(upscaled),
                upscaled.shape[2],
                upscaled.shape[1],
                "bilinear",
                "disabled",
            )
            upscaled = torch.cat((upscaled, alpha.movedim(-3, -1)), dim=-1)

        return io.NodeOutput(upscaled)


class HordeImageUpscaleExtension(ComfyExtension):
    """Registers the upscale node through the V3 extension entry point."""

    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        """Return the V3 nodes this extension provides."""
        return [HordeImageUpscaleWithModel]


async def comfy_entrypoint() -> HordeImageUpscaleExtension:
    """Create the extension instance ComfyUI's custom-node loader awaits."""
    return HordeImageUpscaleExtension()
