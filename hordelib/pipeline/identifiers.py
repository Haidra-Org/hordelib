"""Public identifiers for pipeline selection.

`ImagePipeline` enumerates every registered image pipeline by its registry name, giving
callers a static vocabulary for explicit pipeline selection. `AUTO_PIPELINE` is the
explicit opt-in sentinel for registry-based automatic selection.

The enum is kept in lockstep with the registered definitions by an import-time audit in
:mod:`hordelib.pipeline.families.image_gen` (mirrored by a named test), so a pipeline
cannot be added or removed without updating this vocabulary.

This module is intentionally import-light (no registry, torch, or ComfyUI imports) so the
identifiers are safe to use from any consumer, including torch-free orchestrators.
"""

from enum import Enum, StrEnum, auto
from typing import Final

__all__ = [
    "AUTO_PIPELINE",
    "AutoPipeline",
    "ImagePipeline",
]


class ImagePipeline(StrEnum):
    """The registered image pipelines, by registry name."""

    QR_CODE = "qr_code"
    CREATIVE_UPSCALE = "creative_upscale"
    STABLE_CASCADE_REMIX = "stable_cascade_remix"
    STABLE_CASCADE_2PASS = "stable_cascade_2pass"
    STABLE_CASCADE = "stable_cascade"
    FLUX = "flux"
    QWEN = "qwen"
    Z_IMAGE = "z_image"
    CONTROLNET_ANNOTATOR = "controlnet_annotator"
    CONTROLNET_HIRES_FIX = "controlnet_hires_fix"
    CONTROLNET = "controlnet"
    STABLE_DIFFUSION_IMG2IMG_MASK = "stable_diffusion_img2img_mask"
    STABLE_DIFFUSION_PAINT = "stable_diffusion_paint"
    STABLE_DIFFUSION_HIRES_FIX = "stable_diffusion_hires_fix"
    STABLE_DIFFUSION = "stable_diffusion"


class AutoPipeline(Enum):
    """Sentinel type for opting into registry-based automatic pipeline selection.

    A single-member enum (rather than an `ImagePipeline` member) so that automatic
    selection is a deliberate, visibly different choice from naming a pipeline, and so
    that `ImagePipeline` remains exactly the set of registered pipelines.
    """

    AUTO = auto()


AUTO_PIPELINE: Final[AutoPipeline] = AutoPipeline.AUTO
"""The explicit opt-in sentinel for registry-based automatic pipeline selection."""
