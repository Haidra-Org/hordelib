"""The image family's selection vocabulary: named payload features and the typed selector.

Every condition an image pipeline selects on is either a named :data:`PayloadFeature` below
or a typed field on :class:`ImageSelector` (workflow, baselines). This keeps the family's
whole selection logic auditable on one screen; ``extra_predicate`` should stay unused.
"""

from dataclasses import dataclass
from typing import override

from horde_model_reference.meta_consts import KNOWN_IMAGE_GENERATION_BASELINE

from hordelib.pipeline.context import ModelContext
from hordelib.pipeline.definition import PayloadFeature, Selector
from hordelib.pipeline.payload import ImageGenPayload

__all__ = [
    "CONTROLNET",
    "HIRES_FIX",
    "IMG2IMG_MASK",
    "PAINTING",
    "REMIX",
    "RETURN_CONTROL_MAP",
    "ImageSelector",
]


def _hires_fix_requested(payload: ImageGenPayload) -> bool:
    return payload.hires_fix


def _controlnet_requested(payload: ImageGenPayload) -> bool:
    return bool(payload.control_type)


def _control_map_return_requested(payload: ImageGenPayload) -> bool:
    return payload.return_control_map


def _painting_requested(payload: ImageGenPayload) -> bool:
    return payload.source_processing in ("inpainting", "outpainting")


def _remix_requested(payload: ImageGenPayload) -> bool:
    return payload.source_processing == "remix"


def _img2img_has_mask(payload: ImageGenPayload) -> bool:
    if payload.source_processing != "img2img":
        return False
    if payload.source_mask is not None:
        return True
    return payload.source_image is not None and len(payload.source_image.getbands()) == 4


HIRES_FIX = PayloadFeature[ImageGenPayload](name="hires_fix", is_set=_hires_fix_requested)
CONTROLNET = PayloadFeature[ImageGenPayload](name="controlnet", is_set=_controlnet_requested)
RETURN_CONTROL_MAP = PayloadFeature[ImageGenPayload](name="return_control_map", is_set=_control_map_return_requested)
PAINTING = PayloadFeature[ImageGenPayload](name="painting", is_set=_painting_requested)
REMIX = PayloadFeature[ImageGenPayload](name="remix", is_set=_remix_requested)
IMG2IMG_MASK = PayloadFeature[ImageGenPayload](name="img2img_mask", is_set=_img2img_has_mask)
"""img2img with an explicit mask, or an alpha channel acting as one."""


@dataclass(frozen=True)
class ImageSelector(Selector[ImageGenPayload, ModelContext]):
    """The image family's selector: adds the workflow and baseline selection axes."""

    workflow: str | None = None
    """Match only when the payload requests this workflow (e.g. ``"qr_code"``)."""
    baselines: frozenset[KNOWN_IMAGE_GENERATION_BASELINE] | None = None
    """Match only when the resolved model's baseline is in this set."""

    @override
    def matches(self, payload: ImageGenPayload, context: ModelContext) -> bool:
        if self.workflow is not None and payload.workflow != self.workflow:
            return False
        if self.baselines is not None and context.baseline not in self.baselines:
            return False
        return super().matches(payload, context)

    @override
    def has_criteria(self) -> bool:
        return self.workflow is not None or self.baselines is not None or super().has_criteria()
