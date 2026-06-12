"""Typed payloads for the post-processing family (upscale, facefix, strip-background).

These are the intent objects for standalone post-processing (the alchemy surface) and for the
embedded ``post_processing`` list inside image-generation payloads. Like
:class:`hordelib.pipeline.payload.ImageGenPayload`, they clamp/coerce rather than reject.

Classification of a post-processor name into its operation kind lives here too, mirroring the
membership checks the legacy embedded loop used (``KNOWN_UPSCALERS`` member *names* and
*values* both occur in the wild — e.g. ``four_4x_AnimeSharp`` has the value ``4x_AnimeSharp``).
"""

from enum import Enum, auto
from typing import Any

from horde_sdk.generation_parameters.alchemy.consts import KNOWN_FACEFIXERS, KNOWN_UPSCALERS
from PIL import Image
from pydantic import BaseModel, ConfigDict, field_validator

STRIP_BACKGROUND_NAME = "strip_background"


class PostProcessorKind(Enum):
    """The operation kind a post-processor name maps to."""

    upscaler = auto()
    facefixer = auto()
    strip_background = auto()


def classify_post_processor(name: str) -> PostProcessorKind | None:
    """Map a horde post-processor name to its operation kind, or None if unrecognized."""
    if name == STRIP_BACKGROUND_NAME:
        return PostProcessorKind.strip_background
    if name in KNOWN_UPSCALERS.__members__ or name in KNOWN_UPSCALERS._value2member_map_:
        return PostProcessorKind.upscaler
    if name in KNOWN_FACEFIXERS.__members__ or name in KNOWN_FACEFIXERS._value2member_map_:
        return PostProcessorKind.facefixer
    return None


class _PostProcessingPayloadBase(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    source_image: Image.Image


class UpscalePayload(_PostProcessingPayloadBase):
    """Upscale an image with an ESRGAN-family model.

    If ``rescale_width``/``rescale_height`` are set, the upscaled output is shrunk back down to
    that size (upscale models produce a fixed multiple of the input size).
    """

    model: str
    rescale_width: int | None = None
    rescale_height: int | None = None


class FacefixPayload(_PostProcessingPayloadBase):
    """Restore faces with GFPGAN or CodeFormers."""

    model: str
    fidelity: float = 0.5
    """CodeFormer fidelity weight (0 = strongest restoration, 1 = closest to input)."""

    @field_validator("fidelity", mode="before")
    @classmethod
    def _clamp_fidelity(cls, value: Any) -> float:
        try:
            value = float(value)
        except (TypeError, ValueError):
            return 0.5
        return min(max(value, 0.0), 1.0)


class StripBackgroundPayload(_PostProcessingPayloadBase):
    """Remove the image background (pure-Python rembg; not a ComfyUI graph)."""


type PostProcessingGraphPayload = UpscalePayload | FacefixPayload
"""The post-processing payloads that execute as ComfyUI graphs."""

type PostProcessingPayload = UpscalePayload | FacefixPayload | StripBackgroundPayload


def post_processing_payload_from_horde_dict(data: dict[str, Any]) -> PostProcessingPayload:
    """Build a typed post-processing payload from a legacy horde-style dict.

    The dict shape matches the historical ``image_upscale``/``image_facefix`` payloads:
    ``model``, ``source_image``, and (for upscales) optional ``width``/``height`` meaning
    "shrink the result back to this size".

    Note: ``facefixer_strength`` is deliberately *not* mapped onto ``FacefixPayload.fidelity``
    — the legacy path silently ignored it (the graph's hardcoded 0.5 always applied), and
    honoring it here would change output images for existing callers. Typed callers can set
    ``fidelity`` explicitly.
    """
    model = data.get("model")
    if not isinstance(model, str):
        raise ValueError(f"Post-processing payload requires a 'model' name, got: {model!r}")

    source_image = data.get("source_image")
    if not isinstance(source_image, Image.Image):
        raise ValueError(f"Post-processing payload requires a PIL 'source_image', got: {type(source_image)}")

    kind = classify_post_processor(model)
    match kind:
        case PostProcessorKind.upscaler:
            return UpscalePayload(
                model=model,
                source_image=source_image,
                rescale_width=data.get("width"),
                rescale_height=data.get("height"),
            )
        case PostProcessorKind.facefixer:
            return FacefixPayload(model=model, source_image=source_image)
        case PostProcessorKind.strip_background:
            return StripBackgroundPayload(source_image=source_image)
        case _:
            raise ValueError(f"Unknown post-processor: {model!r}")
