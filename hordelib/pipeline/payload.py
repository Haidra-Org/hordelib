"""Typed payload models with AI Horde's clamp/coerce-don't-reject semantics.

The Horde contract (encoded in ``tests/test_payload_mapping.py``) is that bad input never
raises: wrong types are coerced, out-of-range values are clamped, unknown enum values fall
back to defaults, and invalid sub-entries (loras, tis, extra images/texts) are dropped.
"""

import random
import sys
from collections.abc import Callable
from typing import Any

import PIL.Image
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from hordelib.pipeline.constants import (
    CONTROLNET_IMAGE_PREPROCESSOR_MAP,
    SAMPLERS_MAP,
    SCHEDULERS,
    SOURCE_IMAGE_PROCESSING_OPTIONS,
)


def _clamp(
    value: Any,
    *,
    datatype: type,
    min: float | None = None,
    max: float | None = None,
    default: Any = None,
    values: list[str] | None = None,
    divisible: int | None = None,
) -> Any:
    """Normalize a single value exactly like the legacy ``HordeLib._validate``."""
    if value is None:
        return default

    if not isinstance(value, datatype):
        try:
            value = datatype(value)
        except (ValueError, TypeError):
            return default

    if divisible and value % divisible != 0:
        value = ((value + (divisible - 1)) // divisible) * divisible

    if min is not None and value < min:
        value = min

    if max is not None and value > max:
        value = max

    if values:
        if isinstance(value, str):
            if value.lower() not in values:
                return default
            value = value.lower()
        elif value not in values:
            return default

    return value


def _clamping_validator(*fields: str, **constraints: Any) -> Any:
    """Build a mode="before" validator that applies :func:`_clamp` to the given fields."""

    def _validate(cls: type, value: Any) -> Any:
        return _clamp(value, **constraints)

    return field_validator(*fields, mode="before")(classmethod(_validate))


class _HordePayloadModel(BaseModel):
    """Base for payload models: ignore unknown keys, never raise for known ones."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="ignore")


class LoraSpec(_HordePayloadModel):
    name: str = ""
    model: float = 1.0
    clip: float = 1.0
    inject_trigger: str | None = None
    is_version: bool | None = None

    _v_name = _clamping_validator("name", datatype=str, default="")
    _v_model = _clamping_validator("model", datatype=float, min=-10.0, max=10.0, default=1.0)
    _v_clip = _clamping_validator("clip", datatype=float, min=-10.0, max=10.0, default=1.0)
    _v_trigger = _clamping_validator("inject_trigger", datatype=str, default=None)
    _v_is_version = _clamping_validator("is_version", datatype=bool, default=None)


class TISpec(_HordePayloadModel):
    name: str = ""
    inject_ti: str | None = None
    strength: float = 1.0

    _v_name = _clamping_validator("name", datatype=str, default="")
    _v_inject_ti = _clamping_validator("inject_ti", datatype=str, default=None)
    _v_strength = _clamping_validator("strength", datatype=float, min=-10.0, max=10.0, default=1.0)


class ExtraSourceImage(_HordePayloadModel):
    image: PIL.Image.Image | None = None
    strength: float = 1.0

    _v_strength = _clamping_validator("strength", datatype=float, min=0.0, max=5.0, default=1.0)


class ExtraText(_HordePayloadModel):
    text: str = ""
    reference: str | None = None

    _v_text = _clamping_validator("text", datatype=str, default="")
    _v_reference = _clamping_validator("reference", datatype=str, default=None)


def _drop_invalid(entries: list[Any], model_class: type[BaseModel], keep: Callable[[Any], bool]) -> list[Any]:
    parsed = []
    for entry in entries or []:
        if isinstance(entry, model_class):
            candidate = entry
        elif isinstance(entry, dict):
            candidate = model_class.model_validate(entry)
        else:
            continue
        if keep(candidate):
            parsed.append(candidate)
    return parsed


class ImageGenPayload(_HordePayloadModel):
    """The full, normalized image-generation payload."""

    sampler_name: str = "k_euler"
    cfg_scale: float = 8.0
    denoising_strength: float = 1.0
    control_strength: float = 1.0
    seed: int = Field(default_factory=lambda: random.randint(0, sys.maxsize))
    width: int = 512
    height: int = 512
    hires_fix: bool = False
    clip_skip: int = 1
    control_type: str | None = None
    image_is_control: bool = False
    return_control_map: bool = False
    prompt: str = ""
    negative_prompt: str = ""
    loras: list[LoraSpec] = []
    tis: list[TISpec] = []
    ddim_steps: int = 30
    n_iter: int = 1
    model: str = "stable_diffusion"
    source_mask: PIL.Image.Image | None = None
    source_image: PIL.Image.Image | None = None
    source_processing: str | None = None
    hires_fix_denoising_strength: float = 0.65
    scheduler: str = "normal"
    tiling: bool = False
    model_name: str = "stable_diffusion"
    stable_cascade_stage_b: str | None = None
    stable_cascade_stage_c: str | None = None
    extra_source_images: list[ExtraSourceImage] = []
    extra_texts: list[ExtraText] = []
    workflow: str = "auto_detect"
    transparent: bool = False

    _v_sampler = _clamping_validator(
        "sampler_name",
        datatype=str,
        values=list(SAMPLERS_MAP.keys()),
        default="k_euler",
    )
    _v_cfg = _clamping_validator("cfg_scale", datatype=float, min=1, max=100, default=8.0)
    _v_denoise = _clamping_validator("denoising_strength", datatype=float, min=0.01, max=1.0, default=1.0)
    _v_cstrength = _clamping_validator("control_strength", datatype=float, min=0.01, max=3.0, default=1.0)
    _v_width = _clamping_validator("width", datatype=int, min=64, max=8192, default=512, divisible=64)
    _v_height = _clamping_validator("height", datatype=int, min=64, max=8192, default=512, divisible=64)
    _v_hires = _clamping_validator("hires_fix", datatype=bool, default=False)
    _v_clip_skip = _clamping_validator("clip_skip", datatype=int, min=1, max=20, default=1)
    _v_control_type = _clamping_validator(
        "control_type",
        datatype=str,
        values=list(CONTROLNET_IMAGE_PREPROCESSOR_MAP.keys()),
        default=None,
    )
    _v_image_is_control = _clamping_validator("image_is_control", datatype=bool, default=False)
    _v_return_map = _clamping_validator("return_control_map", datatype=bool, default=False)
    _v_prompt = _clamping_validator("prompt", datatype=str, default="")
    _v_negative = _clamping_validator("negative_prompt", datatype=str, default="")
    _v_steps = _clamping_validator("ddim_steps", datatype=int, min=1, max=500, default=30)
    _v_n_iter = _clamping_validator("n_iter", datatype=int, min=1, max=100, default=1)
    _v_model = _clamping_validator("model", datatype=str, default="stable_diffusion")
    _v_source_processing = _clamping_validator(
        "source_processing",
        datatype=str,
        values=SOURCE_IMAGE_PROCESSING_OPTIONS,
        default=None,
    )
    _v_hires_denoise = _clamping_validator(
        "hires_fix_denoising_strength",
        datatype=float,
        min=0.01,
        max=1.0,
        default=0.65,
    )
    _v_scheduler = _clamping_validator("scheduler", datatype=str, values=SCHEDULERS, default="normal")
    _v_tiling = _clamping_validator("tiling", datatype=bool, default=False)
    _v_model_name = _clamping_validator("model_name", datatype=str, default="stable_diffusion")
    _v_stage_b = _clamping_validator("stable_cascade_stage_b", datatype=str, default=None)
    _v_stage_c = _clamping_validator("stable_cascade_stage_c", datatype=str, default=None)
    _v_workflow = _clamping_validator("workflow", datatype=str, default="auto_detect")
    _v_transparent = _clamping_validator("transparent", datatype=bool, default=False)

    @field_validator("seed", mode="before")
    @classmethod
    def _v_seed(cls, value: Any) -> int:
        seed = _clamp(value, datatype=int, default=None)
        if seed is None:
            return random.randint(0, sys.maxsize)
        return seed

    @model_validator(mode="after")
    def _drop_invalid_subentries(self) -> "ImageGenPayload":
        self.loras = _drop_invalid(self.loras, LoraSpec, lambda lora: bool(lora.name))
        self.tis = _drop_invalid(self.tis, TISpec, lambda ti: bool(ti.name))
        self.extra_source_images = _drop_invalid(
            self.extra_source_images,
            ExtraSourceImage,
            lambda img: img.image is not None,
        )
        self.extra_texts = _drop_invalid(self.extra_texts, ExtraText, lambda text: bool(text.text))
        return self

    @field_validator("loras", "tis", "extra_source_images", "extra_texts", mode="before")
    @classmethod
    def _coerce_lists(cls, value: Any) -> list:
        if value is None or not isinstance(value, list):
            return []
        return [entry for entry in value if isinstance(entry, (dict, BaseModel))]

    @field_validator("source_image", "source_mask", mode="before")
    @classmethod
    def _images_or_none(cls, value: Any) -> PIL.Image.Image | None:
        return value if isinstance(value, PIL.Image.Image) else None

    @classmethod
    def from_horde_dict(cls, data: dict[str, Any]) -> "ImageGenPayload":
        """Build a normalized payload from a raw Horde payload dict (keys lowercased)."""
        return cls.model_validate({str(k).lower(): v for k, v in data.items()})
