"""Adapter from horde_sdk's backend-agnostic generation parameters to the pipeline payload.

This is the typed front door's translation layer: it maps
[`ImageGenerationParameters`][horde_sdk.generation_parameters.image.ImageGenerationParameters]
(the structured, component-based vocabulary shared across the Haidra ecosystem) onto
[`ImageGenPayload`][hordelib.pipeline.payload.ImageGenPayload] (the flat, clamped payload the
pipeline machinery consumes). It is pure and import-light: no torch, no ComfyUI, no registry.

Image bytes are decoded to PIL here; an undecodable image degrades to None with a recorded
fault rather than aborting, matching the pipeline family's coerce-don't-reject contract.
"""

import io
from typing import Any

import PIL.Image
from horde_sdk.ai_horde_api.apimodels.base import GenMetadataEntry
from horde_sdk.ai_horde_api.consts import METADATA_TYPE, METADATA_VALUE
from horde_sdk.generation_parameters.image import (
    ImageGenerationParameters,
    LoRaEntry,
    TIEntry,
)
from horde_sdk.generation_parameters.image.consts import (
    KNOWN_IMAGE_SOURCE_PROCESSING,
    LORA_TRIGGER_INJECT_CHOICE,
    TI_TRIGGER_INJECT_CHOICE,
)
from loguru import logger

from hordelib.pipeline.payload import ImageGenPayload

__all__ = [
    "to_image_gen_payload",
]

_QR_CODE_WORKFLOW_NAME = "qr_code"


def _decode_image(
    image_data: bytes | str | None,
    metadata_type: METADATA_TYPE,
    faults: list[GenMetadataEntry],
    *,
    ref: str | None = None,
) -> PIL.Image.Image | None:
    """Decode raw image bytes to a PIL image, recording a fault when decoding fails."""
    if image_data is None:
        return None

    if isinstance(image_data, str):
        # The generic parameters carry decoded bytes; a string here is a caller error, but the
        # coerce-don't-reject contract still applies.
        logger.warning("Expected decoded image bytes but received a string; ignoring the image.")
        faults.append(GenMetadataEntry(type=metadata_type, value=METADATA_VALUE.parse_failed, ref=ref))
        return None

    try:
        return PIL.Image.open(io.BytesIO(image_data))
    except Exception as err:
        faults.append(GenMetadataEntry(type=metadata_type, value=METADATA_VALUE.parse_failed, ref=ref))
        logger.warning(f"Failed to parse {metadata_type} image data: {err}")
        return None


def _lora_entry_to_spec(lora_entry: LoRaEntry) -> dict[str, Any]:
    """Convert a generic LoRa entry to the payload's lora spec fields."""
    targets_specific_version = lora_entry.remote_version_id is not None

    inject_trigger: str | None = None
    if lora_entry.lora_inject_trigger_choice != LORA_TRIGGER_INJECT_CHOICE.NO_INJECT and lora_entry.lora_triggers:
        inject_trigger = lora_entry.lora_triggers[0]

    return {
        "name": lora_entry.remote_version_id if targets_specific_version else (lora_entry.name or ""),
        "model": lora_entry.model_strength,
        "clip": lora_entry.clip_strength,
        "inject_trigger": inject_trigger,
        "is_version": targets_specific_version,
    }


def _ti_entry_to_spec(ti_entry: TIEntry) -> dict[str, Any]:
    """Convert a generic Textual Inversion entry to the payload's ti spec fields."""
    inject_ti: str | None = None
    if ti_entry.ti_inject_trigger_choice == TI_TRIGGER_INJECT_CHOICE.POSITIVE_PROMPT:
        inject_ti = "prompt"
    elif ti_entry.ti_inject_trigger_choice == TI_TRIGGER_INJECT_CHOICE.NEGATIVE_PROMPT:
        inject_ti = "negprompt"

    return {
        "name": ti_entry.name or "",
        "inject_ti": inject_ti,
        "strength": ti_entry.model_strength,
    }


def to_image_gen_payload(
    params: ImageGenerationParameters,
) -> tuple[ImageGenPayload, list[GenMetadataEntry]]:
    """Convert generic image generation parameters to the pipeline payload.

    The hires-fix component's explicit two-pass values (first-pass resolution, second-pass
    steps and denoise) flow through as payload overrides, so a caller that computed the
    passes itself is authoritative; the pipeline's own recompute only applies when the
    overrides are absent.

    Args:
        params: The backend-agnostic generation parameters.

    Returns:
        The clamped pipeline payload and any faults recorded while decoding images.
    """
    faults: list[GenMetadataEntry] = []
    base_params = params.base_params
    components = params.additional_params

    payload_fields: dict[str, Any] = {
        "model": base_params.model,
        "model_name": base_params.model,
        "prompt": base_params.prompt,
        "negative_prompt": base_params.negative_prompt,
        "seed": base_params.seed,
        "width": base_params.width,
        "height": base_params.height,
        "ddim_steps": base_params.steps,
        "cfg_scale": base_params.cfg_scale,
        "sampler_name": base_params.sampler_name,
        "scheduler": base_params.scheduler,
        "clip_skip": base_params.clip_skip,
        "denoising_strength": base_params.denoising_strength,
        "tiling": base_params.tiling,
        "transparent": base_params.transparent,
        "n_iter": params.batch_size,
        "source_processing": params.source_processing,
    }

    source_image: PIL.Image.Image | None = None
    source_mask: PIL.Image.Image | None = None

    img2img_params = components.image2image_params
    if img2img_params is not None:
        source_image = _decode_image(img2img_params.source_image, METADATA_TYPE.source_image, faults)
        source_mask = _decode_image(img2img_params.source_mask, METADATA_TYPE.source_mask, faults)

    remix_params = components.remix_params
    if remix_params is not None:
        source_image = _decode_image(remix_params.source_image, METADATA_TYPE.source_image, faults)
        extra_source_images = []
        for remix_image_index, remix_image in enumerate(remix_params.remix_images):
            remix_pil_image = _decode_image(
                remix_image.image,
                METADATA_TYPE.extra_source_images,
                faults,
                ref=str(remix_image_index),
            )
            if remix_pil_image is None:
                continue
            extra_source_images.append({"image": remix_pil_image, "strength": remix_image.strength})
        payload_fields["extra_source_images"] = extra_source_images

    controlnet_params = components.controlnet_params
    if controlnet_params is not None:
        control_uses_premade_map = controlnet_params.control_map is not None
        payload_fields["control_type"] = str(controlnet_params.controlnet_type)
        payload_fields["image_is_control"] = control_uses_premade_map
        payload_fields["return_control_map"] = controlnet_params.return_control_map
        control_image = controlnet_params.control_map if control_uses_premade_map else controlnet_params.source_image
        if source_image is None:
            source_image = _decode_image(control_image, METADATA_TYPE.source_image, faults)

    payload_fields["source_image"] = source_image
    payload_fields["source_mask"] = source_mask

    hires_fix_params = components.hires_fix_params
    if hires_fix_params is not None:
        first_pass = hires_fix_params.first_pass
        second_pass = hires_fix_params.second_pass
        payload_fields["hires_fix"] = True
        payload_fields["width"] = second_pass.width
        payload_fields["height"] = second_pass.height
        payload_fields["ddim_steps"] = first_pass.steps
        payload_fields["denoising_strength"] = first_pass.denoising_strength
        payload_fields["hires_fix_first_pass_width"] = first_pass.width
        payload_fields["hires_fix_first_pass_height"] = first_pass.height
        payload_fields["hires_fix_second_pass_steps"] = second_pass.steps
        payload_fields["hires_fix_denoising_strength"] = second_pass.denoising_strength

    lora_entries = components.lora_entries
    if lora_entries:
        payload_fields["loras"] = [_lora_entry_to_spec(lora_entry) for lora_entry in lora_entries]

    ti_entries = components.ti_entries
    if ti_entries:
        payload_fields["tis"] = [_ti_entry_to_spec(ti_entry) for ti_entry in ti_entries]

    workflow_entries = components.custom_workflow_entries
    if workflow_entries:
        workflow_params = workflow_entries[0]
        payload_fields["workflow"] = str(workflow_params.custom_workflow_name)
        if workflow_params.extra_texts:
            payload_fields["extra_texts"] = [
                {"text": extra_text.text, "reference": extra_text.reference}
                for extra_text in workflow_params.extra_texts
            ]

        # The qr_code workflow's composite sampler reuses the hires-fix denoise input; for
        # image-fed generations without an explicit value it inherits the first-pass strength.
        is_txt2img = (
            params.source_processing is None or params.source_processing == KNOWN_IMAGE_SOURCE_PROCESSING.txt2img
        )
        if (
            str(workflow_params.custom_workflow_name) == _QR_CODE_WORKFLOW_NAME
            and not is_txt2img
            and hires_fix_params is None
            and base_params.denoising_strength is not None
        ):
            payload_fields["hires_fix_denoising_strength"] = base_params.denoising_strength

    # Drop unset optional fields so the payload model's own defaults apply.
    payload_fields = {key: value for key, value in payload_fields.items() if value is not None}

    return ImageGenPayload.model_validate(payload_fields), faults
