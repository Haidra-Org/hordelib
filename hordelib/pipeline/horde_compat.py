"""AI Horde payload normalization: the counterintuitive-but-contractual payload hacks.

This is the pure half of the legacy ``_apply_aihorde_compatibility_hacks``: everything that
massages a raw Horde payload dict before validation, given facts already resolved about the
model (see :func:`hordelib.pipeline.resolution.resolve_image_model`). All implicit witchcraft
the AI Horde worker relies on is encapsulated here, with the same semantics it always had.
"""

from copy import deepcopy
from typing import Any

from horde_model_reference.meta_consts import KNOWN_IMAGE_GENERATION_BASELINE
from horde_sdk.ai_horde_api.apimodels.base import GenMetadataEntry
from horde_sdk.ai_horde_api.consts import METADATA_TYPE, METADATA_VALUE
from loguru import logger
from PIL import Image

from hordelib.execution.sigma_schedules import SigmaScheduleRequest
from hordelib.pipeline.constants import (
    SCHEDULERS,
    SIGMA_GENERATOR_SCHEDULES,
    SigmaGeneratorSchedule,
    resolve_schedule,
)
from hordelib.pipeline.context import ModelContext
from hordelib.pipeline.families.image_gen.baselines import align_your_steps_model_type
from hordelib.pipeline.families.image_gen.features import IMG2IMG_MASK
from hordelib.pipeline.payload import ImageGenPayload
from hordelib.utils.image_utils import ImageUtils


def normalize_horde_payload(
    payload: dict[str, Any],
    context: ModelContext,
) -> tuple[dict[str, Any], list[GenMetadataEntry]]:
    """Normalize a raw Horde payload dict (pure; model facts come from ``context``).

    Faithful port of the legacy compatibility hacks: inpainting fallbacks, the karras flag,
    ``###`` prompt splitting, and the hires-fix disable rules.
    """
    payload = deepcopy(payload)
    faults: list[GenMetadataEntry] = []

    # The horde model name; "model" itself historically became the checkpoint file path,
    # which now lives on the context instead.
    payload["model_name"] = context.horde_model_name

    if context.is_inpainting_model:
        if payload.get("source_processing") not in ["inpainting", "outpainting"]:
            logger.warning(
                "Inpainting model detected, but source processing not set to inpainting or outpainting.",
            )
            payload["source_processing"] = "inpainting"

        source_image = payload.get("source_image")
        source_mask = payload.get("source_mask")

        if source_image is None or not isinstance(source_image, Image.Image):
            logger.warning(
                "Inpainting model detected, but source image is not a valid image. Using a noise image.",
            )
            faults.append(
                GenMetadataEntry(type=METADATA_TYPE.source_image, value=METADATA_VALUE.parse_failed),
            )
            payload["source_image"] = ImageUtils.create_noise_image(
                payload.get("width"),
                payload.get("height"),
            )

        source_image = payload.get("source_image")

        if source_mask is None and (
            source_image is None
            or (isinstance(source_image, Image.Image) and not ImageUtils.has_alpha_channel(source_image))
        ):
            logger.warning(
                "Inpainting model detected, but no source mask provided. Using an all white mask.",
            )
            faults.append(
                GenMetadataEntry(type=METADATA_TYPE.source_mask, value=METADATA_VALUE.parse_failed),
            )
            payload["source_mask"] = ImageUtils.create_white_image(
                source_image.width if source_image else int(payload.get("width") or 512),
                source_image.height if source_image else int(payload.get("height") or 512),
            )

    # The horde payload names a schedule only indirectly, through a karras flag that cannot express
    # the other schedules the backend supports. A caller that set `scheduler` outright is therefore
    # more specific than the flag and wins; the flag remains the fallback for every horde request.
    if payload.get("scheduler") not in SCHEDULERS:
        payload["scheduler"] = "karras" if payload.get("karras", False) else "normal"

    # Negative and positive prompts arrive merged, separated by ###
    prompt = payload.get("prompt")
    if prompt is not None:
        if "###" in prompt:
            split_prompts = prompt.split("###")
            payload["prompt"] = split_prompts[0]
            payload["negative_prompt"] = split_prompts[1]
    elif prompt == "":
        logger.warning("Empty prompt detected, this is likely to produce poor results")

    # Turn off hires fix if we're not generating a hires image, or if the params are just confused
    try:
        if "hires_fix" in payload:
            baseline = context.baseline
            if baseline == KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_1 and (
                payload["width"] <= 512 or payload["height"] <= 512
            ):
                payload["hires_fix"] = False
            elif baseline == KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_xl and (
                payload["width"] <= 1024 or payload["height"] <= 1024
            ):
                payload["hires_fix"] = False
    except (TypeError, KeyError):
        payload["hires_fix"] = False

    # Turn off hires fix if we're inpainting as the dimensions are from the source image
    if "hires_fix" in payload and (
        payload.get("source_processing") == "inpainting" or payload.get("source_processing") == "outpainting"
    ):
        payload["hires_fix"] = False

    # Use denoising strength for both samplers if no second denoiser specified,
    # but not for txt2img where denoising will always generally be 1.0
    if payload.get("hires_fix") and payload.get("source_processing") and payload.get("source_processing") != "txt2img":
        if not payload.get("hires_fix_denoising_strength"):
            payload["hires_fix_denoising_strength"] = payload.get("denoising_strength")

    if (
        payload.get("workflow") == "qr_code"
        and payload.get("source_processing")
        and payload.get("source_processing") != "txt2img"
    ):
        if not payload.get("hires_fix_denoising_strength"):
            payload["hires_fix_denoising_strength"] = payload.get("denoising_strength")

    return payload, faults


def enforce_schedule_constraints(
    payload: ImageGenPayload,
) -> tuple[ImageGenPayload, list[GenMetadataEntry]]:
    """Move a sampler that would diverge on the requested schedule onto one where it converges.

    Applied on the shared typed path so it holds for every caller, whether the schedule came from a
    horde request's karras flag or was named outright. Serving the divergent combination would return
    colour noise while still consuming the same GPU time and being charged for, so the schedule is
    substituted and the substitution is disclosed on the result rather than being silently applied.
    """
    resolved, substituted = resolve_schedule(payload.sampler_name, payload.scheduler)
    if not substituted:
        return payload, []

    logger.warning(
        f"Sampler {payload.sampler_name} diverges on the {payload.scheduler} schedule; "
        f"running it on {resolved} instead.",
    )
    return (
        payload.model_copy(update={"scheduler": resolved}),
        [
            GenMetadataEntry(
                type=METADATA_TYPE.information,
                value=METADATA_VALUE.see_ref,
                ref=f"scheduler {payload.scheduler}->{resolved} for {payload.sampler_name}"[:255],
            ),
        ],
    )


class UnsupportedScheduleForBaselineError(ValueError):
    """Raised when a request names a schedule that cannot be built for the model it runs on."""

    def __init__(self, *, scheduler: str, baseline: KNOWN_IMAGE_GENERATION_BASELINE | None) -> None:
        self.scheduler = scheduler
        self.baseline = baseline
        super().__init__(
            f"The {scheduler} schedule has no published values for the {baseline} baseline, so it "
            "cannot be built for this model.",
        )


def resolve_sigma_schedule(
    payload: ImageGenPayload,
    context: ModelContext,
) -> SigmaScheduleRequest | None:
    """Return the sigma-generator schedule this payload needs, or None when the graph resolves it.

    Named schedules are ComfyUI's own and need nothing here. The two generator schedules are built
    outside the graph, and Align Your Steps additionally needs the model's family, because its noise
    levels were measured per family rather than derived.

    Args:
        payload: The finalized payload, whose ``scheduler`` names the schedule to run.
        context: The resolved model facts, read for the baseline.

    Returns:
        The schedule to carry beside the graph, or None for a schedule ComfyUI can name.

    Raises:
        UnsupportedScheduleForBaselineError: If Align Your Steps was asked for on a baseline with no
            published noise levels. Substituting another family's levels would run a schedule measured
            for a different model, and silently substituting a different schedule would answer a
            request nobody made, so a direct consumer is told rather than served something else.
    """
    if payload.scheduler not in SIGMA_GENERATOR_SCHEDULES:
        return None

    schedule = SigmaGeneratorSchedule(payload.scheduler)
    if schedule is SigmaGeneratorSchedule.GITS:
        # GITS tables are keyed by coefficient and hold for any model, so there is nothing to resolve.
        return SigmaScheduleRequest(schedule=schedule)

    model_type = align_your_steps_model_type(context.baseline)
    if model_type is None:
        raise UnsupportedScheduleForBaselineError(scheduler=payload.scheduler, baseline=context.baseline)

    return SigmaScheduleRequest(schedule=schedule, align_your_steps_model_type=str(model_type))


def apply_model_compat(
    payload: ImageGenPayload,
    context: ModelContext,
) -> tuple[ImageGenPayload, list[GenMetadataEntry]]:
    """Apply model-fact-dependent compatibility adjustments to a typed payload.

    The typed counterpart of the inpainting rules in :func:`normalize_horde_payload`: a
    request can resolve to an inpainting model regardless of what it asked for, in which
    case the source processing is forced to inpainting, missing inputs are synthesized
    (noise image, all-white mask) with faults recorded, and hires fix is disabled because
    the dimensions come from the source image.
    """
    faults: list[GenMetadataEntry] = []

    payload = payload.model_copy(deep=False)
    payload.model_name = context.horde_model_name

    if not context.is_inpainting_model:
        return payload, faults

    if payload.source_processing not in ("inpainting", "outpainting"):
        logger.warning(
            "Inpainting model detected, but source processing not set to inpainting or outpainting.",
        )
        payload.source_processing = "inpainting"

    if payload.source_image is None:
        logger.warning(
            "Inpainting model detected, but source image is not a valid image. Using a noise image.",
        )
        faults.append(
            GenMetadataEntry(type=METADATA_TYPE.source_image, value=METADATA_VALUE.parse_failed),
        )
        payload.source_image = ImageUtils.create_noise_image(payload.width, payload.height)

    source_image = payload.source_image

    if payload.source_mask is None and (source_image is None or not ImageUtils.has_alpha_channel(source_image)):
        logger.warning(
            "Inpainting model detected, but no source mask provided. Using an all white mask.",
        )
        faults.append(
            GenMetadataEntry(type=METADATA_TYPE.source_mask, value=METADATA_VALUE.parse_failed),
        )
        payload.source_mask = ImageUtils.create_white_image(
            source_image.width if source_image else payload.width,
            source_image.height if source_image else payload.height,
        )

    # The (in|out)painting dimensions come from the source image, so a two-pass upscale is
    # meaningless; mirrors the dict-path disable rule.
    payload.hires_fix = False

    return payload, faults


def disable_hires_fix_for_masked_img2img(payload: ImageGenPayload) -> ImageGenPayload:
    """Disable hires fix when img2img carries a mask (explicit, or an alpha channel acting as one).

    The masked img2img graph is single-pass, so the hires-fix first-pass shrink would become the
    final output size. Masked payloads with a controlnet keep hires fix: the controlnet hires
    graph has a real second pass. Must run before :func:`resize_sources_to_request`, which merges
    the mask into the source alpha channel and makes every masked payload look alpha-masked.
    """
    if not payload.hires_fix or payload.control_type or not IMG2IMG_MASK.is_set(payload):
        return payload
    payload = payload.model_copy(deep=False)
    payload.hires_fix = False
    return payload


def resize_sources_to_request(payload: ImageGenPayload) -> ImageGenPayload:
    """Ensure the source image and mask match the requested generation size.

    Typed port of ``ImageUtils.resize_sources_to_request``: resizes the source image (to the
    first-pass resolution for hires-fix/controlnet jobs), resizes the mask, and merges the
    mask into the source image's alpha channel.
    """
    source_image = payload.source_image
    if source_image is None:
        return payload

    payload = payload.model_copy(deep=False)

    new_width, new_height = payload.width, payload.height
    if payload.hires_fix or payload.control_type:
        explicit_first_pass_width = payload.hires_fix_first_pass_width
        explicit_first_pass_height = payload.hires_fix_first_pass_height
        if payload.hires_fix and explicit_first_pass_width is not None and explicit_first_pass_height is not None:
            new_width, new_height = explicit_first_pass_width, explicit_first_pass_height
        else:
            new_width, new_height = ImageUtils.get_first_pass_image_resolution_min(payload.width, payload.height)
    if source_image.size != (new_width, new_height):
        source_image = source_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        payload.source_image = source_image

    source_mask = payload.source_mask
    if source_mask is None:
        return payload

    if source_mask.size != (payload.width, payload.height):
        source_mask = source_mask.resize((payload.width, payload.height))
        payload.source_mask = source_mask

    payload.source_image = ImageUtils.add_image_alpha_channel(source_image, source_mask)
    return payload
