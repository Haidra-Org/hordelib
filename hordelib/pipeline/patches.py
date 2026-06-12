"""First-class graph patch operations.

These are the dynamic graph mutations previously inlined in
``HordeLib._final_pipeline_adjustments``, as pure functions over API-format graph dicts so
they can be unit-tested without a GPU.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from horde_model_reference.meta_consts import KNOWN_IMAGE_GENERATION_BASELINE
from loguru import logger

from hordelib.execution.graph_utils import GraphDict, reconnect_input
from hordelib.pipeline.constants import CONTROLNET_IMAGE_PREPROCESSOR_MAP, CONTROLNET_MODEL_MAP
from hordelib.utils.image_utils import ImageUtils


@dataclass(frozen=True)
class ResolvedLora:
    """A LoRA that has been validated/downloaded and resolved to an on-disk filename."""

    filename: str
    strength_model: float
    strength_clip: float


def insert_lora_chain(
    graph: GraphDict,
    loras: Sequence[ResolvedLora],
    *,
    flux: bool = False,
) -> None:
    """Inject a chain of HordeLoraLoader nodes between the model loader and its consumers.

    The first LoRA connects to ``model_loader``; each subsequent LoRA chains from the previous
    one. The last LoRA replaces the model/clip sources of the samplers and clip_skip (or, for
    Flux pipelines, of ``cfg_guider``/``basic_scheduler``). Targets absent from the graph are
    skipped, matching the variant pipelines that lack them (e.g. no ``upscale_sampler``).

    Args:
        graph: The pipeline graph to mutate.
        loras: The LoRAs to chain, in application order.
        flux: Whether this is a Flux pipeline (different model consumers).
    """
    if not loras:
        return

    for index, lora in enumerate(loras):
        source = "model_loader" if index == 0 else f"lora_{index - 1}"
        graph[f"lora_{index}"] = {
            "inputs": {
                "model": [source, 0],
                "clip": [source, 1],
                "lora_name": lora.filename,
                "strength_model": lora.strength_model,
                "strength_clip": lora.strength_clip,
            },
            "class_type": "HordeLoraLoader",
            "_meta": {"title": f"lora_{index}"},
        }

    last = f"lora_{len(loras) - 1}"
    if flux:
        reconnect_input(graph, "cfg_guider.model", last)
        reconnect_input(graph, "basic_scheduler.model", last)
    else:
        reconnect_input(graph, "sampler.model", last)
        reconnect_input(graph, "upscale_sampler.model", last)
        reconnect_input(graph, "clip_skip.clip", last)


def rewire_img2img(graph: GraphDict, *, flux: bool = False) -> None:
    """Feed the sampler from the VAE-encoded source image instead of the empty latent.

    Args:
        graph: The pipeline graph to mutate.
        flux: Whether this is a Flux pipeline (different sampler node).
    """
    if flux:
        reconnect_input(graph, "sampler_custom_advanced.latent_image", "vae_encode")
    else:
        reconnect_input(graph, "sampler.latent_image", "vae_encode")


def rewire_cascade_img2img(graph: GraphDict) -> None:
    """Feed both Stable Cascade samplers from the stage-C VAE-encoded source image."""
    reconnect_input(graph, "sampler_stage_c.latent_image", "stablecascade_stagec_vaeencode")
    reconnect_input(graph, "sampler_stage_b.latent_image", "stablecascade_stagec_vaeencode")


@dataclass(frozen=True)
class RemixImage:
    """An extra source image for a Stable Cascade remix, with its conditioning strength."""

    image: Any
    strength: float = 1.0


def insert_remix_image_chain(graph: GraphDict, extra_images: Sequence[RemixImage]) -> None:
    """Chain extra remix images into the Stable Cascade unCLIP conditioning.

    The primary source image occupies index 0 (``sc_image_loader``/``unclip_conditioning_0``
    in the remix pipeline), so extras start at index 1. Each image gets a loader, a CLIP
    vision encode, and an unCLIP conditioning node that ingests the previous conditioning;
    the last conditioning replaces the stage-C sampler's positive input.

    Args:
        graph: The pipeline graph to mutate.
        extra_images: The extra images, in chain order.
    """
    if not extra_images:
        return

    for image_index, extra in enumerate(extra_images):
        node_index = image_index + 1
        graph[f"sc_image_loader_{node_index}"] = {
            "inputs": {"image": extra.image, "upload": "image"},
            "class_type": "HordeImageLoader",
        }
        graph[f"clip_vision_encode_{node_index}"] = {
            "inputs": {
                "clip_vision": ["model_loader_stage_c", 3],
                "image": [f"sc_image_loader_{node_index}", 0],
                "crop": "center",
            },
            "class_type": "CLIPVisionEncode",
        }
        graph[f"unclip_conditioning_{node_index}"] = {
            "inputs": {
                "strength": extra.strength,
                "noise_augmentation": 0,
                # Each conditioning ingests the conditioning before it like a chain
                "conditioning": [f"unclip_conditioning_{node_index - 1}", 0],
                "clip_vision_output": [f"clip_vision_encode_{node_index}", 0],
            },
            "class_type": "unCLIPConditioning",
        }

    reconnect_input(graph, "sampler_stage_c.positive", f"unclip_conditioning_{len(extra_images)}")


def configure_controlnet(
    graph: GraphDict,
    *,
    control_type: str,
    image_is_control: bool,
    return_control_map: bool,
    width: int,
    height: int,
) -> dict[str, Any]:
    """Configure the controlnet model loader and AIO preprocessor for a control type.

    Args:
        graph: The pipeline graph to mutate (only for the ``return_control_map`` rewire).
        control_type: The horde control type (e.g. ``"canny"``).
        image_is_control: The source image already is the control map.
        return_control_map: Return the annotated control map instead of a generation.
        width: Requested generation width.
        height: Requested generation height.

    Returns:
        dict[str, Any]: Dotted graph params to apply.
    """
    params: dict[str, Any] = {}

    model_name = CONTROLNET_MODEL_MAP.get(control_type)
    if not model_name:
        logger.error("Controlnet model not found: control_type={}", control_type)
    params["controlnet_model_loader.control_net_name"] = model_name

    aux_preprocessor = CONTROLNET_IMAGE_PREPROCESSOR_MAP.get(control_type)
    if not aux_preprocessor:
        logger.error("Controlnet preprocessor not found: control_type={}", control_type)

    # "none" makes the AIO_Preprocessor node pass the (already-control-map) image through
    params["preprocessor.preprocessor"] = "none" if image_is_control else aux_preprocessor

    # Run detection at the generation resolution (the aux node resamples internally)
    params["preprocessor.resolution"] = min(width, height)

    if return_control_map:
        reconnect_input(graph, "output_image.images", "preprocessor")

    return params


LAYERDIFFUSE_BASELINES = frozenset(
    {
        KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_1,
        KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_xl,
    },
)
"""Baselines layerdiffuse (transparent generation) supports; SD2/SD3/Cascade are not."""


def apply_layerdiffuse(
    graph: GraphDict,
    *,
    baseline: KNOWN_IMAGE_GENERATION_BASELINE | None,
    hires_fix: bool,
) -> dict[str, Any]:
    """Route the diffusion and decode path through the layerdiffuse nodes for transparency.

    A transparent gen is basically a fancy lora, so ``model_loader.will_load_loras`` is set
    even when the baseline is unsupported (matching the legacy behavior of leaving the
    graph itself untouched in that case).

    Args:
        graph: The pipeline graph to mutate.
        baseline: The model's baseline; only SD1/SDXL are rewired.
        hires_fix: Whether the upscale sampler also needs the layerdiffuse model.

    Returns:
        dict[str, Any]: Dotted graph params to apply.
    """
    params: dict[str, Any] = {"model_loader.will_load_loras": True}

    if baseline not in LAYERDIFFUSE_BASELINES:
        return params

    reconnect_input(graph, "sampler.model", "layer_diffuse_apply")
    reconnect_input(graph, "layer_diffuse_apply.model", "model_loader")
    reconnect_input(graph, "output_image.images", "layer_diffuse_decode_rgba")
    reconnect_input(graph, "layer_diffuse_decode_rgba.images", "vae_decode")
    if hires_fix:
        reconnect_input(graph, "upscale_sampler.model", "layer_diffuse_apply")

    if baseline == KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_1:
        params["layer_diffuse_apply.config"] = "SD15, Attention Injection, attn_sharing"
        params["layer_diffuse_decode_rgba.sd_version"] = "SD15"
    else:
        params["layer_diffuse_apply.config"] = "SDXL, Conv Injection"
        params["layer_diffuse_decode_rgba.sd_version"] = "SDXL"

    return params


def hires_fix_first_pass_resolution(
    baseline: KNOWN_IMAGE_GENERATION_BASELINE | None,
    width: int,
    height: int,
) -> dict[str, Any]:
    """Compute the two-pass hires-fix resolutions.

    The requested resolution becomes the upscale *target*; the empty latent is shrunk to a
    baseline-appropriate first-pass resolution.

    Args:
        baseline: The model's baseline, which determines the first-pass sizing strategy.
        width: Requested (target) width.
        height: Requested (target) height.

    Returns:
        dict[str, Any]: Dotted graph params to apply.
    """
    params: dict[str, Any] = {
        "latent_upscale.width": width,
        "latent_upscale.height": height,
    }

    new_width, new_height = (None, None)
    if baseline is not None:
        if baseline == KNOWN_IMAGE_GENERATION_BASELINE.stable_cascade:
            new_width, new_height = ImageUtils.get_first_pass_image_resolution_max(width, height)
        elif baseline == KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_xl:
            new_width, new_height = ImageUtils.get_first_pass_image_resolution_sdxl(width, height)
        else:  # fall through case; only `stable_diffusion_1` at time of writing
            new_width, new_height = ImageUtils.get_first_pass_image_resolution_min(width, height)

    if new_width and new_height:
        params["empty_latent_image.width"] = new_width
        params["empty_latent_image.height"] = new_height
    else:
        logger.error("Could not determine new image size for hires fix. Using 1024x1024.")
        params["empty_latent_image.width"] = 1024
        params["empty_latent_image.height"] = 1024

    return params


QR_MODULE_DRAWERS = frozenset(
    {"square", "gapped square", "circle", "rounded", "vertical bars", "horizontal bars"},
)


def qr_params_from_extra_texts(
    extra_texts: Sequence[dict[str, Any]],
    *,
    prompt: str,
    width: int,
    height: int,
) -> dict[str, Any]:
    """Parse the qr_code workflow's ``extra_texts`` entries into dotted graph params.

    Args:
        extra_texts: ``{"text": ..., "reference": ...}`` entries from the payload.
        prompt: The generation prompt (fallback for the function layer prompt).
        width: Requested generation width.
        height: Requested generation height.

    Returns:
        dict[str, Any]: Dotted graph params to apply (offsets may be absent; see
        :func:`qr_layout_params`).
    """
    params: dict[str, Any] = {
        "qr_code_split.max_image_size": max(width, height),
        "qr_code_split.text": "https://haidra.net",
    }

    for text in extra_texts:
        if text["reference"] in ["qr_code", "qr_text"]:
            params["qr_code_split.text"] = text["text"]
        if text["reference"] == "protocol" and text["text"].lower() in ["https", "http"]:
            params["qr_code_split.protocol"] = text["text"].capitalize()
        if text["reference"] == "module_drawer" and text["text"].lower() in QR_MODULE_DRAWERS:
            params["qr_code_split.module_drawer"] = text["text"].capitalize()
        if text["reference"] == "function_layer_prompt":
            params["function_layer_prompt.text"] = text["text"]
        if text["reference"] == "x_offset" and text["text"].lstrip("-").isdigit():
            x_offset = int(text["text"])
            if x_offset < 0:
                x_offset = 10
            params["qr_flattened_composite.x"] = x_offset
        if text["reference"] == "y_offset" and text["text"].lstrip("-").isdigit():
            y_offset = int(text["text"])
            if y_offset < 0:
                y_offset = 10
            params["qr_flattened_composite.y"] = y_offset
        if text["reference"] == "qr_border" and text["text"].lstrip("-").isdigit():
            border = int(text["text"])
            if border < 0:
                border = 10
            params["qr_code_split.border"] = border

    if not params.get("qr_code_split.protocol"):
        params["qr_code_split.protocol"] = "None"
    if not params.get("function_layer_prompt.text"):
        params["function_layer_prompt.text"] = prompt

    return params


def qr_layout_params(
    *,
    width: int,
    height: int,
    qr_size: int,
    x_offset: int | None = None,
    y_offset: int | None = None,
) -> dict[str, Any]:
    """Place the QR code layers within the image, clamping requested offsets to fit.

    Args:
        width: Requested generation width.
        height: Requested generation height.
        qr_size: The rendered QR code size in pixels.
        x_offset: A requested x offset, if any (centered otherwise).
        y_offset: A requested y offset, if any (centered otherwise).

    Returns:
        dict[str, Any]: Dotted graph params positioning every QR composite layer.
    """
    if x_offset is None:
        x_offset = int((width / 2) - qr_size / 2)
        # I don't know why but through trial and error I've discovered that the QR codes
        # are more legible when they're placed in an offset which is a multiple of 64
        x_offset = x_offset - (x_offset % 64) if x_offset % 64 != 0 else x_offset
    if x_offset > int(width - qr_size):
        x_offset = int(width - qr_size) - 10

    if y_offset is None:
        y_offset = int((height / 2) - qr_size / 2)
        y_offset = y_offset - (y_offset % 64) if y_offset % 64 != 0 else y_offset
    if y_offset > int(height - qr_size):
        y_offset = int(height - qr_size) - 10

    return {
        "qr_flattened_composite.x": x_offset,
        "qr_flattened_composite.y": y_offset,
        "module_layer_composite.x": x_offset,
        "module_layer_composite.y": y_offset,
        "function_layer_composite.x": x_offset,
        "function_layer_composite.y": y_offset,
        "mask_composite.x": x_offset,
        "mask_composite.y": y_offset,
    }
