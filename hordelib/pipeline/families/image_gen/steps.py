"""The image family's patch steps: context-dependent graph mutation, in legacy order.

Each step is gated on the payload/context facts that make it applicable, so the shared
:data:`IMAGE_PATCH_STEPS` sequence is safe on every image graph (a step that does not apply
is a no-op). New pipelines opt into exactly the steps they need; the established pipelines
share the full sequence, whose observable behavior is pinned by the snapshot corpus.
"""

from horde_model_reference.meta_consts import KNOWN_IMAGE_GENERATION_BASELINE, get_baseline_native_resolution
from loguru import logger

from hordelib.pipeline.context import ModelContext
from hordelib.pipeline.definition import PatchStep
from hordelib.pipeline.families.image_gen.baselines import (
    FLUX_BASELINES,
    IMAGE_BASELINE_PROFILES,
    LoaderKind,
)
from hordelib.pipeline.graph import ComfyGraph
from hordelib.pipeline.patches import (
    RemixImage,
    apply_layerdiffuse,
    configure_controlnet,
    hires_fix_first_pass_resolution,
    insert_lora_chain,
    insert_remix_image_chain,
    qr_layout_params,
    qr_params_from_extra_texts,
    rewire_cascade_img2img,
    rewire_img2img,
)
from hordelib.pipeline.payload import ImageGenPayload
from hordelib.utils.image_utils import ImageUtils

__all__ = [
    "IMAGE_PATCH_STEPS",
    "apply_controlnet",
    "apply_hires_fix_resolution",
    "apply_img2img_rewire",
    "apply_layerdiffuse_transparency",
    "apply_lora_chain",
    "apply_main_model",
    "apply_cascade_stage_models",
    "apply_qr_layout",
    "apply_remix_chain",
    "apply_upscale_sampler_steps",
]


def apply_lora_chain(graph: ComfyGraph, payload: ImageGenPayload, context: ModelContext) -> None:
    """Insert the resolved LoRA chain between the model loader and its consumers."""
    if context.resolved_loras:
        insert_lora_chain(graph.raw, context.resolved_loras, flux=context.baseline in FLUX_BASELINES)


def apply_main_model(graph: ComfyGraph, payload: ImageGenPayload, context: ModelContext) -> None:
    """Point the main model loader at the resolved checkpoint/diffusion-model file."""
    profile = IMAGE_BASELINE_PROFILES.get(context.baseline) if context.baseline is not None else None
    loads_split_files = profile is not None and profile.loader is LoaderKind.UNET
    graph.set_inputs(
        {
            "model_loader.ckpt_name": context.main_file,
            "model_loader.model_name": context.main_file,
            "model_loader.horde_model_name": context.horde_model_name,
            "model_loader.will_load_loras": context.will_load_loras,
            # The HordeCheckpointLoader needs to know what file to load; "unet" routes the split-files
            # baselines (qwen, z-image) through the diffusion-model loader, None keeps normal SD checkpoints working.
            "model_loader.file_type": "unet" if loads_split_files else None,
        },
    )


def apply_cascade_stage_models(graph: ComfyGraph, payload: ImageGenPayload, context: ModelContext) -> None:
    """Point the Stable Cascade stage-b/stage-c loaders at their resolved files."""
    graph.set_inputs(
        {
            "model_loader_stage_c.ckpt_name": context.extra_files.get("stable_cascade_stage_c"),
            "model_loader_stage_c.model_name": context.extra_files.get("stable_cascade_stage_c"),
            "model_loader_stage_c.horde_model_name": context.horde_model_name,
            "model_loader_stage_c.file_type": "stable_cascade_stage_c",
            "model_loader_stage_c.will_load_loras": False,  # FIXME: Once we support loras for cascade
            "model_loader_stage_c.seamless_tiling_enabled": False,
            "model_loader_stage_b.ckpt_name": context.extra_files.get("stable_cascade_stage_b"),
            "model_loader_stage_b.model_name": context.extra_files.get("stable_cascade_stage_b"),
            "model_loader_stage_b.horde_model_name": context.horde_model_name,
            "model_loader_stage_b.file_type": "stable_cascade_stage_b",
            "model_loader_stage_b.will_load_loras": False,  # FIXME: Once we support loras for cascade
            "model_loader_stage_b.seamless_tiling_enabled": False,
        },
    )


def apply_hires_fix_resolution(graph: ComfyGraph, payload: ImageGenPayload, context: ModelContext) -> None:
    """Shrink the first pass to a baseline-appropriate resolution when hires fix is on."""
    if not payload.hires_fix:
        return
    graph.set_inputs(hires_fix_first_pass_resolution(context.baseline, payload.width, payload.height))


def apply_upscale_sampler_steps(graph: ComfyGraph, payload: ImageGenPayload, context: ModelContext) -> None:
    """Compute the hires-fix second-pass step count from the model's native resolution."""
    if not graph.has_node("upscale_sampler"):
        return
    native_resolution = get_baseline_native_resolution(context.baseline) if context.baseline is not None else None
    graph.set_input(
        "upscale_sampler.steps",
        ImageUtils.calc_upscale_sampler_steps(
            model_native_resolution=native_resolution,
            width=payload.width,
            height=payload.height,
            hires_fix_denoising_strength=payload.hires_fix_denoising_strength,
            ddim_steps=payload.ddim_steps,
        ),
    )


def apply_controlnet(graph: ComfyGraph, payload: ImageGenPayload, context: ModelContext) -> None:
    """Configure the controlnet model/preprocessor pair for the requested control type."""
    if not payload.control_type:
        return
    controlnet_params = configure_controlnet(
        graph.raw,
        control_type=payload.control_type,
        image_is_control=payload.image_is_control,
        return_control_map=payload.return_control_map,
        width=payload.width,
        height=payload.height,
    )
    graph.set_inputs(controlnet_params)
    logger.info(
        "controlnet.configured",
        model_name=controlnet_params["controlnet_model_loader.control_net_name"],
        preprocessor=controlnet_params["preprocessor.preprocessor"],
        return_control_map=payload.return_control_map,
        image_is_control=payload.image_is_control,
    )


def apply_img2img_rewire(graph: ComfyGraph, payload: ImageGenPayload, context: ModelContext) -> None:
    """Feed the sampler from the VAE-encoded source image instead of the empty latent."""
    if payload.source_image is None:
        return
    # The rewires skip graphs lacking the relevant nodes, matching the legacy global attempt.
    if graph.has_node("image_loader"):
        rewire_img2img(graph.raw, flux=context.baseline in FLUX_BASELINES)
    if graph.has_node("sc_image_loader"):
        rewire_cascade_img2img(graph.raw)


def apply_remix_chain(graph: ComfyGraph, payload: ImageGenPayload, context: ModelContext) -> None:
    """Chain the extra remix source images into the Stable Cascade unCLIP conditioning."""
    if payload.source_processing != "remix":
        return
    insert_remix_image_chain(
        graph.raw,
        [RemixImage(image=extra.image, strength=extra.strength) for extra in payload.extra_source_images],
    )


def apply_qr_layout(graph: ComfyGraph, payload: ImageGenPayload, context: ModelContext) -> None:
    """Parse the qr_code workflow's extra_texts and lay out the QR composite layers."""
    if payload.workflow != "qr_code":
        return

    params = qr_params_from_extra_texts(
        [text.model_dump() for text in payload.extra_texts],
        prompt=payload.prompt,
        width=payload.width,
        height=payload.height,
    )

    try:
        # ComfyQR registers QRByModuleSizeSplitFunctionPatterns under this node id; the layout
        # must know the rendered QR size before placing the composite layers.
        from hordelib.comfy_horde import get_node_class

        test_qr = get_node_class("comfy-qr-by-module-split")()
        _, _, _, _, _, qr_size = test_qr.generate_qr(
            protocol=params.get("qr_code_split.protocol"),
            text=params["qr_code_split.text"],
            module_size=16,
            max_image_size=params["qr_code_split.max_image_size"],
            fill_hexcolor="#FFFFFF",
            back_hexcolor="#000000",
            error_correction="High",
            border=1,
            module_drawer="Square",
        )
    except RuntimeError as err:
        logger.error(err)
        params["qr_code_split.text"] = "This QR Code is too large for this image"
        qr_size = 624

    params.update(
        qr_layout_params(
            width=payload.width,
            height=payload.height,
            qr_size=qr_size,
            # Explicit 0 offsets fall back to centered placement, as they always have
            x_offset=params.get("qr_flattened_composite.x") or None,
            y_offset=params.get("qr_flattened_composite.y") or None,
        ),
    )
    if context.baseline is KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_1:
        params["controlnet_qr_model_loader.control_net_name"] = "control_v1p_sd15_qrcode_monster_v2.safetensors"

    graph.set_inputs(params)
    logger.info(
        "qr.configured",
        qr_text=params["qr_code_split.text"],
        qr_size=qr_size,
        x_offset=params["qr_flattened_composite.x"],
        y_offset=params["qr_flattened_composite.y"],
    )


def apply_layerdiffuse_transparency(graph: ComfyGraph, payload: ImageGenPayload, context: ModelContext) -> None:
    """Route diffusion/decode through the layerdiffuse nodes for transparent generation."""
    if payload.transparent is not True:
        return
    graph.set_inputs(
        apply_layerdiffuse(graph.raw, baseline=context.baseline, hires_fix=payload.hires_fix is True),
    )


IMAGE_PATCH_STEPS: tuple[PatchStep[ImageGenPayload, ModelContext], ...] = (
    apply_lora_chain,
    apply_main_model,
    apply_cascade_stage_models,
    apply_hires_fix_resolution,
    apply_upscale_sampler_steps,
    apply_controlnet,
    apply_img2img_rewire,
    apply_remix_chain,
    apply_qr_layout,
    apply_layerdiffuse_transparency,
)
"""The shared step sequence, in the exact order of the legacy ``_final_pipeline_adjustments``."""
