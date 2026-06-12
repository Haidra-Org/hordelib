"""The image-generation pipeline families: selection, bindings, and patch steps.

The predicates and priorities reproduce the legacy decision tree in
``HordeLib._get_appropriate_pipeline`` exactly:

1. qr_code workflow
2. cascade (remix / 2pass / base)
3. flux
4. qwen
5. controlnet (annotator / hires / base)
6. img2img-with-mask / inpainting / outpainting
7. hires fix
8. generic stable diffusion (also plain img2img)

The bindings are the typed port of the legacy ``PAYLOAD_TO_PIPELINE_PARAMETER_MAPPING``: the
full candidate set is declared once and each template keeps the bindings whose target node
exists in its graph (the same apply-where-present semantics the legacy translation loop had).
Everything context-dependent (model files, LoRA chains, controlnet, hires-fix resolutions,
QR layout, layerdiffuse) happens in the shared patch-step sequence, in the same order the
legacy ``_final_pipeline_adjustments`` performed it.
"""

from pathlib import Path
from typing import Any

from horde_model_reference.meta_consts import KNOWN_IMAGE_GENERATION_BASELINE, get_baseline_native_resolution
from loguru import logger

from hordelib.pipeline.constants import SAMPLERS_MAP
from hordelib.pipeline.context import ModelContext
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
from hordelib.pipeline.registry import PipelineRegistry, PipelineSpec
from hordelib.pipeline.template import ParamBinding, PipelineTemplate
from hordelib.utils.image_utils import ImageUtils

PIPELINES_DIR = Path(__file__).parent.parent.parent / "pipelines"

type ImageGenSpec = PipelineSpec[ImageGenPayload, ModelContext]
type ImageGenTemplate = PipelineTemplate[ImageGenPayload, ModelContext]
type ImageGenBinding = ParamBinding[ImageGenPayload]


def _has_img2img_mask(payload: ImageGenPayload) -> bool:
    if payload.source_processing != "img2img":
        return False
    if payload.source_mask is not None:
        return True
    return payload.source_image is not None and len(payload.source_image.getbands()) == 4


def _comfy_sampler(payload: ImageGenPayload) -> str | None:
    return SAMPLERS_MAP.get(payload.sampler_name)


def _comfy_clip_skip(payload: ImageGenPayload) -> int:
    # ComfyUI counts clip skip negatively (-1, -2, ...)
    return -payload.clip_skip if payload.clip_skip > 0 else payload.clip_skip


ALL_IMAGE_BINDINGS: tuple[ImageGenBinding, ...] = (
    ParamBinding(target="sampler.sampler_name", transform=_comfy_sampler),
    ParamBinding(target="sampler.cfg", source="cfg_scale"),
    ParamBinding(target="sampler.denoise", source="denoising_strength"),
    ParamBinding(target="sampler.seed", source="seed"),
    ParamBinding(target="sampler.noise_seed", source="seed"),
    ParamBinding(target="sampler.scheduler", source="scheduler"),
    ParamBinding(target="sampler.steps", source="ddim_steps"),
    ParamBinding(target="empty_latent_image.height", source="height"),
    ParamBinding(target="empty_latent_image.width", source="width"),
    ParamBinding(target="empty_latent_image.batch_size", source="n_iter"),
    ParamBinding(target="repeat_image_batch.amount", source="n_iter"),
    ParamBinding(target="clip_skip.stop_at_clip_layer", transform=_comfy_clip_skip),
    ParamBinding(target="prompt.text", source="prompt"),
    ParamBinding(target="negative_prompt.text", source="negative_prompt"),
    ParamBinding(target="model_loader.seamless_tiling_enabled", source="tiling"),
    ParamBinding(target="image_loader.image", source="source_image"),
    ParamBinding(target="upscale_sampler.denoise", source="hires_fix_denoising_strength"),
    ParamBinding(target="upscale_sampler.seed", source="seed"),
    ParamBinding(target="upscale_sampler.cfg", source="cfg_scale"),
    ParamBinding(target="upscale_sampler.sampler_name", transform=_comfy_sampler),
    ParamBinding(target="controlnet_apply.strength", source="control_strength"),
    ParamBinding(target="controlnet_model_loader.control_net_name", source="control_type"),
    # Flux
    ParamBinding(target="cfg_guider.cfg", source="cfg_scale"),
    ParamBinding(target="random_noise.noise_seed", source="seed"),
    ParamBinding(target="k_sampler_select.sampler_name", transform=_comfy_sampler),
    ParamBinding(target="basic_scheduler.denoise", source="denoising_strength"),
    ParamBinding(target="basic_scheduler.steps", source="ddim_steps"),
    # Stable Cascade
    ParamBinding(target="stable_cascade_empty_latent_image.width", source="width"),
    ParamBinding(target="stable_cascade_empty_latent_image.height", source="height"),
    ParamBinding(target="stable_cascade_empty_latent_image.batch_size", source="n_iter"),
    ParamBinding(target="sc_image_loader.image", source="source_image"),
    ParamBinding(target="sc_image_loader_0.image", source="source_image"),
    ParamBinding(target="sampler_stage_c.sampler_name", transform=_comfy_sampler),
    ParamBinding(target="sampler_stage_b.sampler_name", transform=_comfy_sampler),
    ParamBinding(target="sampler_stage_c.cfg", source="cfg_scale"),
    ParamBinding(target="sampler_stage_c.denoise", source="denoising_strength"),
    ParamBinding(target="sampler_stage_b.seed", source="seed"),
    ParamBinding(target="sampler_stage_c.seed", source="seed"),
    ParamBinding(target="sampler_stage_b.steps", source="ddim_steps", multiplier=0.33),
    ParamBinding(target="sampler_stage_c.steps", source="ddim_steps", multiplier=0.67),
    # Stable Cascade 2pass
    ParamBinding(target="2pass_sampler_stage_c.sampler_name", transform=_comfy_sampler),
    ParamBinding(target="2pass_sampler_stage_c.steps", source="ddim_steps", multiplier=0.67),
    ParamBinding(target="2pass_sampler_stage_c.denoise", source="hires_fix_denoising_strength"),
    ParamBinding(target="2pass_sampler_stage_b.sampler_name", transform=_comfy_sampler),
    ParamBinding(target="2pass_sampler_stage_b.steps", source="ddim_steps", multiplier=0.33),
    # QR Codes
    ParamBinding(target="sampler_bg.sampler_name", transform=_comfy_sampler),
    ParamBinding(target="sampler_bg.cfg", source="cfg_scale"),
    ParamBinding(target="sampler_bg.denoise", source="denoising_strength"),
    ParamBinding(target="sampler_bg.seed", source="seed"),
    ParamBinding(target="sampler_bg.steps", source="ddim_steps"),
    ParamBinding(target="sampler_bg.noise_seed", source="seed"),
    ParamBinding(target="sampler_fg.sampler_name", transform=_comfy_sampler),
    ParamBinding(target="sampler_fg.cfg", source="cfg_scale"),
    ParamBinding(target="sampler_fg.denoise", source="denoising_strength"),
    ParamBinding(target="sampler_fg.seed", source="seed"),
    ParamBinding(target="sampler_fg.steps", source="ddim_steps"),
    ParamBinding(target="sampler_fg.noise_seed", source="seed"),
    ParamBinding(target="controlnet_bg.strength", source="control_strength"),
    ParamBinding(target="solidmask_grey.width", source="width"),
    ParamBinding(target="solidmask_grey.height", source="height"),
    ParamBinding(target="solidmask_white.width", source="width"),
    ParamBinding(target="solidmask_white.height", source="height"),
    ParamBinding(target="solidmask_black.width", source="width"),
    ParamBinding(target="solidmask_black.height", source="height"),
    ParamBinding(target="qr_code_split.max_image_size", source="width"),
)


# --- Patch steps (shared sequence, same order as the legacy _final_pipeline_adjustments) ---


def _insert_lora_chain_step(graph: ComfyGraph, payload: ImageGenPayload, context: ModelContext) -> None:
    if context.resolved_loras:
        insert_lora_chain(graph.raw, context.resolved_loras, flux=context.is_flux)


def _apply_model_context_step(graph: ComfyGraph, payload: ImageGenPayload, context: ModelContext) -> None:
    params: dict[str, Any] = {
        "model_loader.ckpt_name": context.main_file,
        "model_loader.model_name": context.main_file,
        "model_loader.horde_model_name": context.horde_model_name,
        "model_loader.will_load_loras": context.will_load_loras,
        # The HordeCheckpointLoader needs to know what file to load; "unet" routes qwen
        # through the diffusion-model loader, None keeps normal SD checkpoints working.
        "model_loader.file_type": "unet" if context.is_qwen else None,
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
    }
    graph.set_inputs(params)


def _apply_hires_fix_step(graph: ComfyGraph, payload: ImageGenPayload, context: ModelContext) -> None:
    if not payload.hires_fix:
        return
    graph.set_inputs(hires_fix_first_pass_resolution(context.baseline, payload.width, payload.height))


def _apply_upscale_sampler_steps_step(graph: ComfyGraph, payload: ImageGenPayload, context: ModelContext) -> None:
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


def _configure_controlnet_step(graph: ComfyGraph, payload: ImageGenPayload, context: ModelContext) -> None:
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


def _rewire_img2img_step(graph: ComfyGraph, payload: ImageGenPayload, context: ModelContext) -> None:
    if payload.source_image is None:
        return
    # The rewires skip graphs lacking the relevant nodes, matching the legacy global attempt.
    if graph.has_node("image_loader"):
        rewire_img2img(graph.raw, flux=context.is_flux)
    if graph.has_node("sc_image_loader"):
        rewire_cascade_img2img(graph.raw)


def _insert_remix_chain_step(graph: ComfyGraph, payload: ImageGenPayload, context: ModelContext) -> None:
    if payload.source_processing != "remix":
        return
    insert_remix_image_chain(
        graph.raw,
        [RemixImage(image=extra.image, strength=extra.strength) for extra in payload.extra_source_images],
    )


def _apply_qr_step(graph: ComfyGraph, payload: ImageGenPayload, context: ModelContext) -> None:
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
    if context.baseline == KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_1:
        params["controlnet_qr_model_loader.control_net_name"] = "control_v1p_sd15_qrcode_monster_v2.safetensors"

    graph.set_inputs(params)
    logger.info(
        "qr.configured",
        qr_text=params["qr_code_split.text"],
        qr_size=qr_size,
        x_offset=params["qr_flattened_composite.x"],
        y_offset=params["qr_flattened_composite.y"],
    )


def _apply_layerdiffuse_step(graph: ComfyGraph, payload: ImageGenPayload, context: ModelContext) -> None:
    if payload.transparent is not True:
        return
    graph.set_inputs(
        apply_layerdiffuse(graph.raw, baseline=context.baseline, hires_fix=payload.hires_fix is True),
    )


IMAGE_PATCH_STEPS = (
    _insert_lora_chain_step,
    _apply_model_context_step,
    _apply_hires_fix_step,
    _apply_upscale_sampler_steps_step,
    _configure_controlnet_step,
    _rewire_img2img_step,
    _insert_remix_chain_step,
    _apply_qr_step,
    _apply_layerdiffuse_step,
)


def _template(name: str) -> ImageGenTemplate:
    graph_file = PIPELINES_DIR / f"pipeline_{name}.json"
    # Keep the bindings whose target node exists in this template's graph — the same
    # apply-where-present semantics as the legacy parameter mapping.
    node_titles = set(ComfyGraph.from_file(graph_file).node_titles())
    bindings = tuple(b for b in ALL_IMAGE_BINDINGS if b.target.split(".", 1)[0] in node_titles)
    return PipelineTemplate(
        name=name,
        graph_file=graph_file,
        bindings=bindings,
        patch_steps=IMAGE_PATCH_STEPS,
    )


def build_default_registry() -> PipelineRegistry[ImageGenPayload, ModelContext]:
    registry: PipelineRegistry[ImageGenPayload, ModelContext] = PipelineRegistry()

    specs: list[ImageGenSpec] = [
        # Workflow overrides beat everything else
        PipelineSpec(
            template=_template("qr_code"),
            predicate=lambda p, c: p.workflow == "qr_code",
            priority=100,
        ),
        # Baseline-specific families must be checked before the generic SD fallback
        PipelineSpec(
            template=_template("stable_cascade_remix"),
            predicate=lambda p, c: c.is_cascade and p.source_processing == "remix",
            priority=90,
        ),
        PipelineSpec(
            template=_template("stable_cascade_2pass"),
            predicate=lambda p, c: c.is_cascade and p.hires_fix,
            priority=89,
        ),
        PipelineSpec(
            template=_template("stable_cascade"),
            predicate=lambda p, c: c.is_cascade,
            priority=88,
        ),
        PipelineSpec(
            template=_template("flux"),
            predicate=lambda p, c: c.is_flux,
            priority=87,
        ),
        PipelineSpec(
            template=_template("qwen"),
            predicate=lambda p, c: c.is_qwen,
            priority=86,
        ),
        # ControlNet
        PipelineSpec(
            template=_template("controlnet_annotator"),
            predicate=lambda p, c: bool(p.control_type) and p.return_control_map,
            priority=80,
        ),
        PipelineSpec(
            template=_template("controlnet_hires_fix"),
            predicate=lambda p, c: bool(p.control_type) and p.hires_fix,
            priority=79,
        ),
        PipelineSpec(
            template=_template("controlnet"),
            predicate=lambda p, c: bool(p.control_type),
            priority=78,
        ),
        # Masked / painting modes
        PipelineSpec(
            template=_template("stable_diffusion_img2img_mask"),
            predicate=lambda p, c: _has_img2img_mask(p),
            priority=70,
        ),
        PipelineSpec(
            template=_template("stable_diffusion_paint"),
            predicate=lambda p, c: p.source_processing in ("inpainting", "outpainting"),
            priority=69,
        ),
        # Generic stable diffusion
        PipelineSpec(
            template=_template("stable_diffusion_hires_fix"),
            predicate=lambda p, c: p.hires_fix,
            priority=10,
        ),
        PipelineSpec(
            template=_template("stable_diffusion"),
            predicate=lambda p, c: True,
            priority=0,
        ),
    ]

    for spec in specs:
        registry.register(spec)

    return registry


DEFAULT_REGISTRY = build_default_registry()
