"""The image family's parameter bindings, as named, composable groups.

Each group binds one coherent piece of the graph vocabulary (the sampler node, the prompts,
the controlnet nodes, ...). A pipeline composes exactly the groups its graph provides via
:func:`compose`; registration then audits every target against the graph, so a group included
by mistake fails loudly instead of silently skipping.

Group membership is pinned by ``tests/pipeline/materialized_expected/``; changing what a
group binds changes the submitted ComfyUI prompts and must be done deliberately (see the
snapshot harness docstring).
"""

from hordelib.pipeline.constants import SAMPLERS_MAP
from hordelib.pipeline.definition import ParamBinding
from hordelib.pipeline.payload import ImageGenPayload

__all__ = [
    "BATCH_REPEAT",
    "CASCADE_EMPTY_LATENT",
    "CASCADE_REMIX_SOURCE_IMAGE",
    "CASCADE_SAMPLERS",
    "CASCADE_SOURCE_IMAGE",
    "CASCADE_SECOND_PASS",
    "CLIP_SKIP",
    "CONTROLNET",
    "EMPTY_LATENT",
    "FLUX_SAMPLING",
    "HIRES_FIX_UPSCALE",
    "PROMPTS",
    "QR_LAYERS",
    "SAMPLER_CORE",
    "SEAMLESS_TILING",
    "SOURCE_IMAGE",
    "compose",
]

type ImageGenBinding = ParamBinding[ImageGenPayload]
type BindingGroup = tuple[ImageGenBinding, ...]


def compose(*groups: BindingGroup) -> BindingGroup:
    """Flatten binding groups into one binding set, rejecting duplicate targets.

    Args:
        groups: The named groups (or ad-hoc binding tuples) a pipeline exposes.

    Returns:
        tuple[ParamBinding, ...]: The flattened bindings, in group order.

    Raises:
        ValueError: If two groups bind the same target.
    """
    seen_targets: set[str] = set()
    flattened: list[ImageGenBinding] = []
    for group in groups:
        for binding in group:
            if binding.target in seen_targets:
                raise ValueError(f"Duplicate binding target {binding.target!r} across composed groups")
            seen_targets.add(binding.target)
            flattened.append(binding)
    return tuple(flattened)


def comfy_sampler(payload: ImageGenPayload) -> str | None:
    """Translate the horde sampler name to ComfyUI's."""
    return SAMPLERS_MAP.get(payload.sampler_name)


def comfy_clip_skip(payload: ImageGenPayload) -> int:
    """Translate clip skip to ComfyUI's convention (counted negatively: -1, -2, ...)."""
    return -payload.clip_skip if payload.clip_skip > 0 else payload.clip_skip


SAMPLER_CORE: BindingGroup = (
    ParamBinding(target="sampler.sampler_name", transform=comfy_sampler),
    ParamBinding(target="sampler.cfg", source="cfg_scale"),
    ParamBinding(target="sampler.denoise", source="denoising_strength"),
    ParamBinding(target="sampler.seed", source="seed"),
    # noise_seed is spurious on plain KSampler nodes but pinned by the snapshot corpus;
    # pruning it changes prompt hashes and is deferred to a deliberate, GPU-verified pass.
    ParamBinding(target="sampler.noise_seed", source="seed"),
    ParamBinding(target="sampler.scheduler", source="scheduler"),
    ParamBinding(target="sampler.steps", source="ddim_steps"),
)
"""The standard KSampler node titled ``sampler``."""

EMPTY_LATENT: BindingGroup = (
    ParamBinding(target="empty_latent_image.height", source="height"),
    ParamBinding(target="empty_latent_image.width", source="width"),
    ParamBinding(target="empty_latent_image.batch_size", source="n_iter"),
)
"""The txt2img starting latent (``empty_latent_image``)."""

BATCH_REPEAT: BindingGroup = (ParamBinding(target="repeat_image_batch.amount", source="n_iter"),)
"""Graphs that batch via a RepeatImageBatch node instead of (or besides) the latent batch."""

CLIP_SKIP: BindingGroup = (ParamBinding(target="clip_skip.stop_at_clip_layer", transform=comfy_clip_skip),)

PROMPTS: BindingGroup = (
    ParamBinding(target="prompt.text", source="prompt"),
    ParamBinding(target="negative_prompt.text", source="negative_prompt"),
)

SEAMLESS_TILING: BindingGroup = (ParamBinding(target="model_loader.seamless_tiling_enabled", source="tiling"),)

SOURCE_IMAGE: BindingGroup = (ParamBinding(target="image_loader.image", source="source_image"),)
"""The img2img source image loader (rewired into the sampler by the img2img patch step)."""

HIRES_FIX_UPSCALE: BindingGroup = (
    ParamBinding(target="upscale_sampler.denoise", source="hires_fix_denoising_strength"),
    ParamBinding(target="upscale_sampler.seed", source="seed"),
    ParamBinding(target="upscale_sampler.cfg", source="cfg_scale"),
    ParamBinding(target="upscale_sampler.sampler_name", transform=comfy_sampler),
)
"""The second-pass sampler of hires-fix graphs (its steps are computed by a patch step)."""

CONTROLNET: BindingGroup = (
    ParamBinding(target="controlnet_apply.strength", source="control_strength"),
    ParamBinding(target="controlnet_model_loader.control_net_name", source="control_type"),
)
"""The controlnet apply/loader pair (the model name is refined by the controlnet patch step)."""

FLUX_SAMPLING: BindingGroup = (
    ParamBinding(target="cfg_guider.cfg", source="cfg_scale"),
    ParamBinding(target="random_noise.noise_seed", source="seed"),
    ParamBinding(target="k_sampler_select.sampler_name", transform=comfy_sampler),
    ParamBinding(target="basic_scheduler.denoise", source="denoising_strength"),
    ParamBinding(target="basic_scheduler.steps", source="ddim_steps"),
)
"""Flux's split sampling nodes (SamplerCustomAdvanced constellation)."""

CASCADE_EMPTY_LATENT: BindingGroup = (
    ParamBinding(target="stable_cascade_empty_latent_image.width", source="width"),
    ParamBinding(target="stable_cascade_empty_latent_image.height", source="height"),
    ParamBinding(target="stable_cascade_empty_latent_image.batch_size", source="n_iter"),
)

CASCADE_REMIX_SOURCE_IMAGE: BindingGroup = (ParamBinding(target="sc_image_loader_0.image", source="source_image"),)
"""The remix graph's primary source image (extras are chained in by the remix patch step)."""

CASCADE_SOURCE_IMAGE: BindingGroup = (ParamBinding(target="sc_image_loader.image", source="source_image"),)

CASCADE_SAMPLERS: BindingGroup = (
    ParamBinding(target="sampler_stage_c.sampler_name", transform=comfy_sampler),
    ParamBinding(target="sampler_stage_b.sampler_name", transform=comfy_sampler),
    ParamBinding(target="sampler_stage_c.cfg", source="cfg_scale"),
    ParamBinding(target="sampler_stage_c.denoise", source="denoising_strength"),
    ParamBinding(target="sampler_stage_b.seed", source="seed"),
    ParamBinding(target="sampler_stage_c.seed", source="seed"),
    ParamBinding(target="sampler_stage_b.steps", source="ddim_steps", multiplier=0.33),
    ParamBinding(target="sampler_stage_c.steps", source="ddim_steps", multiplier=0.67),
)
"""Stable Cascade's two-stage samplers, with the legacy 0.67/0.33 step split."""

CASCADE_SECOND_PASS: BindingGroup = (
    ParamBinding(target="2pass_sampler_stage_c.sampler_name", transform=comfy_sampler),
    ParamBinding(target="2pass_sampler_stage_c.steps", source="ddim_steps", multiplier=0.67),
    ParamBinding(target="2pass_sampler_stage_c.denoise", source="hires_fix_denoising_strength"),
    ParamBinding(target="2pass_sampler_stage_b.sampler_name", transform=comfy_sampler),
    ParamBinding(target="2pass_sampler_stage_b.steps", source="ddim_steps", multiplier=0.33),
)

QR_LAYERS: BindingGroup = (
    ParamBinding(target="sampler_bg.sampler_name", transform=comfy_sampler),
    ParamBinding(target="sampler_bg.cfg", source="cfg_scale"),
    ParamBinding(target="sampler_bg.denoise", source="denoising_strength"),
    ParamBinding(target="sampler_bg.seed", source="seed"),
    ParamBinding(target="sampler_bg.steps", source="ddim_steps"),
    ParamBinding(target="sampler_bg.noise_seed", source="seed"),
    ParamBinding(target="sampler_fg.sampler_name", transform=comfy_sampler),
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
"""The qr_code workflow's background/foreground samplers and mask layers."""
