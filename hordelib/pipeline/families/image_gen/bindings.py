"""The image family's parameter bindings, as named, composable groups.

Each group binds one coherent piece of the graph vocabulary (the sampler node, the prompts,
the controlnet nodes, ...), authored node-first via :func:`hordelib.pipeline.definition.node`
handles: the handle names the graph node and its expected class, and ``bind`` keywords are the
node's actual input names. A pipeline composes exactly the groups its graph provides via
:func:`compose`; registration then audits every node title, node class, input name (against
the committed ComfyUI node schema snapshot), and payload source, so a typo in any of the four
fails loudly instead of silently skipping.

Group membership is pinned by ``tests/pipeline/materialized_expected/``; changing what a
group binds changes the submitted ComfyUI prompts and must be done deliberately (see the
snapshot harness docstring).
"""

from hordelib.pipeline.constants import SAMPLERS_MAP
from hordelib.pipeline.definition import ParamBinding, node, scaled
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
    "QR_SAMPLER_CORE",
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


SAMPLER_CORE: BindingGroup = node("sampler", "KSampler").bind(
    sampler_name=comfy_sampler,
    cfg="cfg_scale",
    denoise="denoising_strength",
    seed="seed",
    # noise_seed is spurious on KSampler (grandfathered via known_spurious_inputs) but pinned
    # by the snapshot corpus; pruning it changes prompt hashes and is deferred to a
    # deliberate, GPU-verified pass.
    noise_seed="seed",
    scheduler="scheduler",
    steps="ddim_steps",
)
"""The standard KSampler node titled ``sampler``."""

QR_SAMPLER_CORE: BindingGroup = node("sampler", "KSamplerAdvanced").bind(
    sampler_name=comfy_sampler,
    cfg="cfg_scale",
    # KSamplerAdvanced has no denoise/seed inputs (it takes noise_seed); both are
    # grandfathered no-ops pinned by the snapshot corpus, like SAMPLER_CORE's noise_seed.
    denoise="denoising_strength",
    seed="seed",
    noise_seed="seed",
    scheduler="scheduler",
    steps="ddim_steps",
)
"""The qr_code graph's main sampler: same targets as :data:`SAMPLER_CORE`, but the node is a
KSamplerAdvanced there."""

EMPTY_LATENT: BindingGroup = node(
    "empty_latent_image",
    "EmptyLatentImage",
    "EmptySD3LatentImage",
    "StableCascade_EmptyLatentImage",
).bind(
    height="height",
    width="width",
    batch_size="n_iter",
)
"""The txt2img starting latent; the empty-latent node class varies by graph family but all
of them declare the same three inputs."""

BATCH_REPEAT: BindingGroup = node("repeat_image_batch", "RepeatImageBatch").bind(amount="n_iter")
"""Graphs that batch via a RepeatImageBatch node instead of (or besides) the latent batch."""

CLIP_SKIP: BindingGroup = node("clip_skip", "CLIPSetLastLayer").bind(stop_at_clip_layer=comfy_clip_skip)

PROMPTS: BindingGroup = (
    *node("prompt", "CLIPTextEncode").bind(text="prompt"),
    *node("negative_prompt", "CLIPTextEncode").bind(text="negative_prompt"),
)

SEAMLESS_TILING: BindingGroup = node("model_loader", "HordeCheckpointLoader").bind(seamless_tiling_enabled="tiling")

SOURCE_IMAGE: BindingGroup = node("image_loader", "HordeImageLoader").bind(image="source_image")
"""The img2img source image loader (rewired into the sampler by the img2img patch step)."""

HIRES_FIX_UPSCALE: BindingGroup = node("upscale_sampler", "KSampler").bind(
    denoise="hires_fix_denoising_strength",
    seed="seed",
    cfg="cfg_scale",
    sampler_name=comfy_sampler,
)
"""The second-pass sampler of hires-fix graphs (its steps are computed by a patch step)."""

CONTROLNET: BindingGroup = (
    *node("controlnet_apply", "ControlNetApply").bind(strength="control_strength"),
    *node("controlnet_model_loader", "DiffControlNetLoader").bind(control_net_name="control_type"),
)
"""The controlnet apply/loader pair (the model name is refined by the controlnet patch step)."""

FLUX_SAMPLING: BindingGroup = (
    *node("cfg_guider", "CFGGuider").bind(cfg="cfg_scale"),
    *node("random_noise", "RandomNoise").bind(noise_seed="seed"),
    *node("k_sampler_select", "KSamplerSelect").bind(sampler_name=comfy_sampler),
    *node("basic_scheduler", "BasicScheduler").bind(denoise="denoising_strength", steps="ddim_steps"),
)
"""Flux's split sampling nodes (SamplerCustomAdvanced constellation)."""

CASCADE_EMPTY_LATENT: BindingGroup = node(
    "stable_cascade_empty_latent_image",
    "StableCascade_EmptyLatentImage",
).bind(
    width="width",
    height="height",
    batch_size="n_iter",
)

CASCADE_REMIX_SOURCE_IMAGE: BindingGroup = node("sc_image_loader_0", "HordeImageLoader").bind(image="source_image")
"""The remix graph's primary source image (extras are chained in by the remix patch step)."""

CASCADE_SOURCE_IMAGE: BindingGroup = node("sc_image_loader", "HordeImageLoader").bind(image="source_image")

CASCADE_SAMPLERS: BindingGroup = (
    *node("sampler_stage_c", "KSampler").bind(
        sampler_name=comfy_sampler,
        cfg="cfg_scale",
        denoise="denoising_strength",
        seed="seed",
        steps=scaled("ddim_steps", 0.67),
    ),
    *node("sampler_stage_b", "KSampler").bind(
        sampler_name=comfy_sampler,
        seed="seed",
        steps=scaled("ddim_steps", 0.33),
    ),
)
"""Stable Cascade's two-stage samplers, with the legacy 0.67/0.33 step split."""

CASCADE_SECOND_PASS: BindingGroup = (
    *node("2pass_sampler_stage_c", "KSampler").bind(
        sampler_name=comfy_sampler,
        steps=scaled("ddim_steps", 0.67),
        denoise="hires_fix_denoising_strength",
    ),
    *node("2pass_sampler_stage_b", "KSampler").bind(
        sampler_name=comfy_sampler,
        steps=scaled("ddim_steps", 0.33),
    ),
)

QR_LAYERS: BindingGroup = (
    *node("sampler_bg", "KSamplerAdvanced").bind(
        sampler_name=comfy_sampler,
        cfg="cfg_scale",
        # denoise/seed are not KSamplerAdvanced inputs; grandfathered no-ops (see above).
        denoise="denoising_strength",
        seed="seed",
        steps="ddim_steps",
        noise_seed="seed",
    ),
    *node("sampler_fg", "KSamplerAdvanced").bind(
        sampler_name=comfy_sampler,
        cfg="cfg_scale",
        denoise="denoising_strength",
        seed="seed",
        steps="ddim_steps",
        noise_seed="seed",
    ),
    *node("controlnet_bg", "ControlNetApplyAdvanced").bind(strength="control_strength"),
    *node("solidmask_grey", "SolidMask").bind(width="width", height="height"),
    *node("solidmask_white", "SolidMask").bind(width="width", height="height"),
    *node("solidmask_black", "SolidMask").bind(width="width", height="height"),
    *node("qr_code_split", "comfy-qr-by-module-split").bind(max_image_size="width"),
)
"""The qr_code workflow's background/foreground samplers and mask layers."""
