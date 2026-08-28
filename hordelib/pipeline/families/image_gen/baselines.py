"""Per-baseline knowledge the image family's selection, patch steps, and loading policy share.

Baseline knowledge for the image family lives here (and in the per-pipeline selectors), not
scattered through patch steps or the execution layer: each baseline with non-default behavior
gets a :class:`BaselineProfile` row declaring how its weights load, which comfy ``model_base``
classes must never be force-loaded onto the GPU, the CLIP type its text encoder is loaded with,
and the flow-matching shift its sampling takes. A single model that deviates from its baseline's
row can get an :class:`ModelOverride` in :data:`IMAGE_MODEL_OVERRIDES` rather than a graph of its own.

``execution/comfy_patches.py`` consumes the profiles lazily (it must stay importable without
horde_model_reference), and its startup tripwire cross-checks the profile class names against
its horde_model_reference-free flat skip list and against the live ``comfy.model_base``.
"""

from dataclasses import dataclass
from enum import StrEnum, auto
from types import MappingProxyType

from horde_model_reference.meta_consts import KNOWN_IMAGE_GENERATION_BASELINE

from hordelib.pipeline.patches import FlowShiftNode

__all__ = [
    "ALIGN_YOUR_STEPS_MODEL_TYPES",
    "CASCADE_BASELINES",
    "FLUX_BASELINES",
    "IMAGE_BASELINE_PROFILES",
    "IMAGE_MODEL_OVERRIDES",
    "AlignYourStepsModelType",
    "BaselineProfile",
    "FlowShiftNode",
    "LoaderKind",
    "ModelOverride",
    "QWEN_GRAPH_BASELINES",
    "UNET_LOADER_BASELINES",
    "Z_IMAGE_BASELINES",
    "align_your_steps_model_type",
    "resolve_clip_type",
    "resolve_flow_shift",
]


class LoaderKind(StrEnum):
    """How a baseline's diffusion weights are loaded."""

    CHECKPOINT = auto()
    """A fused checkpoint through the standard checkpoint loader."""
    UNET = auto()
    """Split files: the bare diffusion model (file_type "unet"), with CLIP/VAE wired from
    their own loader nodes."""


@dataclass(frozen=True)
class BaselineProfile:
    """The image family's per-baseline loading knowledge."""

    baseline: KNOWN_IMAGE_GENERATION_BASELINE
    loader: LoaderKind = LoaderKind.CHECKPOINT
    force_load_skip_classes: tuple[str, ...] = ()
    """comfy ``model_base`` class names of this baseline that must never be force-loaded
    (large models whose forced full GPU load would OOM/segfault on smaller cards)."""
    clip_type: str | None = None
    """The comfy ``CLIPType`` a split-file text encoder is loaded with; None for a baseline whose
    text encoder comes out of the checkpoint."""
    flow_shift_node: FlowShiftNode | None = None
    """The shift node this baseline's graph takes; None for a baseline whose sampling has no shift."""
    default_flow_shift: float | None = None
    """The shift applied when the payload requests none; None leaves the model unshifted."""


@dataclass(frozen=True)
class ModelOverride:
    """One model's deviations from its baseline's profile.

    A family member can ship its own components (a different text encoder, hence a different CLIP
    type) or be distilled so that the family's shift no longer applies to it, without being a
    baseline of its own.
    """

    clip_type: str | None = None
    """Overrides the baseline's CLIP type; None keeps it."""
    applies_flow_shift: bool = True
    """False when this model takes no shift node even though its baseline does."""


CASCADE_BASELINES: frozenset[KNOWN_IMAGE_GENERATION_BASELINE] = frozenset(
    {KNOWN_IMAGE_GENERATION_BASELINE.stable_cascade},
)
"""Stable Cascade (a single member; selectors take baseline sets uniformly)."""

QWEN_GRAPH_BASELINES: frozenset[KNOWN_IMAGE_GENERATION_BASELINE] = frozenset(
    {
        KNOWN_IMAGE_GENERATION_BASELINE.qwen_image,
        KNOWN_IMAGE_GENERATION_BASELINE.krea2_turbo,
        KNOWN_IMAGE_GENERATION_BASELINE.anima,
    },
)
"""Architecturally distinct split-file baselines that share the same standard sampler graph shape."""

Z_IMAGE_BASELINES: frozenset[KNOWN_IMAGE_GENERATION_BASELINE] = frozenset(
    {KNOWN_IMAGE_GENERATION_BASELINE.z_image_turbo},
)

FLUX_BASELINES: frozenset[KNOWN_IMAGE_GENERATION_BASELINE] = frozenset(
    {
        KNOWN_IMAGE_GENERATION_BASELINE.flux_1,
        KNOWN_IMAGE_GENERATION_BASELINE.flux_schnell,
        KNOWN_IMAGE_GENERATION_BASELINE.flux_dev,
    },
)
"""The flux family spans several baseline enum members; selection and the patch steps treat
them uniformly, so they are grouped here rather than compared one by one."""

IMAGE_BASELINE_PROFILES: MappingProxyType[KNOWN_IMAGE_GENERATION_BASELINE, BaselineProfile] = MappingProxyType(
    {
        profile.baseline: profile
        for profile in (
            BaselineProfile(
                baseline=KNOWN_IMAGE_GENERATION_BASELINE.stable_cascade,
                force_load_skip_classes=("StableCascade_C", "StableCascade_B"),
            ),
            BaselineProfile(
                baseline=KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_xl,
                force_load_skip_classes=("SDXL", "SDXLRefiner"),
            ),
            BaselineProfile(
                baseline=KNOWN_IMAGE_GENERATION_BASELINE.flux_1,
                force_load_skip_classes=("Flux",),
                flow_shift_node=FlowShiftNode.FLUX,
            ),
            BaselineProfile(
                baseline=KNOWN_IMAGE_GENERATION_BASELINE.flux_schnell,
                force_load_skip_classes=("Flux",),
                flow_shift_node=FlowShiftNode.FLUX,
            ),
            BaselineProfile(
                baseline=KNOWN_IMAGE_GENERATION_BASELINE.flux_dev,
                force_load_skip_classes=("Flux",),
                flow_shift_node=FlowShiftNode.FLUX,
            ),
            BaselineProfile(
                baseline=KNOWN_IMAGE_GENERATION_BASELINE.qwen_image,
                loader=LoaderKind.UNET,
                force_load_skip_classes=("QwenImage",),
                clip_type="qwen_image",
                flow_shift_node=FlowShiftNode.AURA_FLOW,
                # The shift Qwen-Image is served with; carried as a float artifact of the workflow
                # the graph was exported from, kept exactly so the submitted prompt is unchanged.
                default_flow_shift=3.1000000000000005,
            ),
            BaselineProfile(
                baseline=KNOWN_IMAGE_GENERATION_BASELINE.krea2_turbo,
                loader=LoaderKind.UNET,
                force_load_skip_classes=("Krea2",),
                clip_type="krea2",
            ),
            BaselineProfile(
                baseline=KNOWN_IMAGE_GENERATION_BASELINE.anima,
                loader=LoaderKind.UNET,
                force_load_skip_classes=("Anima",),
                # ComfyUI detects Anima's Qwen3-0.6B wrapper from the encoder weights through
                # its default CLIP loader path. The model config supplies Anima's native shift.
                clip_type="stable_diffusion",
            ),
            BaselineProfile(
                baseline=KNOWN_IMAGE_GENERATION_BASELINE.z_image_turbo,
                loader=LoaderKind.UNET,
                # Z-Image (incl. Z-Image-Turbo) loads as comfy's Lumina2 model_base class.
                force_load_skip_classes=("Lumina2",),
                clip_type="lumina2",
            ),
        )
    },
)
"""Baselines with non-default loading behavior; absent baselines use checkpoint loading and
have no force-load policy."""

IMAGE_MODEL_OVERRIDES: MappingProxyType[str, ModelOverride] = MappingProxyType(
    {},
)
"""Per-model deviations from the baseline profiles, keyed by horde model name."""


def _profile_for(baseline: KNOWN_IMAGE_GENERATION_BASELINE | str | None) -> BaselineProfile | None:
    """Return the profile for *baseline*, accepting the raw reference spelling, else None."""
    if baseline is None:
        return None
    try:
        return IMAGE_BASELINE_PROFILES.get(KNOWN_IMAGE_GENERATION_BASELINE(baseline))
    except ValueError:
        return None


def resolve_clip_type(
    baseline: KNOWN_IMAGE_GENERATION_BASELINE | str | None,
    model_name: str | None = None,
) -> str | None:
    """Return the comfy ``CLIPType`` name a model's split-file text encoder must be loaded with.

    The same answer serves the graph's ``clip_loader`` and the component lane's standalone load of that
    encoder: loading one encoder under two types produces two different modules, and no consumer would
    adopt the other's. None means the baseline has no split-file encoder to type.
    """
    override = IMAGE_MODEL_OVERRIDES.get(model_name) if model_name is not None else None
    if override is not None and override.clip_type is not None:
        return override.clip_type
    profile = _profile_for(baseline)
    return profile.clip_type if profile is not None else None


def resolve_flow_shift(
    baseline: KNOWN_IMAGE_GENERATION_BASELINE | str | None,
    model_name: str | None,
    requested: float | None,
) -> tuple[FlowShiftNode | None, float | None]:
    """Return the shift node this model's graph takes and the shift to apply to it.

    A None node means the model's sampling has no shift, whatever was requested. The baseline's own
    default stands in when the payload asks for nothing, so a family's graph is shifted the way the
    model was trained without every caller having to know the number.
    """
    override = IMAGE_MODEL_OVERRIDES.get(model_name) if model_name is not None else None
    if override is not None and not override.applies_flow_shift:
        return None, requested
    profile = _profile_for(baseline)
    if profile is None:
        return None, requested
    return profile.flow_shift_node, requested if requested is not None else profile.default_flow_shift


UNET_LOADER_BASELINES: frozenset[KNOWN_IMAGE_GENERATION_BASELINE] = frozenset(
    profile.baseline for profile in IMAGE_BASELINE_PROFILES.values() if profile.loader is LoaderKind.UNET
)
"""Split-files baselines, derived from :data:`IMAGE_BASELINE_PROFILES`."""


class AlignYourStepsModelType(StrEnum):
    """A model family NVIDIA published Align Your Steps noise levels for.

    The values are the keys of ``comfy_extras.nodes_align_your_steps.NOISE_LEVELS`` and are used to
    index it directly, so they carry upstream's spelling rather than this package's.
    """

    SD1 = "SD1"
    SDXL = "SDXL"
    SVD = "SVD"


ALIGN_YOUR_STEPS_MODEL_TYPES: MappingProxyType[KNOWN_IMAGE_GENERATION_BASELINE, AlignYourStepsModelType] = (
    MappingProxyType(
        {
            KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_1: AlignYourStepsModelType.SD1,
            KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_xl: AlignYourStepsModelType.SDXL,
        },
    )
)
"""Baselines an Align Your Steps schedule can be built for.

The noise levels are measured per family rather than derived, so a baseline without published levels
has no substitute here: `SVD` is video and unreachable from an image request, and the SD2 baselines
were never measured. Running one family's levels on another is not a schedule for that model, so an
unmapped baseline is refused rather than approximated (see
:func:`hordelib.pipeline.horde_compat.resolve_sigma_schedule`).
"""


def align_your_steps_model_type(
    baseline: KNOWN_IMAGE_GENERATION_BASELINE | None,
) -> AlignYourStepsModelType | None:
    """Return the Align Your Steps family for *baseline*, or None when it has no published levels."""
    if baseline is None:
        return None
    return ALIGN_YOUR_STEPS_MODEL_TYPES.get(baseline)
