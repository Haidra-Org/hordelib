"""Per-baseline knowledge the image family's selection, patch steps, and loading policy share.

Baseline knowledge for the image family lives here (and in the per-pipeline selectors), not
scattered through patch steps or the execution layer: each baseline with non-default behavior
gets a :class:`BaselineProfile` row declaring how its weights load and which comfy
``model_base`` classes must never be force-loaded onto the GPU.

``execution/comfy_patches.py`` consumes the profiles lazily (it must stay importable without
horde_model_reference), and its startup tripwire cross-checks the profile class names against
its horde_model_reference-free flat skip list and against the live ``comfy.model_base``.
"""

from dataclasses import dataclass
from enum import StrEnum, auto
from types import MappingProxyType

from horde_model_reference.meta_consts import KNOWN_IMAGE_GENERATION_BASELINE

__all__ = [
    "CASCADE_BASELINES",
    "FLUX_BASELINES",
    "IMAGE_BASELINE_PROFILES",
    "BaselineProfile",
    "LoaderKind",
    "QWEN_BASELINES",
    "UNET_LOADER_BASELINES",
    "Z_IMAGE_BASELINES",
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


CASCADE_BASELINES: frozenset[KNOWN_IMAGE_GENERATION_BASELINE] = frozenset(
    {KNOWN_IMAGE_GENERATION_BASELINE.stable_cascade},
)
"""Stable Cascade (a single member; selectors take baseline sets uniformly)."""

QWEN_BASELINES: frozenset[KNOWN_IMAGE_GENERATION_BASELINE] = frozenset(
    {KNOWN_IMAGE_GENERATION_BASELINE.qwen_image},
)

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
            ),
            BaselineProfile(
                baseline=KNOWN_IMAGE_GENERATION_BASELINE.flux_schnell,
                force_load_skip_classes=("Flux",),
            ),
            BaselineProfile(
                baseline=KNOWN_IMAGE_GENERATION_BASELINE.flux_dev,
                force_load_skip_classes=("Flux",),
            ),
            BaselineProfile(
                baseline=KNOWN_IMAGE_GENERATION_BASELINE.qwen_image,
                loader=LoaderKind.UNET,
                force_load_skip_classes=("QwenImage",),
            ),
            BaselineProfile(
                baseline=KNOWN_IMAGE_GENERATION_BASELINE.z_image_turbo,
                loader=LoaderKind.UNET,
                # Z-Image (incl. Z-Image-Turbo) loads as comfy's Lumina2 model_base class.
                force_load_skip_classes=("Lumina2",),
            ),
        )
    },
)
"""Baselines with non-default loading behavior; absent baselines use checkpoint loading and
have no force-load policy."""

UNET_LOADER_BASELINES: frozenset[KNOWN_IMAGE_GENERATION_BASELINE] = frozenset(
    profile.baseline for profile in IMAGE_BASELINE_PROFILES.values() if profile.loader is LoaderKind.UNET
)
"""Split-files baselines, derived from :data:`IMAGE_BASELINE_PROFILES`."""
