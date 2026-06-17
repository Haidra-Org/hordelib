"""Which optional features are actually installable in this environment.

Some hordelib features depend on heavy, platform-constrained native packages that are not
part of the lean base install (they have no wheels for some accelerators, e.g. Intel XPU /
Apple MPS / Ascend, so forcing them into base would block a base install there). Those
packages live in ``horde-engine`` feature extras (see ``[project.optional-dependencies]``)
and the feature's call site guards its import via :mod:`hordelib.utils.optional_deps`.

This module is the single, typed source of truth mapping each such feature to the extra and
the import-names that back it, plus a runtime probe (:func:`feature_available`) so a consumer
(notably the worker) can decide what to advertise *before* a job arrives rather than failing
mid-job. It is the one place to edit when ComfyUI's (or a node's) backend support changes.

Features with no entry here have no optional dependency and are always considered available
(core inference, the safety classifier, ESRGAN upscalers and the CodeFormer/GFPGAN face
fixers are all pure PyTorch and run on every backend). Detection only: this module never
installs anything; extras are resolved at lock/sync time by the packaging tool.

This module is comfy-free and import-safe before ``hordelib.initialise()``; it is part of the
public API surface re-exported from :mod:`hordelib.api`.
"""

import importlib.util

from pydantic import BaseModel

from hordelib.feature_impact import FEATURE_KIND

__all__ = [
    "FeatureRequirement",
    "available_features",
    "feature_available",
    "get_feature_requirement",
    "get_feature_requirement_registry",
    "missing_packages",
]


class FeatureRequirement(BaseModel):
    """The optional packages one feature needs, and the extra that provides them."""

    feature: FEATURE_KIND
    packages: tuple[str, ...]
    """Top-level import names that must ALL be importable for the feature to be available."""
    extra: str
    """The ``horde-engine`` extra that installs ``packages`` (e.g. ``rembg`` -> ``[rembg]``)."""
    label: str
    """Human-readable feature name, used in "install this extra" messages."""


# Only the backend-constrained features appear here. controlnet gates on onnxruntime alone: of the
# preprocessors hordelib exposes as horde control_types (CONTROLNET_IMAGE_PREPROCESSOR_MAP), only
# Openpose (DWPose) needs a blocker dep, and that dep is onnxruntime. mediapipe backs only
# mediapipe_face / mesh_graphormer, which are not horde control_types, so it is shipped in the
# `controlnet` extra for node parity but is not required to serve controlnet jobs.
_REQUIREMENT_SEEDS: list[FeatureRequirement] = [
    FeatureRequirement(
        feature=FEATURE_KIND.strip_background,
        packages=("rembg",),
        extra="rembg",
        label="strip_background (background removal)",
    ),
    FeatureRequirement(
        feature=FEATURE_KIND.controlnet,
        packages=("onnxruntime",),
        extra="controlnet",
        label="controlnet preprocessing (annotators)",
    ),
]

_REGISTRY: dict[FEATURE_KIND, FeatureRequirement] = {req.feature: req for req in _REQUIREMENT_SEEDS}


def _is_importable(package: str) -> bool:
    """Return whether *package* can be found in the environment, never raising.

    ``find_spec`` can raise (e.g. ``ModuleNotFoundError`` for a missing parent, or ``ValueError``)
    rather than returning ``None``; for a capability probe any failure means "not available".
    """
    try:
        return importlib.util.find_spec(package) is not None
    except (ImportError, ValueError):
        return False


def get_feature_requirement_registry() -> dict[FEATURE_KIND, FeatureRequirement]:
    """Return the mapping of features to their optional-dependency requirements."""
    return _REGISTRY


def get_feature_requirement(feature: FEATURE_KIND) -> FeatureRequirement | None:
    """Return the requirement for *feature*, or None when it has no optional dependency."""
    return _REGISTRY.get(feature)


def missing_packages(feature: FEATURE_KIND) -> tuple[str, ...]:
    """Return the requirement's packages that are not importable (empty when all present)."""
    requirement = _REGISTRY.get(feature)
    if requirement is None:
        return ()
    return tuple(package for package in requirement.packages if not _is_importable(package))


def feature_available(feature: FEATURE_KIND) -> bool:
    """Return whether *feature* can run here.

    A feature with no requirement entry is always available; otherwise every package its
    requirement names must be importable.
    """
    return not missing_packages(feature)


def available_features() -> set[FEATURE_KIND]:
    """Return the set of features runnable in this environment (always-available ones included)."""
    return {feature for feature in FEATURE_KIND if feature_available(feature)}
