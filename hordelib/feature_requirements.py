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

An in-graph call site that would otherwise degrade or fail deep inside execution when its extra is
absent calls :func:`ensure_feature_available` up front, which raises the typed
:class:`MissingFeatureDependencyError` before any graph runs. That error carries the failing feature,
the missing package names, and the extra that installs them, so a consumer can catch it and act on the
structured fields rather than parsing a message.

This module is comfy-free and import-safe before ``hordelib.initialise()``; it is part of the
public API surface re-exported from :mod:`hordelib.api`.
"""

import importlib.util

from pydantic import BaseModel

from hordelib.feature_impact import FEATURE_KIND
from hordelib.utils.optional_deps import _DISTRIBUTION_NAME

__all__ = [
    "FeatureRequirement",
    "MissingFeatureDependencyError",
    "available_features",
    "ensure_feature_available",
    "feature_available",
    "get_feature_requirement",
    "get_feature_requirement_registry",
    "missing_packages",
]


class MissingFeatureDependencyError(ImportError):
    """Raised when an optional feature is invoked but its backing packages are not installed.

    The typed fields let a consumer branch on exactly which feature failed and which extra would
    satisfy it, instead of parsing the message. It subclasses ``ImportError`` (like the lazy
    :class:`~hordelib.utils.optional_deps.MissingOptionalDependency`) so a caller that already handles
    a missing optional dependency catches this earlier, pre-run variant with the same ``except``.
    """

    def __init__(
        self,
        *,
        feature: FEATURE_KIND,
        missing_packages: tuple[str, ...],
        extra: str,
        label: str,
    ) -> None:
        """Build the error from the failing feature's requirement.

        Args:
            feature: The feature whose optional dependency is missing.
            missing_packages: The requirement's import names that are not importable.
            extra: The ``horde-engine`` extra that installs the missing packages.
            label: Human-readable feature name, used in the message.
        """
        self.feature = feature
        self.missing_packages = missing_packages
        self.extra = extra
        self.label = label
        joined = ", ".join(missing_packages)
        super().__init__(
            f"{label} requires the optional package(s) {joined}, which are not installed. "
            f"Install them with: pip install {_DISTRIBUTION_NAME}[{extra}]",
        )


class FeatureRequirement(BaseModel):
    """The optional packages one feature needs, and the extra that provides them."""

    feature: FEATURE_KIND
    packages: tuple[str, ...]
    """Top-level import names that must ALL be importable for the feature to be available."""
    extra: str
    """The ``horde-engine`` extra that installs ``packages`` (e.g. ``rembg`` -> ``[rembg]``)."""
    label: str
    """Human-readable feature name, used in "install this extra" messages."""


# Every preprocessor hordelib exposes as a horde control_type (CONTROLNET_IMAGE_PREPROCESSOR_MAP) is
# pure torch, so serving a controlnet job needs no backend-constrained package (its `packages` are
# empty and it is therefore always available). The onnxruntime-backed detectors (DWPose) and the
# mediapipe-backed nodes (mediapipe_face / mesh_graphormer) are not exposed as horde control_types;
# they ship in the `controlnet` extra for comfyui_controlnet_aux node parity only. The requirement is
# retained (rather than dropped) so the `controlnet` extra stays present in this registry, which the
# worker mirrors as the source of truth for its provisioned feature extras. onnxruntime enters the
# environment only as a transitive dependency of rembg, so `strip_background` below is the sole
# feature whose availability actually turns on it.
_REQUIREMENT_SEEDS: list[FeatureRequirement] = [
    FeatureRequirement(
        feature=FEATURE_KIND.strip_background,
        packages=("rembg",),
        extra="rembg",
        label="strip_background (background removal)",
    ),
    FeatureRequirement(
        feature=FEATURE_KIND.controlnet,
        packages=(),
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


def ensure_feature_available(feature: FEATURE_KIND) -> None:
    """Raise if *feature*'s optional dependency is absent, probing via ``find_spec``.

    A feature with no requirement entry, or one whose packages are all importable, returns without
    raising. This is the fail-fast guard an in-graph call site invokes before it starts work, so a
    missing extra surfaces as :class:`MissingFeatureDependencyError` up front rather than as an obscure
    failure mid-run. The probe only checks importability; it never imports the package.

    Args:
        feature: The feature to check.

    Raises:
        MissingFeatureDependencyError: If any package the feature requires is not importable.
    """
    requirement = _REGISTRY.get(feature)
    if requirement is None:
        return
    missing = tuple(package for package in requirement.packages if not _is_importable(package))
    if missing:
        raise MissingFeatureDependencyError(
            feature=feature,
            missing_packages=missing,
            extra=requirement.extra,
            label=requirement.label,
        )
