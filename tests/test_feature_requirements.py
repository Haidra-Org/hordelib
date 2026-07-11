"""Unit tests for the optional-feature capability registry (``hordelib.feature_requirements``).

These assert that ``feature_available`` / ``available_features`` reflect what is importable in the
environment, so a consumer (the worker) can decide what to advertise before a job arrives. The
``find_spec`` probe is monkeypatched so the tests need none of the optional packages installed.
"""

import importlib.util

import pytest

from hordelib.feature_impact import FEATURE_KIND
from hordelib.feature_requirements import (
    MissingFeatureDependencyError,
    available_features,
    ensure_feature_available,
    feature_available,
    get_feature_requirement,
    get_feature_requirement_registry,
    missing_packages,
)


def _fake_find_spec(present: set[str]):
    """Return a ``find_spec`` replacement that reports only *present* top-level packages as importable."""

    def _find_spec(name: str, package: str | None = None) -> object | None:
        # The registry only cares whether the result is None; a sentinel object is enough.
        return object() if name.split(".", 1)[0] in present else None

    return _find_spec


def test_strip_background_requires_rembg() -> None:
    requirement = get_feature_requirement(FEATURE_KIND.strip_background)
    assert requirement is not None
    assert requirement.packages == ("rembg",)
    assert requirement.extra == "rembg"


def test_controlnet_has_no_blocking_packages() -> None:
    # Every exposed horde control_type preprocessor is pure torch, so controlnet gates on no package.
    # The requirement is kept only so the `controlnet` extra (node-parity onnxruntime/mediapipe) stays
    # named in the registry the worker mirrors for provisioning.
    requirement = get_feature_requirement(FEATURE_KIND.controlnet)
    assert requirement is not None
    assert requirement.packages == ()
    assert requirement.extra == "controlnet"


def test_pure_torch_features_have_no_requirement_and_are_available() -> None:
    # Upscalers/face-fix/lora/img2img are pure torch: no optional dependency, always available.
    for feature in (
        FEATURE_KIND.post_processing_upscale,
        FEATURE_KIND.post_processing_facefix,
        FEATURE_KIND.lora,
        FEATURE_KIND.img2img,
    ):
        assert get_feature_requirement(feature) is None
        assert feature_available(feature) is True


def test_feature_unavailable_when_package_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(importlib.util, "find_spec", _fake_find_spec(present=set()))

    assert feature_available(FEATURE_KIND.strip_background) is False
    # controlnet has no blocking package, so it is available even with nothing importable.
    assert feature_available(FEATURE_KIND.controlnet) is True
    assert missing_packages(FEATURE_KIND.strip_background) == ("rembg",)
    assert missing_packages(FEATURE_KIND.controlnet) == ()


def test_feature_available_when_package_present(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(importlib.util, "find_spec", _fake_find_spec(present={"rembg"}))

    assert feature_available(FEATURE_KIND.strip_background) is True
    assert feature_available(FEATURE_KIND.controlnet) is True
    assert missing_packages(FEATURE_KIND.strip_background) == ()


def test_available_features_includes_unblocked_controlnet(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(importlib.util, "find_spec", _fake_find_spec(present=set()))

    available = available_features()
    # strip_background is gated on rembg and drops out; controlnet has no blocking package and stays.
    assert FEATURE_KIND.strip_background not in available
    assert FEATURE_KIND.controlnet in available
    # An always-available pure-torch feature is still present.
    assert FEATURE_KIND.post_processing_upscale in available


def test_probe_never_raises_on_bad_name(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raising_find_spec(name: str, package: str | None = None):
        raise ValueError("simulated bad module name")

    monkeypatch.setattr(importlib.util, "find_spec", _raising_find_spec)
    # A probe failure must read as "not available", not propagate.
    assert feature_available(FEATURE_KIND.strip_background) is False


def test_registry_is_subset_of_feature_kinds() -> None:
    registry = get_feature_requirement_registry()
    assert set(registry).issubset(set(FEATURE_KIND))
    assert all(req.feature == key for key, req in registry.items())


def test_missing_feature_dependency_error_is_importerror() -> None:
    # Consumers that already catch a missing optional dependency (ImportError) catch this too.
    assert issubclass(MissingFeatureDependencyError, ImportError)


def test_ensure_feature_available_raises_with_structured_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(importlib.util, "find_spec", _fake_find_spec(present=set()))

    with pytest.raises(MissingFeatureDependencyError) as exc_info:
        ensure_feature_available(FEATURE_KIND.strip_background)

    error = exc_info.value
    assert error.feature is FEATURE_KIND.strip_background
    assert error.missing_packages == ("rembg",)
    assert error.extra == "rembg"
    assert "horde-engine[rembg]" in str(error)


def test_ensure_feature_available_passes_when_present(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(importlib.util, "find_spec", _fake_find_spec(present={"rembg", "onnxruntime"}))

    ensure_feature_available(FEATURE_KIND.strip_background)
    ensure_feature_available(FEATURE_KIND.controlnet)


def test_ensure_feature_available_is_a_noop_for_requirementless_feature() -> None:
    # A pure-torch feature has no requirement entry and must never raise, regardless of the environment.
    ensure_feature_available(FEATURE_KIND.post_processing_upscale)
