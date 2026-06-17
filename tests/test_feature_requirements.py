"""Unit tests for the optional-feature capability registry (``hordelib.feature_requirements``).

These assert that ``feature_available`` / ``available_features`` reflect what is importable in the
environment, so a consumer (the worker) can decide what to advertise before a job arrives. The
``find_spec`` probe is monkeypatched so the tests need none of the optional packages installed.
"""

import importlib.util

import pytest

from hordelib.feature_impact import FEATURE_KIND
from hordelib.feature_requirements import (
    available_features,
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


def test_controlnet_gates_on_onnxruntime_only() -> None:
    # mediapipe ships in the extra for node parity but is not a horde control_type dependency.
    requirement = get_feature_requirement(FEATURE_KIND.controlnet)
    assert requirement is not None
    assert requirement.packages == ("onnxruntime",)
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
    assert feature_available(FEATURE_KIND.controlnet) is False
    assert missing_packages(FEATURE_KIND.strip_background) == ("rembg",)
    assert missing_packages(FEATURE_KIND.controlnet) == ("onnxruntime",)


def test_feature_available_when_package_present(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(importlib.util, "find_spec", _fake_find_spec(present={"rembg", "onnxruntime"}))

    assert feature_available(FEATURE_KIND.strip_background) is True
    assert feature_available(FEATURE_KIND.controlnet) is True
    assert missing_packages(FEATURE_KIND.strip_background) == ()


def test_available_features_excludes_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(importlib.util, "find_spec", _fake_find_spec(present={"rembg"}))

    available = available_features()
    assert FEATURE_KIND.strip_background in available
    assert FEATURE_KIND.controlnet not in available
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
