"""Fail-fast guard tests for the strip_background post-processor. No GPU required.

``HordeLib.post_process`` is exercised as an unbound method with a dummy ``self`` because the
``StripBackgroundPayload`` path uses no instance state and no ComfyUI backend, so the guard can be
tested without constructing the (GPU-bound) backend. The ``find_spec`` probe is monkeypatched so the
tests need neither rembg nor onnxruntime installed.
"""

import importlib.util

import PIL.Image
import pytest

from hordelib.feature_impact import FEATURE_KIND
from hordelib.feature_requirements import MissingFeatureDependencyError
from hordelib.horde import HordeLib
from hordelib.pipeline.payload_pp import StripBackgroundPayload
from hordelib.utils.image_utils import ImageUtils


def _fake_find_spec(present: set[str]):
    """Return a ``find_spec`` replacement reporting only *present* top-level packages as importable."""

    def _find_spec(name: str, package: str | None = None) -> object | None:
        return object() if name.split(".", 1)[0] in present else None

    return _find_spec


def test_post_process_strip_background_raises_when_rembg_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(importlib.util, "find_spec", _fake_find_spec(present=set()))
    payload = StripBackgroundPayload(source_image=PIL.Image.new("RGB", (8, 8)))

    with pytest.raises(MissingFeatureDependencyError) as exc_info:
        HordeLib.post_process(object(), payload)

    error = exc_info.value
    assert error.feature is FEATURE_KIND.strip_background
    assert error.missing_packages == ("rembg",)
    assert error.extra == "rembg"


def test_post_process_strip_background_runs_when_rembg_present(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(importlib.util, "find_spec", _fake_find_spec(present={"rembg"}))

    source = PIL.Image.new("RGB", (8, 8))
    stripped = PIL.Image.new("RGBA", (8, 8))
    seen: dict[str, PIL.Image.Image] = {}

    def _fake_strip(image: PIL.Image.Image) -> PIL.Image.Image:
        seen["image"] = image
        return stripped

    # Mock the actual rembg-backed removal; the guard must let the call through when rembg is present.
    monkeypatch.setattr(ImageUtils, "strip_background", _fake_strip)

    result = HordeLib.post_process(object(), StripBackgroundPayload(source_image=source))

    assert result.image is stripped
    assert seen["image"] is source
