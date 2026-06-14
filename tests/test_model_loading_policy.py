"""GPU-free unit tests for the force-load skip policy and multi-file component placement.

These guard the two robustness fixes that let large multi-file models (e.g. Qwen-Image) load:
  * the force-load skip list is keyed on comfy *class names* (resolved by identity at runtime),
    not a fragile lowercased substring of ``str(type(model))``; and
  * vae/text-encoder components are routed to their sibling ComfyUI folders, matching both the
    loaders and the pre-refactor on-disk layout.

The drift tripwire is also exercised here against a *faked* ``comfy.model_base`` so it stays
GPU-free; the real check runs at ``hordelib.initialise()`` time.
"""

import sys
import types
from pathlib import Path

import pytest
from horde_model_reference import component_relative_path
from horde_model_reference.meta_consts import KNOWN_IMAGE_GENERATION_BASELINE

from hordelib.execution import comfy_patches


def _all_referenced_class_names() -> set[str]:
    names = set(comfy_patches.FORCE_LOAD_SKIP_CLASS_NAMES)
    for class_names in comfy_patches._baseline_class_names().values():
        names.update(class_names)
    return names


def _install_fake_comfy_model_base(monkeypatch: pytest.MonkeyPatch, class_names: set[str]) -> None:
    """Put a stand-in ``comfy.model_base`` (exposing *class_names* as classes) into sys.modules."""
    model_base = types.ModuleType("comfy.model_base")
    for name in class_names:
        setattr(model_base, name, type(name, (), {}))
    comfy_pkg = types.ModuleType("comfy")
    comfy_pkg.model_base = model_base  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "comfy", comfy_pkg)
    monkeypatch.setitem(sys.modules, "comfy.model_base", model_base)


class TestForceLoadSkipResolution:
    def test_default_list_uses_comfy_class_names(self) -> None:
        # The class names must match comfy.model_base exactly; the old "qwen_image" fragment
        # never matched the comfy class QwenImage (str(type()).lower() == "...qwenimage...").
        assert "QwenImage" in comfy_patches.models_not_to_force_load
        assert "Flux" in comfy_patches.models_not_to_force_load
        assert "qwen_image" not in comfy_patches.models_not_to_force_load

    def test_baseline_resolves_to_class_names(self) -> None:
        resolved = comfy_patches.resolve_force_load_skip_entries(
            [KNOWN_IMAGE_GENERATION_BASELINE.qwen_image],
        )
        assert resolved == ["QwenImage"]

    def test_cascade_baseline_resolves_to_both_stages(self) -> None:
        resolved = comfy_patches.resolve_force_load_skip_entries(
            [KNOWN_IMAGE_GENERATION_BASELINE.stable_cascade],
        )
        assert resolved == ["StableCascade_C", "StableCascade_B"]

    def test_raw_strings_pass_through_and_dedupe(self) -> None:
        resolved = comfy_patches.resolve_force_load_skip_entries(
            [
                KNOWN_IMAGE_GENERATION_BASELINE.flux_1,
                KNOWN_IMAGE_GENERATION_BASELINE.flux_dev,  # also -> "Flux"; must dedupe
                "SomeCustomClass",
            ],
        )
        assert resolved == ["Flux", "SomeCustomClass"]


class TestComponentRelativePath:
    def test_vae_is_redirected_to_sibling_folder(self) -> None:
        assert component_relative_path("qwen_image_vae.safetensors", "vae") == Path(
            "../vae/qwen_image_vae.safetensors"
        )

    def test_text_encoders_redirected(self) -> None:
        assert component_relative_path("te.safetensors", "text_encoders") == Path("../text_encoders/te.safetensors")
        # tolerate the singular spelling used by the node loader's COMPONENT_FILE_TYPES
        assert component_relative_path("te.safetensors", "text_encoder") == Path("../text_encoders/te.safetensors")

    def test_unet_and_unknown_stay_in_manager_folder(self) -> None:
        assert component_relative_path("qwen_image_fp8.safetensors", "unet") == Path("qwen_image_fp8.safetensors")
        assert component_relative_path("model.safetensors", None) == Path("model.safetensors")
        assert component_relative_path("model.safetensors", "checkpoint") == Path("model.safetensors")


class TestForceLoadClassNameDriftTripwire:
    """`assert_force_load_class_names_exist` is the fail-fast for the hard-coded comfy class names."""

    def test_baseline_map_and_flat_list_agree(self) -> None:
        # Internal source-of-truth consistency, checkable without comfy: the baseline map may
        # only reference names declared in FORCE_LOAD_SKIP_CLASS_NAMES, and vice versa.
        baseline_names = {n for names in comfy_patches._baseline_class_names().values() for n in names}
        assert baseline_names == set(comfy_patches.FORCE_LOAD_SKIP_CLASS_NAMES)

    def test_passes_when_all_classes_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_comfy_model_base(monkeypatch, _all_referenced_class_names())
        comfy_patches.assert_force_load_class_names_exist()  # must not raise

    def test_raises_when_a_referenced_class_is_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        drifted = _all_referenced_class_names()
        drifted.discard("QwenImage")  # simulate ComfyUI renaming/removing the class
        _install_fake_comfy_model_base(monkeypatch, drifted)
        with pytest.raises(RuntimeError, match="QwenImage"):
            comfy_patches.assert_force_load_class_names_exist()
