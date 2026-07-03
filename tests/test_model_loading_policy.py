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


class _FakePatcher:
    """Stand-in ModelPatcher exposing just what the load hijack inspects."""

    def __init__(self, model: object, size: int) -> None:
        self.model = model
        self._size = size

    def model_size(self) -> int:
        return self._size


class TestSmallModelWorkingClamp:
    """VAE-sized loads must not evict co-resident diffusion weights to satisfy a working estimate.

    Callers pass a worst-case working-memory estimate (a 1MP decode estimates several GB) as
    ``memory_required``; ComfyUI frees that much up front, evicting a resident diffusion model to
    host a few hundred MB of VAE. The hijack clamps the estimate for small non-diffusion loads,
    relying on ComfyUI's tiled-decode OOM fallback as the backstop.
    """

    def _fake_base_model_cls(self, monkeypatch: pytest.MonkeyPatch) -> type:
        _install_fake_comfy_model_base(monkeypatch, _all_referenced_class_names() | {"BaseModel"})
        import comfy.model_base

        return comfy.model_base.BaseModel

    def test_small_support_models_qualify(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._fake_base_model_cls(monkeypatch)
        vae = _FakePatcher(model=object(), size=160 * 1024**2)
        assert comfy_patches._small_support_models_only([vae]) is True

    def test_diffusion_model_disqualifies(self, monkeypatch: pytest.MonkeyPatch) -> None:
        base_model_cls = self._fake_base_model_cls(monkeypatch)
        unet = _FakePatcher(model=base_model_cls(), size=160 * 1024**2)
        assert comfy_patches._small_support_models_only([unet]) is False

    def test_oversize_support_model_disqualifies(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._fake_base_model_cls(monkeypatch)
        big = _FakePatcher(model=object(), size=comfy_patches.SMALL_MODEL_WORKING_CLAMP_MAX_BYTES + 1)
        assert comfy_patches._small_support_models_only([big]) is False

    def test_sizing_failure_fails_closed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._fake_base_model_cls(monkeypatch)

        class _Unsizable:
            model = object()

            def model_size(self) -> int:
                raise RuntimeError("no size")

        assert comfy_patches._small_support_models_only([_Unsizable()]) is False

    def test_hijack_clamps_vae_working_estimate(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._fake_base_model_cls(monkeypatch)
        captured: dict = {}

        def capture(models, **kwargs):
            captured.update(kwargs)

        monkeypatch.setitem(comfy_patches._originals, "load_models_gpu", capture)
        vae = _FakePatcher(model=object(), size=160 * 1024**2)

        comfy_patches._load_models_gpu_hijack([vae], memory_required=5 * 1024**3)

        assert captured["force_full_load"] is True
        assert captured["memory_required"] == 0

    def test_hijack_preserves_estimate_for_diffusion_models(self, monkeypatch: pytest.MonkeyPatch) -> None:
        base_model_cls = self._fake_base_model_cls(monkeypatch)
        captured: dict = {}

        def capture(models, **kwargs):
            captured.update(kwargs)

        monkeypatch.setitem(comfy_patches._originals, "load_models_gpu", capture)
        unet = _FakePatcher(model=base_model_cls(), size=5 * 1024**3)

        comfy_patches._load_models_gpu_hijack([unet], memory_required=7 * 1024**3)

        assert captured["force_full_load"] is True
        assert captured["memory_required"] == 7 * 1024**3
