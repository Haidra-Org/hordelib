"""The manifest-driven encoder must produce one vector per payload, forever.

The golden fixture pins both the layout and the exact float32 values, so a refactor, a dependency
bump or a port to the AI-Horde tree that changes an encoding fails here rather than changing what
jobs pay. The semantic assertions beside it pin the behaviour the fixture only illustrates:
collapse, aliasing, defaults and the fallback chain.

CPU-only: nothing here loads a model or touches a GPU.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from hordelib.kudos_training import default_manifest, to_vector
from hordelib.kudos_training.manifest import VECTOR_DTYPE, FloatFeature, KudosFeatureManifest

GOLDEN_VECTORS_PATH = Path(__file__).parent / "kudos_golden_vectors_v22.json"


@pytest.fixture(scope="module")
def manifest() -> KudosFeatureManifest:
    return default_manifest()


@pytest.fixture(scope="module")
def golden() -> dict[str, Any]:
    return json.loads(GOLDEN_VECTORS_PATH.read_text(encoding="utf-8"))


def _case_ids(golden_document: dict[str, Any]) -> list[str]:
    return [case["name"] for case in golden_document["cases"]]


def test_golden_fixture_matches_the_shipped_manifest(manifest: KudosFeatureManifest, golden: dict[str, Any]) -> None:
    assert golden["manifest_version"] == manifest.manifest_version
    assert golden["sampler_semantics"] == manifest.sampler_semantics.model_dump(mode="json")
    assert golden["vector_length"] == manifest.vector_length()
    assert golden["slot_names"] == list(manifest.slot_names())


def test_golden_cases(manifest: KudosFeatureManifest, golden: dict[str, Any]) -> None:
    """Every recorded payload encodes to its recorded vector, exactly."""
    for case in golden["cases"]:
        result = manifest.encode(case["payload"])
        expected = np.asarray(case["expected_vector"], dtype=VECTOR_DTYPE)
        assert np.array_equal(result.vector, expected), f"{case['name']} encoded differently than the golden vector"
        assert result.collapsed == case["expected_collapsed"], case["name"]
        assert result.dropped_unknown == case["expected_dropped_unknown"], case["name"]


def test_golden_cases_cover_the_documented_shapes(golden: dict[str, Any]) -> None:
    """The fixture keeps carrying the cases the encoder is most likely to regress on."""
    names = set(_case_ids(golden))
    assert {
        "basis_payload",
        "v21_payload_example",
        "unknown_sampler_and_post_processor",
        "remix_collapses_to_img2img",
        "missing_optional_fields",
        "karras_false_encodes_normal",
        "named_scheduler_overrides_karras",
        "five_loras_batch_eight",
    } <= names


def test_encoding_is_deterministic(manifest: KudosFeatureManifest, golden: dict[str, Any]) -> None:
    """The same payload encodes to identical bytes, including under a different key order."""
    for case in golden["cases"]:
        payload = case["payload"]
        reordered = dict(reversed(list(payload.items())))
        first = manifest.to_vector(payload)
        second = manifest.to_vector(payload)
        third = manifest.to_vector(reordered)
        assert first.tobytes() == second.tobytes(), case["name"]
        assert first.tobytes() == third.tobytes(), case["name"]


def test_module_level_encoder_uses_the_shipped_manifest(manifest: KudosFeatureManifest) -> None:
    payload = dict(manifest.basis_payload)
    assert to_vector(payload).tobytes() == manifest.to_vector(payload).tobytes()


def test_layout_is_manifest_order(manifest: KudosFeatureManifest) -> None:
    """Slots appear in manifest order, one per float and one per vocabulary entry."""
    slots = manifest.slot_names()
    assert len(slots) == manifest.vector_length()
    assert len(set(slots)) == len(slots)

    offset = 0
    for feature in manifest.features:
        assert slots[offset : offset + feature.width] == feature.slot_names()
        offset += feature.width
    assert offset == len(slots)


def test_one_hot_lands_on_the_slot_the_layout_names(manifest: KudosFeatureManifest) -> None:
    slots = manifest.slot_names()
    vector = manifest.to_vector({**manifest.basis_payload, "sampler_name": "k_dpmpp_2m", "scheduler": "simple"})
    assert vector[slots.index("sampler_name=k_dpmpp_2m")] == 1.0
    assert vector[slots.index("scheduler=simple")] == 1.0
    assert vector[slots.index("sampler_name=k_euler")] == 0.0
    assert sum(float(vector[index]) for index, name in enumerate(slots) if name.startswith("sampler_name=")) == 1.0


def test_unknown_categorical_collapses_and_is_reported(manifest: KudosFeatureManifest) -> None:
    slots = manifest.slot_names()
    result = manifest.encode({**manifest.basis_payload, "sampler_name": "not_a_sampler"})
    assert result.collapsed == {"sampler_name": "k_euler"}
    assert result.vector[slots.index("sampler_name=k_euler")] == 1.0


def test_unknown_post_processor_is_dropped_and_counted(manifest: KudosFeatureManifest) -> None:
    slots = manifest.slot_names()
    result = manifest.encode(
        {**manifest.basis_payload, "post_processing": ["GFPGAN", "not_a_post_processor", "also_not_one"]},
    )
    assert result.dropped_unknown == {"post_processing": 2}
    assert result.vector[slots.index("post_processing=GFPGAN")] == 1.0


def test_repeated_post_processor_counts_up(manifest: KudosFeatureManifest) -> None:
    """The multi-hot slot counts occurrences, as the v21 combined one-hot did."""
    slots = manifest.slot_names()
    vector = manifest.to_vector({**manifest.basis_payload, "post_processing": ["GFPGAN", "GFPGAN"]})
    assert vector[slots.index("post_processing=GFPGAN")] == 2.0


def test_remix_encodes_as_img2img(manifest: KudosFeatureManifest) -> None:
    remix = manifest.encode({**manifest.basis_payload, "source_processing": "remix"})
    img2img = manifest.encode({**manifest.basis_payload, "source_processing": "img2img"})
    assert remix.vector.tobytes() == img2img.vector.tobytes()
    assert not remix.has_unknowns


def _scheduler_slot(manifest: KudosFeatureManifest, payload: dict[str, Any]) -> str:
    """Return the scheduler vocabulary entry *payload* encodes to."""
    slots = manifest.slot_names()
    vector = manifest.to_vector(payload)
    hot = [name for index, name in enumerate(slots) if name.startswith("scheduler=") and vector[index] == 1.0]
    assert len(hot) == 1, f"expected exactly one scheduler slot set, got {hot}"
    return hot[0].removeprefix("scheduler=")


def test_absent_scheduler_reads_the_legacy_karras_flag(manifest: KudosFeatureManifest) -> None:
    """A live payload selects its schedule through `karras`, so a false flag must not encode as karras."""
    base = {"width": 512, "height": 512, "steps": 20}
    assert _scheduler_slot(manifest, {**base, "karras": False}) == "normal"
    assert _scheduler_slot(manifest, {**base, "karras": True}) == "karras"


def test_named_scheduler_wins_over_the_karras_flag(manifest: KudosFeatureManifest) -> None:
    base = {"width": 512, "height": 512, "steps": 20}
    assert _scheduler_slot(manifest, {**base, "scheduler": "sgm_uniform", "karras": True}) == "sgm_uniform"
    assert _scheduler_slot(manifest, {**base, "scheduler": "normal", "karras": True}) == "normal"


def test_scheduler_defaults_to_karras_when_neither_key_is_present(manifest: KudosFeatureManifest) -> None:
    """The SDK defaults karras to true, so a payload naming neither prices as karras."""
    assert _scheduler_slot(manifest, {"width": 512, "height": 512, "steps": 20}) == "karras"


def test_unknown_named_scheduler_collapses_rather_than_reading_karras(manifest: KudosFeatureManifest) -> None:
    """A named schedule is a real selection, so it collapses and is reported instead of falling back."""
    result = manifest.encode({"width": 512, "height": 512, "scheduler": "a_new_schedule", "karras": False})
    assert result.collapsed == {"scheduler": "karras"}


def test_control_strength_falls_back_to_denoising_strength(manifest: KudosFeatureManifest) -> None:
    slots = manifest.slot_names()
    payload = {**manifest.basis_payload, "denoising_strength": 0.4}
    payload.pop("control_strength")
    vector = manifest.to_vector(payload)
    assert vector[slots.index("control_strength")] == pytest.approx(0.4)


def test_missing_fields_take_manifest_defaults(manifest: KudosFeatureManifest) -> None:
    slots = manifest.slot_names()
    vector = manifest.to_vector({})
    for feature in manifest.features:
        if isinstance(feature, FloatFeature) and feature.derived is None:
            expected = feature.default / feature.divisor
            assert vector[slots.index(feature.name)] == pytest.approx(expected), feature.name


def test_derived_megapixels_follows_the_clamped_dimensions(manifest: KudosFeatureManifest) -> None:
    slots = manifest.slot_names()
    vector = manifest.to_vector({"width": 1024, "height": 1536})
    assert vector[slots.index("megapixels")] == pytest.approx(1024 * 1536 / 1e6 / 2.0, rel=1e-6)


def test_out_of_range_values_are_clamped_not_rejected(manifest: KudosFeatureManifest) -> None:
    slots = manifest.slot_names()
    vector = manifest.to_vector({"width": 512, "height": 512, "steps": 100_000, "loras_count": 99})
    assert vector[slots.index("trajectory_steps")] == pytest.approx(500.0 / 100.0)
    assert vector[slots.index("loras_count")] == pytest.approx(1.0)


def test_uncoercible_value_takes_the_default(manifest: KudosFeatureManifest) -> None:
    slots = manifest.slot_names()
    vector = manifest.to_vector({"width": 512, "height": 512, "cfg_scale": "not a number"})
    assert vector[slots.index("cfg_scale")] == pytest.approx(7.5 / 30.0)


def test_v21_step_spelling_is_read(manifest: KudosFeatureManifest) -> None:
    slots = manifest.slot_names()
    trajectory_slot = slots.index("trajectory_steps")
    assert manifest.to_vector({"ddim_steps": 40})[trajectory_slot] == pytest.approx(0.4)
    assert manifest.to_vector({"steps": 40})[trajectory_slot] == pytest.approx(0.4)


def test_sampler_work_rate_does_not_rewrite_trajectory_steps(manifest: KudosFeatureManifest) -> None:
    """Keep raw trajectory length separate from sampler-specific marginal work."""
    slots = manifest.slot_names()
    trajectory_slot = slots.index("trajectory_steps")
    euler_twenty = manifest.to_vector({"steps": 20, "sampler_name": "k_euler"})
    heun_twenty = manifest.to_vector({"steps": 20, "sampler_name": "k_heun"})
    euler_forty = manifest.to_vector({"steps": 40, "sampler_name": "k_euler"})

    assert euler_twenty[trajectory_slot] == pytest.approx(0.2)
    assert heun_twenty[trajectory_slot] == pytest.approx(0.2)
    assert euler_forty[trajectory_slot] == pytest.approx(0.4)
    assert heun_twenty[slots.index("sampler_name=k_heun")] == 1.0
    assert not np.array_equal(heun_twenty, euler_forty)


def test_api_ddim_spelling_aliases_to_backend_spelling(manifest: KudosFeatureManifest) -> None:
    """Encode the SDK/API and backend DDIM spellings as the same sampler without collapse."""
    api_result = manifest.encode({"steps": 20, "sampler_name": "DDIM"})
    backend_result = manifest.encode({"steps": 20, "sampler_name": "ddim"})

    assert np.array_equal(api_result.vector, backend_result.vector)
    assert not api_result.has_unknowns
