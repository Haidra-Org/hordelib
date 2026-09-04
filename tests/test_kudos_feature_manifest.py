"""The manifest's vocabularies must still cover what hordelib offers callers.

A published manifest is frozen, so this suite does not compare the two for equality; it checks that
the live vocabulary is a subset of the manifest's. hordelib gaining a sampler or a schedule is
therefore the alarm that a manifest revision (and a retrained model) is due, rather than something
that quietly re-prices jobs through the unknown-collapse path.

CPU-only: nothing here loads a model or touches a GPU.
"""

import pytest
from horde_model_reference.model_consts.image import KNOWN_IMAGE_GENERATION_BASELINE
from horde_sdk.generation_parameters.alchemy.consts import KNOWN_FACEFIXERS, KNOWN_UPSCALERS
from horde_sdk.generation_parameters.image.constraints_document import SAMPLER_CONSTRAINTS_DOCUMENT_SCHEMA_VERSION
from horde_sdk.generation_parameters.image.consts import KNOWN_IMAGE_SAMPLERS
from horde_sdk.generation_parameters.image.sampler_work import SamplerExecutionContractVersion

from hordelib.kudos_training import default_manifest
from hordelib.kudos_training.manifest import CategoricalFeature, MultiHotFeature
from hordelib.pipeline.constants import (
    CONTROLNET_IMAGE_PREPROCESSOR_MAP,
    SAMPLERS_MAP,
    SCHEDULERS,
    SOURCE_IMAGE_PROCESSING_OPTIONS,
    UPSCALER_SCALE_FACTORS,
)
from hordelib.pipeline.payload_pp import STRIP_BACKGROUND_NAME

MANIFEST_REVISION_ADVICE = (
    "hordelib now offers a value the frozen kudos manifest does not carry, so the encoder would "
    "price it through its unknown-collapse target. Cut a new manifest revision "
    "(hordelib/kudos_training/kudos_feature_manifest_vNN.json) covering it and retrain, or record a "
    "deliberate decision to keep collapsing it."
)


BACKEND_DEFAULT_NAME = "BACKEND_DEFAULT"
"""The name that asks the worker for whichever post-processor it defaults to.

It selects a model rather than naming one, so it is not a vocabulary slot: whichever upscaler or
face fixer it resolves to is already priced under that model's own name.
"""


def _vocabulary(name: str) -> set[str]:
    """Return the manifest vocabulary of the named categorical or multi-hot feature."""
    manifest = default_manifest()
    for feature in manifest.features:
        if feature.name == name:
            assert isinstance(feature, CategoricalFeature | MultiHotFeature)
            return set(feature.vocabulary)
    raise AssertionError(f"manifest has no feature named {name!r}")


def _aliases(name: str) -> dict[str, str]:
    """Return the value aliases of the named categorical or multi-hot feature."""
    manifest = default_manifest()
    for feature in manifest.features:
        if feature.name == name:
            assert isinstance(feature, CategoricalFeature | MultiHotFeature)
            return dict(feature.value_aliases)
    raise AssertionError(f"manifest has no feature named {name!r}")


def test_manifest_identity() -> None:
    manifest = default_manifest()
    assert manifest.manifest_version == "v22"
    assert manifest.target == "sampler_window_seconds"
    assert manifest.sampler_semantics.horde_sdk_version == "0.29.0"
    assert (
        manifest.sampler_semantics.constraints_document_schema_version == SAMPLER_CONSTRAINTS_DOCUMENT_SCHEMA_VERSION
    )
    assert manifest.sampler_semantics.execution_contract_version is SamplerExecutionContractVersion.V1
    assert len(manifest.sampler_semantics.constraints_artifact_sha256) == 64


def test_sampler_vocabulary_covers_hordelib() -> None:
    missing = sorted(set(SAMPLERS_MAP) - _vocabulary("sampler_name"))
    assert not missing, f"samplers absent from the manifest: {missing}. {MANIFEST_REVISION_ADVICE}"


def test_sampler_vocabulary_covers_sdk_api_names() -> None:
    """Require every SDK sampler name to be a slot or an explicit spelling alias."""
    covered = _vocabulary("sampler_name") | set(_aliases("sampler_name"))
    missing = sorted(sampler.value for sampler in KNOWN_IMAGE_SAMPLERS if sampler.value not in covered)
    assert not missing, f"SDK samplers absent from the manifest: {missing}. {MANIFEST_REVISION_ADVICE}"


def test_ddim_api_spelling_is_an_explicit_alias() -> None:
    assert _aliases("sampler_name")[KNOWN_IMAGE_SAMPLERS.DDIM.value] == "ddim"


def test_scheduler_vocabulary_covers_hordelib() -> None:
    missing = sorted(set(SCHEDULERS) - _vocabulary("scheduler"))
    assert not missing, f"schedulers absent from the manifest: {missing}. {MANIFEST_REVISION_ADVICE}"


def test_baseline_vocabulary_covers_the_model_reference() -> None:
    """Every baseline the model reference can assign holds a slot or aliases to one.

    A baseline outside the vocabulary collapses to ``other`` and is priced as the average of whatever
    else landed there, which for a heavy family under-prices every job on it.
    """
    live = {baseline.value for baseline in KNOWN_IMAGE_GENERATION_BASELINE} - {"infer"}
    covered = _vocabulary("baseline") | set(_aliases("baseline"))
    missing = sorted(live - covered)
    assert not missing, f"baselines absent from the manifest: {missing}. {MANIFEST_REVISION_ADVICE}"


def test_control_type_vocabulary_covers_hordelib() -> None:
    live = set(CONTROLNET_IMAGE_PREPROCESSOR_MAP) | {"None"}
    missing = sorted(live - _vocabulary("control_type"))
    assert not missing, f"control types absent from the manifest: {missing}. {MANIFEST_REVISION_ADVICE}"


def test_post_processing_vocabulary_covers_hordelib() -> None:
    """Every post-processor the pipeline can actually run must hold a slot.

    A post-processor outside the vocabulary is dropped rather than collapsed, so it would be priced
    as though it had never been requested.
    """
    live = (
        set(UPSCALER_SCALE_FACTORS)
        | {upscaler.value for upscaler in KNOWN_UPSCALERS}
        | {facefixer.value for facefixer in KNOWN_FACEFIXERS}
        | {STRIP_BACKGROUND_NAME}
    ) - {BACKEND_DEFAULT_NAME}
    missing = sorted(live - _vocabulary("post_processing"))
    assert not missing, f"post-processors absent from the manifest: {missing}. {MANIFEST_REVISION_ADVICE}"


def test_source_processing_vocabulary_covers_hordelib() -> None:
    live = set(SOURCE_IMAGE_PROCESSING_OPTIONS) | {"txt2img"}
    covered = _vocabulary("source_processing") | set(_aliases("source_processing"))
    missing = sorted(live - covered)
    assert not missing, f"source processing modes absent from the manifest: {missing}. {MANIFEST_REVISION_ADVICE}"


def test_remix_is_aliased_rather_than_collapsed() -> None:
    """`remix` prices as img2img by a named alias, so it is not reported as an unknown value."""
    assert _aliases("source_processing")["remix"] == "img2img"
    assert "remix" not in _vocabulary("source_processing")


@pytest.mark.parametrize(
    ("feature_name", "expected_collapse"),
    [
        ("sampler_name", "k_euler"),
        ("scheduler", "karras"),
        ("baseline", "other"),
        ("control_type", "None"),
        ("source_processing", "txt2img"),
    ],
)
def test_every_categorical_names_its_collapse_target(feature_name: str, expected_collapse: str) -> None:
    manifest = default_manifest()
    feature = next(entry for entry in manifest.features if entry.name == feature_name)
    assert isinstance(feature, CategoricalFeature)
    assert feature.unknown_collapse == expected_collapse
    assert feature.unknown_collapse in feature.vocabulary


def test_scheduler_declares_its_legacy_boolean_fallback() -> None:
    """The schedule a horde request selects through `karras` must be named in the manifest, not in code."""
    manifest = default_manifest()
    feature = next(entry for entry in manifest.features if entry.name == "scheduler")
    assert isinstance(feature, CategoricalFeature)
    assert feature.boolean_fallback is not None
    assert feature.boolean_fallback.payload_key == "karras"
    assert feature.boolean_fallback.when_true == "karras"
    assert feature.boolean_fallback.when_false == "normal"


def test_basis_payload_encodes_without_collapse_or_drop() -> None:
    """The anchoring job must be expressible in the manifest's own vocabularies."""
    manifest = default_manifest()
    result = manifest.encode(manifest.basis_payload)
    assert not result.has_unknowns
    assert len(result.vector) == manifest.vector_length()
