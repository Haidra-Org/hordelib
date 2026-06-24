"""Tests for the hordelib ControlNet-annotator model provider.

The provider surfaces the installed ``comfyui_controlnet_aux`` annotator set as a queryable
``controlnet_annotator`` source. The annotator records and category are supplied by the pinned
``horde_model_reference``.
"""

from __future__ import annotations

from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORY

from hordelib.model_manager.annotator_provider import ANNOTATOR_SOURCE_ID, AnnotatorModelProvider


def test_source_id_is_stable_and_not_reserved() -> None:
    """The provider advertises the stable ``comfyui_controlnet_aux`` id and passes registry validation."""
    provider = AnnotatorModelProvider()
    assert provider.source_id == ANNOTATOR_SOURCE_ID
    provider.validate_source_id()  # raises if empty/reserved


def test_serves_only_the_annotator_category() -> None:
    """It advertises ``controlnet_annotator`` and nothing else."""
    provider = AnnotatorModelProvider()
    assert provider.provided_categories() == {MODEL_REFERENCE_CATEGORY.controlnet_annotator}
    assert provider.serves_category(MODEL_REFERENCE_CATEGORY.controlnet_annotator)
    assert not provider.serves_category(MODEL_REFERENCE_CATEGORY.controlnet)


def test_fetch_returns_annotator_records_for_its_category() -> None:
    """``fetch_category`` returns the catalog-derived records for its category, and ``None`` for others."""
    from horde_model_reference.annotator_records import annotator_records

    provider = AnnotatorModelProvider()
    fetched = provider.fetch_category(MODEL_REFERENCE_CATEGORY.controlnet_annotator)
    assert fetched is not None
    assert set(fetched) == set(annotator_records())
    assert provider.fetch_category(MODEL_REFERENCE_CATEGORY.controlnet) is None


def test_is_read_only() -> None:
    """Providers never write; the annotator provider is no exception."""
    assert AnnotatorModelProvider().supports_writes() is False
