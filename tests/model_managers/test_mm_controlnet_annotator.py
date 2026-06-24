"""Unit tests for the first-class ``ControlNetAnnotatorModelManager`` (no ComfyUI / GPU init).

The ControlNet annotator checkpoints are modelled by ``horde_model_reference`` as ``controlnet_annotator``
records; this manager surfaces them through the standard ``BaseModelManager`` interface (enumeration,
per-file on-disk presence, per-file downloads) rooted at the shared ``controlnet/annotators`` folder. The
records come from the in-package ``comfyui_controlnet_aux`` provider, so the manager loads offline without any
on-disk reference file and without constructing the inference stack.
"""

from __future__ import annotations

from collections.abc import Generator

import pytest
from horde_model_reference import MODEL_REFERENCE_CATEGORY, ModelReferenceManager, horde_model_reference_settings
from horde_model_reference.annotator_records import annotator_records

from hordelib.model_manager.annotator_provider import ANNOTATOR_SOURCE_ID
from hordelib.model_manager.controlnet_annotator import ControlNetAnnotatorModelManager
from hordelib.shared_model_manager import SharedModelManager


@pytest.fixture
def annotator_only_manager() -> Generator[ControlNetAnnotatorModelManager]:
    """Load only the annotator manager (offline), saving/restoring the process singletons around the test."""
    mrm_prev = ModelReferenceManager._instance
    smm_inst_prev = SharedModelManager._instance
    smm_mgr_prev = getattr(SharedModelManager, "manager", None)
    offline_prev = SharedModelManager._reference_offline
    settings_offline_prev = horde_model_reference_settings.offline

    ModelReferenceManager._instance = None
    SharedModelManager._instance = None  # type: ignore[assignment]
    SharedModelManager._reference_offline = None
    try:
        SharedModelManager(do_not_load_model_mangers=True)
        SharedModelManager.load_model_managers(
            [MODEL_REFERENCE_CATEGORY.controlnet_annotator],
            reference_offline=True,
        )
        manager = SharedModelManager.manager.controlnet_annotator
        assert manager is not None
        yield manager
    finally:
        ModelReferenceManager._instance = mrm_prev
        SharedModelManager._instance = smm_inst_prev
        SharedModelManager.manager = smm_mgr_prev  # type: ignore[assignment]
        SharedModelManager._reference_offline = offline_prev
        horde_model_reference_settings.offline = settings_offline_prev


def test_manager_is_first_class_and_loads_the_annotator_records(
    annotator_only_manager: ControlNetAnnotatorModelManager,
) -> None:
    """The ``controlnet_annotator`` manager is a real manager whose reference is the catalog's records."""
    assert isinstance(annotator_only_manager, ControlNetAnnotatorModelManager)
    assert set(annotator_only_manager.model_reference) == set(annotator_records())
    assert annotator_only_manager.model_reference  # the catalog is non-empty


def test_files_are_rooted_at_the_shared_controlnet_annotators_folder(
    annotator_only_manager: ControlNetAnnotatorModelManager,
) -> None:
    """Annotators share the ControlNet folder, so a downloaded file lands at ``controlnet/annotators/...``.

    The category resolves to the same on-disk folder as ``controlnet`` and each record's file name is rooted
    at ``annotators/``; together that reproduces the layout ``comfyui_controlnet_aux`` reads from.
    """
    assert annotator_only_manager.model_folder_path.name == "controlnet"
    for name in annotator_only_manager.model_reference:
        for download in annotator_only_manager.get_model_download(name):
            assert download["file_name"].startswith("annotators/")


def test_reference_source_is_the_annotator_provider(
    annotator_only_manager: ControlNetAnnotatorModelManager,
) -> None:
    """The manager loads from the ``comfyui_controlnet_aux`` provider, not the (empty) canonical source."""
    ref_manager = ModelReferenceManager.get_instance()
    assert annotator_only_manager._reference_source(ref_manager) == ANNOTATOR_SOURCE_ID


def test_presence_is_per_record_and_existence_based(
    annotator_only_manager: ControlNetAnnotatorModelManager,
) -> None:
    """``is_model_available`` answers per record (existence on disk), the same authority as every category."""
    # Whatever the on-disk state, availability is a clean bool per record (never raises, never partial).
    for name in annotator_only_manager.model_reference:
        assert isinstance(annotator_only_manager.is_model_available(name), bool)
