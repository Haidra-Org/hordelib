"""Tests for the controlnet annotator download-size ROM table (GPU-free)."""

from hordelib.pipeline.constants import (
    CONTROLNET_ANNOTATOR_DOWNLOAD_BYTES,
    CONTROLNET_IMAGE_PREPROCESSOR_MAP,
    controlnet_annotator_download_bytes,
)


def test_known_type_returns_its_size() -> None:
    assert controlnet_annotator_download_bytes(["depth"]) == CONTROLNET_ANNOTATOR_DOWNLOAD_BYTES["depth"]


def test_duplicates_are_counted_once() -> None:
    """An annotator is fetched once and shared on disk, so a repeated control type costs it once."""
    assert controlnet_annotator_download_bytes(["depth", "depth"]) == CONTROLNET_ANNOTATOR_DOWNLOAD_BYTES["depth"]


def test_distinct_types_sum() -> None:
    expected = CONTROLNET_ANNOTATOR_DOWNLOAD_BYTES["depth"] + CONTROLNET_ANNOTATOR_DOWNLOAD_BYTES["hed"]
    assert controlnet_annotator_download_bytes(["depth", "hed"]) == expected


def test_unknown_and_none_are_ignored() -> None:
    assert controlnet_annotator_download_bytes(["not-a-real-type", None]) == 0


def test_empty_is_zero() -> None:
    assert controlnet_annotator_download_bytes([]) == 0


def test_pure_cv2_types_are_zero() -> None:
    """canny/scribble run on cv2 alone and download no checkpoint."""
    assert controlnet_annotator_download_bytes(["canny", "scribble"]) == 0


def test_keys_align_with_the_preprocessor_map() -> None:
    """Every sizing key is a real control type, so the table can never drift to a phantom type."""
    assert set(CONTROLNET_ANNOTATOR_DOWNLOAD_BYTES) <= set(CONTROLNET_IMAGE_PREPROCESSOR_MAP)


def test_api_reexports_the_helper() -> None:
    from hordelib import api

    assert api.controlnet_annotator_download_bytes is controlnet_annotator_download_bytes
    assert api.CONTROLNET_ANNOTATOR_DOWNLOAD_BYTES is CONTROLNET_ANNOTATOR_DOWNLOAD_BYTES
