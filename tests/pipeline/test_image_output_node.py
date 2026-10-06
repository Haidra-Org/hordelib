"""Unit tests for ``HordeImageOutput`` in its encoded and unencoded modes, and their collection."""

import io

import torch
from PIL import Image

from hordelib.execution.results import (
    UI_ENTRY_IMAGE_KEY,
    UNENCODED_IMAGE_TYPE,
    collect_output_entries,
    is_unencoded_image_entry,
)
from hordelib.nodes.node_image_output import HordeImageOutput


def _images() -> torch.Tensor:
    # One 4x3 RGB image in ComfyUI's [batch, height, width, channel] float layout.
    return torch.full((1, 3, 4, 3), 0.5)


def test_default_encodes_png() -> None:
    entries = HordeImageOutput().get_image(_images())["ui"]["images"]

    assert len(entries) == 1
    assert entries[0]["type"] == "PNG"
    assert isinstance(entries[0]["imagedata"], io.BytesIO)
    assert entries[0]["imagedata"].getvalue().startswith(b"\x89PNG")


def test_unencoded_returns_pil_image() -> None:
    entries = HordeImageOutput().get_image(_images(), encode_png=False, prompt={"1": {}})["ui"]["images"]

    assert len(entries) == 1
    entry = entries[0]
    assert set(entry) == {UI_ENTRY_IMAGE_KEY, "type"}
    assert entry["type"] == UNENCODED_IMAGE_TYPE
    image = entry[UI_ENTRY_IMAGE_KEY]
    assert isinstance(image, Image.Image)
    assert image.size == (4, 3)
    assert image.getpixel((0, 0)) == (127, 127, 127)
    assert not getattr(image, "text", {})


def test_encode_png_is_an_optional_boolean_defaulting_true() -> None:
    optional = HordeImageOutput.INPUT_TYPES()["optional"]

    assert optional["encode_png"] == ("BOOLEAN", {"default": True})


def test_collection_keeps_unencoded_entries_with_their_source_node() -> None:
    ui = HordeImageOutput().get_image(_images(), encode_png=False)["ui"]

    entries = collect_output_entries({"output_image": ui})

    assert len(entries) == 1
    assert is_unencoded_image_entry(entries[0])
    assert entries[0]["source_node"] == "output_image"


def test_collection_skips_mislabelled_unencoded_entry() -> None:
    ui = {"images": [{UI_ENTRY_IMAGE_KEY: io.BytesIO(b"\x89PNG"), "type": UNENCODED_IMAGE_TYPE}]}

    assert collect_output_entries({"output_image": ui}) == []
