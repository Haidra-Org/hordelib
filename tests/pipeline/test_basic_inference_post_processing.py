"""``basic_inference``'s per-image post-processing step, with ``post_process_chain`` replaced."""

import io
from collections.abc import Sequence
from typing import Any
from unittest.mock import MagicMock

import pytest
from horde_sdk.ai_horde_api.apimodels.base import GenMetadataEntry
from horde_sdk.ai_horde_api.consts import METADATA_TYPE, METADATA_VALUE
from PIL import Image

from hordelib.horde import HordeLib, ResultingImageReturn


def _fault() -> GenMetadataEntry:
    return GenMetadataEntry(type=next(iter(METADATA_TYPE)), value=METADATA_VALUE.parse_failed)


def _generated() -> ResultingImageReturn:
    image = Image.new("RGB", (8, 8), (10, 20, 30))
    return ResultingImageReturn(image=image, rawpng=io.BytesIO(b"generation-png"), faults=[])


def _run(chain: Any, ret: ResultingImageReturn, faults: list[GenMetadataEntry]) -> ResultingImageReturn | None:
    fake_self = MagicMock()
    fake_self.post_process_chain = chain
    return HordeLib._post_process_inference_result(
        fake_self,
        ret,
        ["4x_AnimeSharp", "GFPGAN"],
        facefixer_strength=0.5,
        faults=faults,
        image_index=0,
    )


def test_one_chain_call_and_one_encode_matching_the_image() -> None:
    ret = _generated()
    processed_image = Image.new("RGB", (32, 32), (200, 100, 50))
    chain_fault = _fault()
    calls: list[tuple[Image.Image, Sequence[str], float | None]] = []

    def chain(source: Image.Image, operations: Sequence[str], *, facefixer_strength: float | None) -> Any:
        calls.append((source, operations, facefixer_strength))
        return ResultingImageReturn(image=processed_image, rawpng=None, faults=[chain_fault])

    inference_fault = _fault()
    result = _run(chain, ret, [inference_fault])

    assert result is not None
    assert calls == [(ret.image, ["4x_AnimeSharp", "GFPGAN"], 0.5)]
    assert result.image is processed_image
    assert result.faults == [inference_fault, chain_fault]
    assert result.rawpng is not None
    decoded = Image.open(result.rawpng)
    assert decoded.format == "PNG"
    assert decoded.size == (32, 32)
    assert decoded.convert("RGB").getpixel((0, 0)) == (200, 100, 50)


def test_unchanged_image_keeps_generation_rawpng() -> None:
    ret = _generated()

    def chain(source: Image.Image, _operations: Sequence[str], **_kwargs: Any) -> Any:
        return ResultingImageReturn(image=source, rawpng=None, faults=[])

    result = _run(chain, ret, [])

    assert result is not None
    assert result.rawpng is ret.rawpng


def test_no_output_image_drops_the_result() -> None:
    def chain(*_args: Any, **_kwargs: Any) -> Any:
        return ResultingImageReturn(image=None, rawpng=None, faults=[])

    assert _run(chain, _generated(), []) is None


def test_chain_failure_propagates() -> None:
    def chain(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("stage failed")

    with pytest.raises(RuntimeError, match="stage failed"):
        _run(chain, _generated(), [])
