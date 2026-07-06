"""Round-trip tests for disaggregated-stage wire serialization (:mod:`hordelib.execution.stage_payloads`).

These run GPU-free with CPU tensors: they pin that a CONDITIONING (tensor + pooled_output + scalar
metadata) and a LATENT survive serialize/deserialize byte-identically with their dtypes preserved, and
that a dict entry that is neither a tensor nor a JSON scalar is dropped (with a warning) without
corrupting the rest of the payload.
"""

from __future__ import annotations

import torch
from loguru import logger

from hordelib.execution.stage_payloads import (
    deserialize_conditioning,
    deserialize_latent,
    serialize_conditioning,
    serialize_latent,
)


def _assert_identical(a: torch.Tensor, b: torch.Tensor) -> None:
    assert a.dtype == b.dtype
    assert a.shape == b.shape
    assert torch.equal(a, b)


class TestConditioningRoundTrip:
    def test_pooled_output_and_metadata_survive_fp32(self) -> None:
        cond = torch.randn(1, 77, 768, dtype=torch.float32)
        pooled = torch.randn(1, 1280, dtype=torch.float32)
        # SDXL conditioning carries width/height/crop as separate int scalars alongside the pooled tensor.
        conditioning = [[cond, {"pooled_output": pooled, "width": 1024, "height": 768, "crop_w": 0, "crop_h": 0}]]

        restored = deserialize_conditioning(serialize_conditioning(conditioning))

        assert len(restored) == 1
        restored_cond, restored_dict = restored[0]
        _assert_identical(restored_cond, cond)
        _assert_identical(restored_dict["pooled_output"], pooled)
        assert restored_dict["width"] == 1024
        assert restored_dict["height"] == 768
        assert restored_dict["crop_w"] == 0
        assert restored_dict["crop_h"] == 0

    def test_fp16_tensors_keep_dtype(self) -> None:
        cond = (torch.randn(1, 77, 768) * 4).to(torch.float16)
        pooled = (torch.randn(1, 1280) * 4).to(torch.float16)
        conditioning = [[cond, {"pooled_output": pooled}]]

        restored_cond, restored_dict = deserialize_conditioning(serialize_conditioning(conditioning))[0]

        _assert_identical(restored_cond, cond)
        _assert_identical(restored_dict["pooled_output"], pooled)

    def test_multiple_pairs_preserved(self) -> None:
        conditioning = [
            [torch.zeros(1, 4, dtype=torch.float32), {"width": 512}],
            [torch.ones(1, 4, dtype=torch.float32), {"width": 768}],
        ]

        restored = deserialize_conditioning(serialize_conditioning(conditioning))

        assert [pair[1]["width"] for pair in restored] == [512, 768]
        _assert_identical(restored[0][0], conditioning[0][0])
        _assert_identical(restored[1][0], conditioning[1][0])

    def test_non_serializable_entry_dropped_with_warning(self) -> None:
        cond = torch.zeros(1, 4, dtype=torch.float32)
        pooled = torch.ones(1, 4, dtype=torch.float32)
        # A live model ref (a plain object) is neither a tensor nor a JSON scalar: it must be dropped,
        # and the drop must be logged rather than silent.
        conditioning = [[cond, {"pooled_output": pooled, "width": 640, "control": object()}]]

        warnings: list[str] = []
        sink_id = logger.add(lambda message: warnings.append(message.record["message"]), level="WARNING")
        try:
            blob = serialize_conditioning(conditioning)
        finally:
            logger.remove(sink_id)

        restored_cond, restored_dict = deserialize_conditioning(blob)[0]
        assert "control" not in restored_dict
        assert restored_dict["width"] == 640
        _assert_identical(restored_dict["pooled_output"], pooled)
        _assert_identical(restored_cond, cond)
        assert any("control" in message for message in warnings)


class TestLatentRoundTrip:
    def test_samples_scalar_and_extra_tensor_survive(self) -> None:
        # samples plus an extra tensor entry (e.g. a noise mask) and a scalar; a non-scalar/non-tensor
        # entry is dropped (only samples/tensors/scalars are on the wire).
        noise_mask = torch.ones(1, 1, 64, 64, dtype=torch.float32)
        latent = {"samples": torch.randn(1, 4, 64, 64, dtype=torch.float32), "noise_mask": noise_mask, "flag": True}

        restored = deserialize_latent(serialize_latent(latent))

        _assert_identical(restored["samples"], latent["samples"])
        _assert_identical(restored["noise_mask"], noise_mask)
        assert restored["flag"] is True

    def test_fp16_samples_keep_dtype(self) -> None:
        latent = {"samples": (torch.randn(1, 4, 8, 8) * 4).to(torch.float16)}

        restored = deserialize_latent(serialize_latent(latent))

        _assert_identical(restored["samples"], latent["samples"])
