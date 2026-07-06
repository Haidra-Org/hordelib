"""Tests for the loaded-component content hash (:func:`hordelib.execution.shared_components.hash_state_dict`).

The hash keys on tensor bytes, shapes and dtypes, so byte-identical components hash equal and any
semantic difference (data, shape, or dtype) changes the digest.
"""

from __future__ import annotations

import torch

from hordelib.execution.shared_components import hash_state_dict


class TestHashing:
    """Content hashing: equality on identical bytes, sensitivity to any semantic difference."""

    def test_identical_state_dicts_hash_equal(self) -> None:
        a = {"w": torch.arange(16, dtype=torch.float32).reshape(4, 4)}
        b = {"w": torch.arange(16, dtype=torch.float32).reshape(4, 4)}
        assert hash_state_dict(a) == hash_state_dict(b)

    def test_data_shape_and_dtype_all_matter(self) -> None:
        base = {"w": torch.zeros(4, 4, dtype=torch.float32)}
        assert hash_state_dict({"w": torch.ones(4, 4)}) != hash_state_dict(base)
        assert hash_state_dict({"w": torch.zeros(16)}) != hash_state_dict(base)
        assert hash_state_dict({"w": torch.zeros(4, 4, dtype=torch.float16)}) != hash_state_dict(base)

    def test_hashes_zero_dim_scalar(self) -> None:
        # A 0-dim scalar (e.g. an fp8 scale factor baked into a Qwen text encoder) must hash without raising:
        # view(torch.uint8) rejects a 0-dim tensor, so the hash flattens first.
        assert hash_state_dict({"scale": torch.tensor(1.5)}) != hash_state_dict({"scale": torch.tensor(2.5)})

    def test_non_tensor_entries_are_ignored(self) -> None:
        # Non-tensor state dict values do not contribute to (or break) the digest.
        base = {"w": torch.zeros(2, 2)}
        assert hash_state_dict({"w": torch.zeros(2, 2), "meta": "unused"}) == hash_state_dict(base)
