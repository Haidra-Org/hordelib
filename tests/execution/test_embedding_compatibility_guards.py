"""Defense-in-depth guards around ComfyUI textual-inversion handling."""

import torch

from hordelib.execution import comfy_patches


def test_load_embed_hijack_rejects_a_tensor_with_the_wrong_width(monkeypatch) -> None:
    incompatible = torch.zeros((1, 768))
    monkeypatch.setitem(comfy_patches._originals, "load_embed", lambda *_args: incompatible)

    result = comfy_patches.load_embed_hijack("sd15", [], embedding_size=1024)

    assert result is None


def test_load_embed_hijack_preserves_a_tensor_with_the_expected_width(monkeypatch) -> None:
    compatible = torch.zeros((1, 1024))
    monkeypatch.setitem(comfy_patches._originals, "load_embed", lambda *_args: compatible)

    result = comfy_patches.load_embed_hijack("qwen", [], embedding_size=1024)

    assert result is compatible


def test_anima_guard_discards_tensors_before_the_integer_conversion(monkeypatch) -> None:
    captured: dict[str, list[list[tuple]]] = {}

    def capture(_model: object, token_weight_pairs: dict[str, list[list[tuple]]]) -> str:
        captured.update(token_weight_pairs)
        return "encoded"

    monkeypatch.setitem(comfy_patches._originals, "anima_encode_token_weights", capture)
    token_weight_pairs: dict[str, list[list[tuple]]] = {
        "qwen3_06b": [[(21, 1.0)]],
        "t5xxl": [[(11, 1.0), (torch.zeros(768), 0.5), (12, 0.75)]],
    }

    result = comfy_patches.anima_encode_token_weights_hijack(object(), token_weight_pairs)

    assert result == "encoded"
    assert captured["qwen3_06b"] == token_weight_pairs["qwen3_06b"]
    assert captured["t5xxl"] == [[(11, 1.0), (12, 0.75)]]


def test_anima_guard_keeps_a_pad_token_when_the_embedding_was_the_only_entry() -> None:
    result = comfy_patches._integer_only_token_batches([[(torch.zeros(768), 1.0)]])

    assert result == [[(0, 1.0)]]
