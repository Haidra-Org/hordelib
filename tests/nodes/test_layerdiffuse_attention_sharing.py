"""Unit tests for the layer-diffuse attention-sharing unit's handling of an already-patched attention block.

A resident model keeps the object patches its last run applied, so a second transparent job on the same
process hands ``AttentionSharingUnit`` one of its own instances instead of the attention block. The unit must
wrap the block underneath, so the second job builds the same unit the first did.

``attention_sharing`` imports ``comfy`` at module scope, so these depend on the session ``init_horde`` fixture
and import locally rather than at collection time.
"""

from __future__ import annotations

import pytest
import torch


def _attention_block(dim: int = 8, heads: int = 2) -> torch.nn.Module:
    block = torch.nn.Module()
    block.heads = heads
    block.to_q = torch.nn.Linear(dim, dim)
    block.to_k = torch.nn.Linear(dim, dim)
    block.to_v = torch.nn.Linear(dim, dim)
    block.to_out = torch.nn.ModuleList([torch.nn.Linear(dim, dim)])
    return block


@pytest.mark.usefixtures("init_horde")
class TestAttentionSharingUnitRewrap:
    def test_unit_built_on_a_unit_wraps_the_underlying_attention(self) -> None:
        from hordelib.nodes.comfyui_layerdiffuse.lib_layerdiffusion.attention_sharing import AttentionSharingUnit

        block = _attention_block()
        first = AttentionSharingUnit(block, frames=2, use_control=False, rank=4)

        second = AttentionSharingUnit(first, frames=2, use_control=False, rank=4)

        assert second.original_module[0] is block
        assert second.heads == block.heads
        assert len(second.to_q_lora) == 2

    def test_unit_built_on_a_plain_block_is_unchanged(self) -> None:
        from hordelib.nodes.comfyui_layerdiffuse.lib_layerdiffusion.attention_sharing import AttentionSharingUnit

        block = _attention_block()

        unit = AttentionSharingUnit(block, frames=2, use_control=False, rank=4)

        assert unit.original_module[0] is block
