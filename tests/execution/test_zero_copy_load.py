"""Tests for the zero-copy (assign-on-load) state-dict adoption used by the checkpoint loader.

The contract: inside the context manager, a dtype-matching load adopts the incoming tensors (module
parameters share the source storage: what makes mmap-backed checkpoint weights stay shared across
processes); a dtype-mismatched load falls back to the ordinary copy (preserving cast semantics); and
outside the context manager nothing changes.
"""

from __future__ import annotations

import torch

from hordelib.execution.zero_copy_load import zero_copy_state_dict_assignment


def _module() -> torch.nn.Module:
    module = torch.nn.Linear(4, 4, bias=False)
    return module.to(torch.float16)


def test_matching_dtype_adopts_source_storage() -> None:
    """Inside the scope, a same-dtype load makes the parameter share the incoming tensor's memory."""
    module = _module()
    incoming = torch.ones(4, 4, dtype=torch.float16)

    with zero_copy_state_dict_assignment():
        module.load_state_dict({"weight": incoming})

    assert module.weight.data_ptr() == incoming.data_ptr()


def test_mismatched_dtype_falls_back_to_copy() -> None:
    """A dtype mismatch (an implied cast) must copy, never adopt, so numerics are unchanged."""
    module = _module()
    incoming = torch.ones(4, 4, dtype=torch.float32)

    with zero_copy_state_dict_assignment():
        module.load_state_dict({"weight": incoming})

    assert module.weight.dtype == torch.float16
    assert module.weight.data_ptr() != incoming.data_ptr()
    assert torch.equal(module.weight.float(), incoming)


def test_outside_scope_copies_as_normal() -> None:
    """Without the context manager, load_state_dict keeps torch's default copying behavior."""
    module = _module()
    incoming = torch.ones(4, 4, dtype=torch.float16)

    module.load_state_dict({"weight": incoming})

    assert module.weight.data_ptr() != incoming.data_ptr()


def test_hook_is_removed_after_scope() -> None:
    """The global method patch is scoped: after exit the original load_state_dict is restored."""
    original = torch.nn.Module.load_state_dict
    with zero_copy_state_dict_assignment():
        assert torch.nn.Module.load_state_dict is not original
    assert torch.nn.Module.load_state_dict is original


def test_nested_scopes_are_safe() -> None:
    """Nesting keeps the hook installed until the outermost exit and still restores it."""
    original = torch.nn.Module.load_state_dict
    with zero_copy_state_dict_assignment(), zero_copy_state_dict_assignment():
        module = _module()
        incoming = torch.ones(4, 4, dtype=torch.float16)
        module.load_state_dict({"weight": incoming})
        assert module.weight.data_ptr() == incoming.data_ptr()
    assert torch.nn.Module.load_state_dict is original
