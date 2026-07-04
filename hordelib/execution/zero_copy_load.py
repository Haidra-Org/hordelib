"""Zero-copy checkpoint loading: keep module weights backed by the safetensors mmap.

safetensors (and comfy's ``load_torch_file``) already produce CPU tensors that are zero-copy views
over a memory-mapped checkpoint file. The materialization happens one step later: building the model
calls ``torch.nn.Module.load_state_dict``, whose default behavior *copies* every view into privately
committed module parameters. Measured on a 6.5GB SDXL checkpoint, that turns a ~0.4GB-private load
into ~7GB private, per process; with several worker processes pinning the same model in RAM, the
duplication multiplies, while mmap-backed views would all share one set of physical pages through the
OS page cache (and warm reloads become nearly free).

``assign=True`` (torch >= 2.1) makes ``load_state_dict`` adopt the incoming tensors instead of copying.
Adopting is only byte-identical to copying when no conversion was implied, so the wrapper applies it
per call and only when every incoming tensor's dtype matches its destination parameter; any mismatch
(e.g. a component the caller intends to cast) falls back to the ordinary copying load, preserving the
backend's cast semantics exactly. The hook is scoped by context manager to the checkpoint build alone,
so no other ``load_state_dict`` caller in the process is affected.
"""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Generator, Mapping
from typing import Any

import torch
from loguru import logger

_hook_state = threading.local()

_original_load_state_dict = torch.nn.Module.load_state_dict


def _all_dtypes_match(module: torch.nn.Module, state_dict: Mapping[str, Any]) -> bool:
    """Whether every state-dict tensor matches the dtype of the destination it would replace.

    Only keys present on both sides are compared (comfy loads with ``strict=False`` and prunes
    prefixes); an entry the module does not have cannot be adopted wrongly, and a missing entry is
    the caller's concern either way.
    """
    destinations: dict[str, torch.Tensor] = dict(module.named_parameters())
    destinations.update(dict(module.named_buffers()))
    for key, incoming in state_dict.items():
        destination = destinations.get(key)
        if destination is None or not isinstance(incoming, torch.Tensor):
            continue
        if incoming.dtype != destination.dtype:
            return False
    return True


def _assigning_load_state_dict(
    self: torch.nn.Module,
    state_dict: Mapping[str, Any],
    strict: bool = True,
    assign: bool = False,
) -> Any:
    """``load_state_dict`` that adopts mmap-backed tensors when doing so is byte-identical to copying."""
    if not assign and getattr(_hook_state, "active", False) and _all_dtypes_match(self, state_dict):
        return _original_load_state_dict(self, state_dict, strict=strict, assign=True)
    return _original_load_state_dict(self, state_dict, strict=strict, assign=assign)


@contextlib.contextmanager
def zero_copy_state_dict_assignment() -> Generator[None, None, None]:
    """Scope within which module loads adopt (rather than copy) dtype-matching state-dict tensors.

    Re-entrant and thread-local: only the calling thread's loads are affected, and nesting is safe.
    Any failure to install degrades to the ordinary copying behavior rather than raising.
    """
    already_active = getattr(_hook_state, "active", False)
    _hook_state.active = True
    installed = False
    if not already_active:
        try:
            if torch.nn.Module.load_state_dict is _original_load_state_dict:
                torch.nn.Module.load_state_dict = _assigning_load_state_dict  # type: ignore[method-assign]
                installed = True
        except Exception as hook_error:
            logger.debug(f"Zero-copy load hook not installed ({hook_error})")
    try:
        yield
    finally:
        _hook_state.active = already_active
        if installed:
            torch.nn.Module.load_state_dict = _original_load_state_dict  # type: ignore[method-assign]
