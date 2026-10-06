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

``checked_file_mappings`` guards the same loads against a file mapping the host could not commit,
turning what would be an access violation into a ``RuntimeError`` the loader can report. The assignment
scope enters it itself, so every adopting load is guarded and the guard sees every mapping the load makes.

A weight adopted from a mapping is tagged with its data pointer (``is_file_backed``). Registering such a
weight with the CUDA driver as pinned host memory (ComfyUI pins every weight it leaves off the device)
forces the kernel to make its pages private, which undoes the sharing above at the size of the weight; the
tag is what lets the pinning hook decline those weights and pin only memory that is already private.
"""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Callable, Generator, Mapping
from typing import Any

import torch
from loguru import logger

_hook_state = threading.local()

_original_load_state_dict = torch.nn.Module.load_state_dict

_INVALID_STORAGE_MESSAGE_FRAGMENT = "invalid python storage"
"""Fragment of torch's ``data_ptr()`` error for a storage object with no backing allocation."""

_mapping_guard_lock = threading.Lock()
_mapping_guard_depth = 0
_MISSING_ATTRIBUTE = object()
_shadowed_from_file: object = _MISSING_ATTRIBUTE
"""The ``from_file`` entry in ``torch.UntypedStorage``'s own namespace before the guard shadowed it."""
_unguarded_from_file: Callable[..., Any] | None = None
_mapping_ranges: list[tuple[int, int]] = []
"""``(base pointer, byte length)`` of every file mapping made while the guard is active; cleared at its exit."""

_FILE_BACKED_ATTR = "_hordelib_file_backed_ptr"
"""Attribute holding the data pointer a weight had when it was adopted from a file mapping."""


def _data_pointer(tensor: torch.Tensor) -> int | None:
    try:
        return tensor.data_ptr()
    except Exception:
        return None


def _in_mapping_range(pointer: int | None) -> bool:
    if pointer is None or pointer == 0:
        return False
    return any(base <= pointer < base + length for base, length in _mapping_ranges)


def mark_file_backed(tensor: torch.Tensor) -> None:
    """Tag ``tensor`` as a view over a checkpoint file mapping at its current data pointer."""
    pointer = _data_pointer(tensor)
    if pointer is None:
        return
    try:
        setattr(tensor, _FILE_BACKED_ATTR, pointer)
    except AttributeError:
        pass


def is_file_backed(tensor: torch.Tensor) -> bool:
    """Whether ``tensor``'s data is still the checkpoint file mapping it was adopted from.

    The tag records the data pointer at adoption, so a weight whose data was later replaced by a private
    copy (a device round trip that copied back) no longer counts, and one restored to its origin counts again.
    """
    tag = getattr(tensor, _FILE_BACKED_ATTR, None)
    return tag is not None and tag == _data_pointer(tensor)


def _tag_file_backed_destinations(module: torch.nn.Module, state_dict: Mapping[str, Any]) -> None:
    """After an adopting load, tag each destination weight whose data is a state-dict tensor from a mapping."""
    destinations: dict[str, torch.Tensor] = dict(module.named_parameters())
    destinations.update(dict(module.named_buffers()))
    for key, incoming in state_dict.items():
        destination = destinations.get(key)
        if destination is None or not isinstance(incoming, torch.Tensor):
            continue
        pointer = _data_pointer(incoming)
        if not (is_file_backed(incoming) or _in_mapping_range(pointer)):
            continue
        if _data_pointer(destination) == pointer:
            mark_file_backed(destination)


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
        assign = True
    result = _original_load_state_dict(self, state_dict, strict=strict, assign=assign)
    if assign:
        _tag_file_backed_destinations(self, state_dict)
    return result


@contextlib.contextmanager
def zero_copy_state_dict_assignment() -> Generator[None, None, None]:
    """Scope within which module loads adopt (rather than copy) dtype-matching state-dict tensors.

    Re-entrant and thread-local: only the calling thread's loads are affected, and nesting is safe.
    Any failure to install degrades to the ordinary copying behavior rather than raising. The scope also
    holds ``checked_file_mappings`` so the mappings a load makes are known when its weights are tagged.
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
        with checked_file_mappings():
            yield
    finally:
        _hook_state.active = already_active
        if installed:
            torch.nn.Module.load_state_dict = _original_load_state_dict  # type: ignore[method-assign]


def _checked_from_file(filename: Any, shared: bool = False, nbytes: int = 0) -> Any:
    """``torch.UntypedStorage.from_file`` that refuses a storage with no backing mapping.

    Raises:
        RuntimeError: The mapping came back without a data pointer, so the host could not commit it.
    """
    if _unguarded_from_file is None:
        raise RuntimeError("checked_file_mappings is not active; the unguarded from_file is unknown")
    storage = _unguarded_from_file(filename, shared, nbytes)
    commit_failure = RuntimeError(
        f"The host could not commit a {nbytes}-byte mapping of {filename}: "
        "torch.UntypedStorage.from_file returned a storage with no data pointer",
    )
    try:
        data_pointer = storage.data_ptr()
    except RuntimeError as data_pointer_error:
        if _INVALID_STORAGE_MESSAGE_FRAGMENT in str(data_pointer_error):
            raise commit_failure from data_pointer_error
        raise
    if nbytes > 0 and data_pointer == 0:
        raise commit_failure
    if nbytes > 0:
        _mapping_ranges.append((data_pointer, nbytes))
    return storage


@contextlib.contextmanager
def checked_file_mappings() -> Generator[None, None, None]:
    """Context manager that makes a failed ``torch.UntypedStorage.from_file`` mapping raise.

    On Windows torch maps a non-shared file copy-on-write, which charges the whole file to system commit,
    and at the commit ceiling ``from_file`` returns an invalid storage without raising, which safetensors
    then slices into a read at a NULL base. Inside this scope each mapping's data pointer is checked and a
    missing one raises ``RuntimeError`` naming the file and size, so the load fails and the process lives.

    The patch is process-wide (it only adds a check) and reference-counted: nested or concurrent scopes
    keep it installed until the last one exits, which restores the previous ``from_file``.
    """
    global _mapping_guard_depth, _shadowed_from_file, _unguarded_from_file
    with _mapping_guard_lock:
        if _mapping_guard_depth == 0:
            _shadowed_from_file = vars(torch.UntypedStorage).get("from_file", _MISSING_ATTRIBUTE)
            _unguarded_from_file = torch.UntypedStorage.from_file
            torch.UntypedStorage.from_file = staticmethod(_checked_from_file)  # type: ignore[method-assign,assignment]
        _mapping_guard_depth += 1
    try:
        yield
    finally:
        with _mapping_guard_lock:
            _mapping_guard_depth -= 1
            if _mapping_guard_depth == 0:
                _mapping_ranges.clear()
                if _shadowed_from_file is _MISSING_ATTRIBUTE:
                    del torch.UntypedStorage.from_file
                else:
                    torch.UntypedStorage.from_file = _shadowed_from_file  # type: ignore[method-assign,assignment]
                _shadowed_from_file = _MISSING_ATTRIBUTE
                _unguarded_from_file = None
