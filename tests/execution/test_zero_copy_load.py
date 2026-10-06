"""Tests for the zero-copy (assign-on-load) state-dict adoption used by the checkpoint loader.

The contract: inside the context manager, a dtype-matching load adopts the incoming tensors (module
parameters share the source storage: what makes mmap-backed checkpoint weights stay shared across
processes); a dtype-mismatched load falls back to the ordinary copy (preserving cast semantics); and
outside the context manager nothing changes.

The mapping guard's contract: inside ``checked_file_mappings`` a ``from_file`` storage without a data
pointer raises a ``RuntimeError`` naming the file, a valid storage passes through unchanged, and the
original ``from_file`` is back after the scope.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from hordelib.execution.zero_copy_load import (
    checked_file_mappings,
    is_file_backed,
    zero_copy_state_dict_assignment,
)

_INVALID_STORAGE_MESSAGE = "Attempted to access the data pointer on an invalid python storage."


def _module() -> torch.nn.Module:
    module = torch.nn.Linear(4, 4, bias=False)
    return module.to(torch.float16)


class _FakeStorage:
    """Stand-in for the storage ``from_file`` returns, with a scripted ``data_ptr``."""

    def __init__(self, *, size: int, data_pointer: int | None, data_pointer_error: str | None = None) -> None:
        self._size = size
        self._data_pointer = data_pointer
        self._data_pointer_error = data_pointer_error

    def data_ptr(self) -> int:
        if self._data_pointer_error is not None:
            raise RuntimeError(self._data_pointer_error)
        assert self._data_pointer is not None
        return self._data_pointer

    def nbytes(self) -> int:
        return self._size


def _install_fake_from_file(monkeypatch: pytest.MonkeyPatch, storage: _FakeStorage) -> None:
    def _fake_from_file(filename: str, shared: bool = False, nbytes: int = 0) -> _FakeStorage:
        return storage

    monkeypatch.setattr(torch.UntypedStorage, "from_file", _fake_from_file)


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


def test_invalid_mapping_raises_naming_the_file(monkeypatch: pytest.MonkeyPatch) -> None:
    """A storage whose data_ptr raises torch's invalid-storage error becomes a RuntimeError naming the file."""
    size = 19 * 1024**3
    _install_fake_from_file(
        monkeypatch,
        _FakeStorage(size=size, data_pointer=None, data_pointer_error=_INVALID_STORAGE_MESSAGE),
    )

    with checked_file_mappings(), pytest.raises(RuntimeError, match="could not commit") as raised:
        torch.UntypedStorage.from_file("checkpoints/big_model.safetensors", False, size)

    assert "checkpoints/big_model.safetensors" in str(raised.value)
    assert str(size) in str(raised.value)
    assert isinstance(raised.value.__cause__, RuntimeError)


def test_null_data_pointer_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """A storage that reports a NULL pointer for a non-empty mapping is refused the same way."""
    _install_fake_from_file(monkeypatch, _FakeStorage(size=1024, data_pointer=0))

    with checked_file_mappings(), pytest.raises(RuntimeError, match="model.safetensors"):
        torch.UntypedStorage.from_file("model.safetensors", False, 1024)


def test_unrelated_data_pointer_error_propagates_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    """A data_ptr failure other than the invalid-storage one is not relabelled as a commit failure."""
    _install_fake_from_file(monkeypatch, _FakeStorage(size=8, data_pointer=None, data_pointer_error="other"))

    with checked_file_mappings(), pytest.raises(RuntimeError, match="^other$"):
        torch.UntypedStorage.from_file("model.safetensors", False, 8)


def test_valid_mapping_passes_through_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    """A storage with a data pointer is returned as-is."""
    storage = _FakeStorage(size=1024, data_pointer=0x1000)
    _install_fake_from_file(monkeypatch, storage)

    with checked_file_mappings():
        returned = torch.UntypedStorage.from_file("model.safetensors", False, 1024)

    assert returned is storage


def test_real_mapping_passes_through(tmp_path: Path) -> None:
    """A real mapping of a small file is valid inside the scope and reads the file's bytes."""
    file_path = tmp_path / "tiny.bin"
    file_path.write_bytes(bytes(range(16)))

    with checked_file_mappings():
        storage = torch.UntypedStorage.from_file(str(file_path), False, 16)

    assert storage.data_ptr() != 0
    assert storage.tolist() == list(range(16))
    del storage


def test_safetensors_load_goes_through_the_guard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """safetensors' own mapping call reaches the guarded from_file and still loads the tensor."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    file_path = tmp_path / "tiny.safetensors"
    save_file({"weight": torch.arange(8, dtype=torch.float16)}, str(file_path))
    unpatched_from_file = torch.UntypedStorage.from_file
    mapped_files: list[str] = []

    def _recording_from_file(filename: str, shared: bool = False, nbytes: int = 0) -> object:
        mapped_files.append(str(filename))
        return unpatched_from_file(filename, shared, nbytes)

    monkeypatch.setattr(torch.UntypedStorage, "from_file", _recording_from_file)

    with checked_file_mappings(), safe_open(str(file_path), framework="pt", device="cpu") as opened:
        loaded = opened.get_tensor("weight")

    assert mapped_files == [str(file_path)]
    assert torch.equal(loaded, torch.arange(8, dtype=torch.float16))
    del loaded


def test_a_weight_adopted_from_a_mapping_is_tagged_file_backed(tmp_path: Path) -> None:
    """A parameter adopted from a safetensors mapping is file-backed; a private tensor or copy is not."""
    from safetensors.torch import load_file, save_file

    file_path = tmp_path / "tiny.safetensors"
    save_file({"weight": torch.ones(4, 4, dtype=torch.float16)}, str(file_path))
    module = _module()
    private = _module()

    with zero_copy_state_dict_assignment():
        mapped = load_file(str(file_path), device="cpu")
        module.load_state_dict(mapped)
        private.load_state_dict({"weight": torch.ones(4, 4, dtype=torch.float16)})

    assert module.weight.data_ptr() == mapped["weight"].data_ptr()
    assert is_file_backed(module.weight)
    assert not is_file_backed(private.weight)
    origin = module.weight.data
    module.weight.data = origin.clone()
    assert not is_file_backed(module.weight)
    module.weight.data = origin
    assert is_file_backed(module.weight)
    del mapped, origin


def test_a_tag_survives_a_later_adopting_load_of_the_same_tensor(tmp_path: Path) -> None:
    """A tagged weight adopted again by another module, after the mapping scope closed, stays tagged."""
    from safetensors.torch import load_file, save_file

    file_path = tmp_path / "tiny.safetensors"
    save_file({"weight": torch.ones(4, 4, dtype=torch.float16)}, str(file_path))
    with zero_copy_state_dict_assignment():
        mapped = load_file(str(file_path), device="cpu")
        first = _module()
        first.load_state_dict(mapped)
    second = _module()
    with zero_copy_state_dict_assignment():
        second.load_state_dict({"weight": first.weight})
    assert is_file_backed(second.weight)
    del mapped


def test_mapping_guard_is_restored_after_scope_and_error() -> None:
    """Nested scopes keep the guard until the outermost exit, which restores from_file even on an error."""
    original_namespace_entry = vars(torch.UntypedStorage).get("from_file")

    with pytest.raises(ValueError), checked_file_mappings():
        with checked_file_mappings():
            guarded = vars(torch.UntypedStorage).get("from_file")
            assert guarded is not None
        assert vars(torch.UntypedStorage).get("from_file") is guarded
        raise ValueError("load failed")

    assert vars(torch.UntypedStorage).get("from_file") is original_namespace_entry
