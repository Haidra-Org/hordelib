"""Unit tests for the split-role ad-hoc manager primitives (no CivitAI network, no GPU).

These exercise the four primitives the reader/writer split relies on, in isolation:

* atomic reference saves (temp file plus :func:`os.replace`, so a reader never sees a torn file);
* :meth:`CivitaiAdhocModelManager.refresh_reference_if_stale` (cheap stat-gated reload);
* ``read_only`` mode (a pure reader that never writes, downloads, or evicts); and
* eviction pins (reference keys every eviction and deletion path must skip).

Managers are built with ``object.__new__`` and wired with only the attributes each path needs,
mirroring the existing ad-hoc unit tests, so no real construction (and therefore no network or
shared reference file) is involved.
"""

from __future__ import annotations

import json
import os
import threading
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import pytest

import hordelib.model_manager.civitai_adhoc as civitai_adhoc
from hordelib.model_manager.civitai_adhoc import (
    DownloadTarget,
    ReadOnlyModelManagerError,
)
from hordelib.model_manager.civitai_records import (
    HordeLoraModelRecord,
    HordeTextualInversionModelRecord,
    LoraVersionEntry,
)
from hordelib.model_manager.lora import LoraModelManager
from hordelib.model_manager.ti import TextualInversionModelManager


def _make_lora_record(key: str, version_id: int, *, size_mb: float = 100, adhoc: bool = True) -> HordeLoraModelRecord:
    """Return a single-version LoRA record with a deterministic last-used stamp."""
    version = LoraVersionEntry(
        filename=f"{key}_{version_id}.safetensors",
        url="http://example/x",
        version_id=str(version_id),
        lora_key=key,
        sha256="deadbeef",
        adhoc=adhoc,
        size_mb=size_mb,
        triggers=[],
        base_model="SD 1.5",
        availability="Public",
        last_used=f"2026-01-01 00:00:{version_id % 60:02d}",
    )
    return HordeLoraModelRecord(
        name=key,
        civitai_id=version_id,
        orig_name=key,
        nsfw=False,
        versions={str(version_id): version},
    )


def _make_ti_record(key: str, civitai_id: int, *, adhoc: bool = True) -> HordeTextualInversionModelRecord:
    """Return a textual inversion record for reader/writer round-trip tests."""
    return HordeTextualInversionModelRecord(
        name=key,
        civitai_id=civitai_id,
        orig_name=key,
        filename=f"{civitai_id}.safetensors",
        url="http://example/x",
        sha256="deadbeef",
        size_kb=24,
        base_model="SD 1.5",
        version_id=civitai_id,
        adhoc=adhoc,
        last_used=f"2026-01-01 00:00:{civitai_id % 60:02d}",
    )


def _wire_persistence_shell(manager: Any, tmp_path: Path, db_path: Path, *, read_only: bool) -> None:
    """Wire the minimal attributes a manager shell needs for load/save/reload/refresh."""
    manager.model_folder_path = str(tmp_path)
    manager.models_db_path = db_path
    manager.models_db_name = str(manager.MODEL_CATEGORY)
    manager.model_reference = {}
    manager.available_models = []
    manager._mutex = threading.Lock()
    manager._file_mutex = threading.Lock()
    manager._file_lock = nullcontext()
    manager._using_multiprocessing = False
    manager.read_only = read_only
    manager.eviction_pins = set()
    manager._reference_stamp = None
    manager._index_ids = {}
    manager._index_orig_names = {}
    manager._index_version_ids = {}
    manager.min_free_disk_mb = 0
    manager.max_adhoc_disk = 100_000
    manager._max_top_disk = 100_000


def _lora_shell(tmp_path: Path, db_path: Path, *, read_only: bool = False) -> LoraModelManager:
    """Build a LoRA manager shell wired for persistence primitives."""
    manager = object.__new__(LoraModelManager)
    _wire_persistence_shell(manager, tmp_path, db_path, read_only=read_only)
    return manager


def _ti_shell(tmp_path: Path, db_path: Path, *, read_only: bool = False) -> TextualInversionModelManager:
    """Build a TI manager shell wired for persistence primitives."""
    manager = object.__new__(TextualInversionModelManager)
    _wire_persistence_shell(manager, tmp_path, db_path, read_only=read_only)
    return manager


class TestAtomicSave:
    def test_save_uses_temp_then_replace(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A save writes a sibling temp file and renames it onto the target; the target is valid JSON."""
        db_path = tmp_path / "lora.json"
        manager = _lora_shell(tmp_path, db_path)
        manager.model_reference = {"a": _make_lora_record("a", 1)}

        real_replace = os.replace
        replace_calls: list[tuple[str, str]] = []

        def _spy_replace(src: Any, dst: Any) -> None:
            replace_calls.append((str(src), str(dst)))
            real_replace(src, dst)

        monkeypatch.setattr(civitai_adhoc.os, "replace", _spy_replace)

        manager.save_reference_to_disk()

        assert len(replace_calls) == 1, "an atomic save must go through exactly one os.replace"
        src, dst = replace_calls[0]
        assert dst == str(db_path)
        assert ".tmp-" in src and src != str(db_path), "the payload must be written to a sibling temp file first"
        assert json.loads(db_path.read_text()) == {"a": manager.model_reference["a"].model_dump(mode="json")}

    def test_interleaved_saves_never_leave_partial_json(self, tmp_path: Path) -> None:
        """A reader polling the file during concurrent saves always sees complete, valid JSON."""
        db_path = tmp_path / "lora.json"
        manager = _lora_shell(tmp_path, db_path)
        manager.model_reference = {f"lora{i}": _make_lora_record(f"lora{i}", 1000 + i) for i in range(20)}
        manager.save_reference_to_disk()  # ensure the file exists before the reader starts

        stop = threading.Event()
        reader_errors: list[Exception] = []

        def _reader() -> None:
            while not stop.is_set():
                try:
                    parsed = json.loads(db_path.read_text())
                    assert isinstance(parsed, dict)
                except json.JSONDecodeError as decode_error:
                    reader_errors.append(decode_error)
                    return

        def _writer() -> None:
            for _ in range(50):
                manager.save_reference_to_disk()

        reader = threading.Thread(target=_reader)
        reader.start()
        _writer()
        stop.set()
        reader.join(timeout=5)

        assert not reader_errors, f"a reader must never observe a partially written reference: {reader_errors}"
        assert isinstance(json.loads(db_path.read_text()), dict)


class TestOrphanTempFilePrune:
    def test_load_prunes_only_aged_temp_files(self, tmp_path: Path) -> None:
        """load_model_database removes a stale orphan temp but spares a fresh one a live writer may own.

        A fresh temp file may belong to the download process still transferring bytes when a sibling
        manager is constructed on the same host; deleting it would fail that in-flight fetch. Only a
        temp older than the age gate (a crashed writer's leak) is pruned.
        """
        db_path = tmp_path / "lora.json"
        manager = _lora_shell(tmp_path, db_path)

        fresh_temp = tmp_path / "fresh.safetensors.tmp-aaaaaaaa"
        stale_temp = tmp_path / "stale.safetensors.tmp-bbbbbbbb"
        fresh_temp.write_bytes(b"in-flight")
        stale_temp.write_bytes(b"crashed")
        aged = time.time() - (civitai_adhoc.ORPHAN_TEMP_MIN_AGE_SECONDS + 60)
        os.utime(stale_temp, (aged, aged))

        manager.load_model_database()

        assert fresh_temp.exists(), "a fresh temp (a possibly live writer's in-flight file) must survive the prune"
        assert not stale_temp.exists(), "a temp older than the age gate (a crashed writer's leak) must be pruned"


class TestRefreshReferenceIfStale:
    def test_unchanged_file_returns_false_without_reparsing(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When the file is unchanged since the last load, refresh returns False and does not reparse."""
        db_path = tmp_path / "lora.json"
        writer = _lora_shell(tmp_path, db_path)
        writer.model_reference = {"a": _make_lora_record("a", 1)}
        writer.save_reference_to_disk()

        reader = _lora_shell(tmp_path, db_path, read_only=True)
        reader.reload_reference_from_disk()  # seeds the stamp and the in-memory view

        def _must_not_reparse(_raw: dict) -> dict:
            raise AssertionError("refresh must not reparse an unchanged reference")

        monkeypatch.setattr(reader, "_deserialise_reference", _must_not_reparse)

        assert reader.refresh_reference_if_stale() is False

    def test_external_modification_triggers_reload(self, tmp_path: Path) -> None:
        """An out-of-band write (another process) makes refresh return True and expose the new record."""
        db_path = tmp_path / "lora.json"
        writer = _lora_shell(tmp_path, db_path)
        writer.model_reference = {"a": _make_lora_record("a", 1)}
        writer.save_reference_to_disk()

        reader = _lora_shell(tmp_path, db_path, read_only=True)
        reader.reload_reference_from_disk()
        assert "b" not in reader.model_reference

        writer.model_reference["b"] = _make_lora_record("b", 2)
        writer.save_reference_to_disk()

        assert reader.refresh_reference_if_stale() is True
        assert "b" in reader.model_reference

    def test_reload_updates_available_models(self, tmp_path: Path) -> None:
        """A reload (via refresh) refreshes ``available_models`` alongside the reference and indices.

        Without this a reader that picks up a writer's new record keeps a permanently stale
        ``available_models`` list.
        """
        db_path = tmp_path / "lora.json"
        writer = _lora_shell(tmp_path, db_path)
        writer.model_reference = {"a": _make_lora_record("a", 1)}
        writer.save_reference_to_disk()

        reader = _lora_shell(tmp_path, db_path, read_only=True)
        reader.load_model_database()
        assert "b" not in reader.available_models

        writer.model_reference["b"] = _make_lora_record("b", 2)
        writer.save_reference_to_disk()

        assert reader.refresh_reference_if_stale() is True
        assert "b" in reader.available_models
        assert set(reader.available_models) == {"a", "b"}

    def test_no_false_positive_right_after_own_save(self, tmp_path: Path) -> None:
        """A manager that just saved must not see its own write as a stale-triggering change."""
        db_path = tmp_path / "lora.json"
        manager = _lora_shell(tmp_path, db_path)
        manager.model_reference = {"a": _make_lora_record("a", 1)}
        manager.save_reference_to_disk()

        assert manager.refresh_reference_if_stale() is False


class TestReadOnlyMode:
    def test_construction_load_writes_nothing(self, tmp_path: Path) -> None:
        """Loading an existing reference in read-only mode must not touch the file at all."""
        db_path = tmp_path / "lora.json"
        writer = _lora_shell(tmp_path, db_path)
        writer.model_reference = {"a": _make_lora_record("a", 1)}
        writer.save_reference_to_disk()

        before_bytes = db_path.read_bytes()
        before_mtime = db_path.stat().st_mtime_ns

        reader = _lora_shell(tmp_path, db_path, read_only=True)
        reader.load_model_database()

        assert "a" in reader.model_reference, "the reader must still load the reference into memory"
        assert db_path.read_bytes() == before_bytes
        assert db_path.stat().st_mtime_ns == before_mtime

    def test_is_model_available_does_not_write(self, tmp_path: Path) -> None:
        """An availability hit refreshes last-used in memory only; the file stays byte-for-byte identical."""
        db_path = tmp_path / "lora.json"
        writer = _lora_shell(tmp_path, db_path)
        writer.model_reference = {"a": _make_lora_record("a", 1)}
        writer.save_reference_to_disk()

        reader = _lora_shell(tmp_path, db_path, read_only=True)
        reader.load_model_database()
        before_bytes = db_path.read_bytes()

        assert reader.is_model_available("a") is True
        assert db_path.read_bytes() == before_bytes

    def test_mutating_paths_raise(self, tmp_path: Path) -> None:
        """Every write, download, and eviction entry point raises in read-only mode."""
        db_path = tmp_path / "lora.json"
        writer = _lora_shell(tmp_path, db_path)
        writer.model_reference = {"a": _make_lora_record("a", 1)}
        writer.save_reference_to_disk()

        reader = _lora_shell(tmp_path, db_path, read_only=True)
        reader.load_model_database()
        record = _make_lora_record("b", 2)

        with pytest.raises(ReadOnlyModelManagerError):
            reader.save_reference_to_disk()
        with pytest.raises(ReadOnlyModelManagerError):
            reader._enqueue_download(record)
        with pytest.raises(ReadOnlyModelManagerError):
            reader.enforce_adhoc_budget()
        with pytest.raises(ReadOnlyModelManagerError):
            reader.evict_adhoc_for_free_space()
        with pytest.raises(ReadOnlyModelManagerError):
            reader.delete_unused_models()
        with pytest.raises(ReadOnlyModelManagerError):
            reader._delete_model_entry("a", "1")
        with pytest.raises(ReadOnlyModelManagerError):
            reader.fetch_adhoc_lora("anything")

    def test_writer_download_becomes_visible_after_refresh(self, tmp_path: Path) -> None:
        """A record a writer persists is invisible to a reader until it refreshes, then visible."""
        db_path = tmp_path / "lora.json"
        writer = _lora_shell(tmp_path, db_path)
        writer.model_reference = {"a": _make_lora_record("a", 1)}
        writer.save_reference_to_disk()

        reader = _lora_shell(tmp_path, db_path, read_only=True)
        reader.load_model_database()
        assert reader.is_model_available("b") is False

        writer.model_reference["b"] = _make_lora_record("b", 2)
        writer.save_reference_to_disk()

        assert reader.refresh_reference_if_stale() is True
        assert reader.is_model_available("b") is True


def _make_eviction_manager(*, max_adhoc_disk: int, count: int, size_mb: float) -> LoraModelManager:
    """Build a LoRA shell seeded with *count* ad-hoc records for eviction/pin tests."""
    manager = object.__new__(LoraModelManager)
    manager.read_only = False
    manager.eviction_pins = set()
    manager._mutex = threading.Lock()
    manager.max_adhoc_disk = max_adhoc_disk
    manager._max_top_disk = max_adhoc_disk
    manager.min_free_disk_mb = 1024
    manager.model_reference = {}
    manager._index_ids = {}
    manager._index_version_ids = {}
    manager._index_orig_names = {}
    for i in range(count):
        record = _make_lora_record(f"lora{i}", 1000 + i, size_mb=size_mb)
        manager.model_reference[record.name] = record
    return manager


class TestEvictionPins:
    def test_pinned_survive_budget_eviction_unpinned_evicted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Budget eviction chooses the oldest unpinned entry and never a pinned one."""
        manager = _make_eviction_manager(max_adhoc_disk=1024, count=5, size_mb=300)  # 1500 MB, over budget
        monkeypatch.setattr(manager, "_delete_weight_files", lambda filename: None)
        monkeypatch.setattr(manager, "save_reference_to_disk", lambda: None)
        manager.set_eviction_pins({"lora0", "lora1", "lora2", "lora3"})  # only lora4 is evictable

        manager.enforce_adhoc_budget()

        assert "lora4" not in manager.model_reference, "the sole unpinned entry must be the eviction victim"
        for pinned in ("lora0", "lora1", "lora2", "lora3"):
            assert pinned in manager.model_reference, "pinned entries must survive eviction"

    def test_all_pinned_yields_without_deleting_or_raising(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When every candidate is pinned, eviction yields: no deletion, no raise, a logged note."""
        manager = _make_eviction_manager(max_adhoc_disk=1024, count=3, size_mb=300)  # 900 MB
        manager.max_adhoc_disk = 512  # force over-budget so eviction is attempted
        deleted: list[str] = []
        monkeypatch.setattr(manager, "_delete_weight_files", lambda filename: deleted.append(filename))
        monkeypatch.setattr(manager, "save_reference_to_disk", lambda: None)
        manager.set_eviction_pins({"lora0", "lora1", "lora2"})

        yields: list[int] = []
        real_note = manager._note_eviction_yielded

        def _spy_note() -> None:
            yields.append(1)
            real_note()

        monkeypatch.setattr(manager, "_note_eviction_yielded", _spy_note)

        manager.enforce_adhoc_budget()  # must not raise

        assert deleted == [], "no file may be deleted when every candidate is pinned"
        assert len(manager.model_reference) == 3, "all entries must survive when all are pinned"
        assert yields, "the all-pinned yield must be noted"

    def test_pins_protect_free_space_eviction(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A pinned entry is never evicted to reclaim disk space, even when the floor is unmet."""
        manager = _make_eviction_manager(max_adhoc_disk=10240, count=2, size_mb=300)
        monkeypatch.setattr(manager, "_delete_weight_files", lambda filename: None)
        monkeypatch.setattr(manager, "save_reference_to_disk", lambda: None)
        monkeypatch.setattr(manager, "disk_free_mb", lambda: 200.0)  # below the 1024 floor
        manager.set_eviction_pins({"lora0", "lora1"})

        solved = manager.evict_adhoc_for_free_space()

        assert solved is False, "no unpinned entry can be evicted, so the floor cannot be met"
        assert len(manager.model_reference) == 2, "pinned entries must survive free-space eviction"

    def test_clearing_pins_restores_eviction(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Clearing pins lets a previously protected entry be evicted again."""
        manager = _make_eviction_manager(max_adhoc_disk=512, count=2, size_mb=300)  # 600 MB, over budget
        monkeypatch.setattr(manager, "_delete_weight_files", lambda filename: None)
        monkeypatch.setattr(manager, "save_reference_to_disk", lambda: None)
        manager.set_eviction_pins({"lora0", "lora1"})
        manager.enforce_adhoc_budget()
        assert len(manager.model_reference) == 2, "all pinned, nothing evicted"

        manager.set_eviction_pins(set())
        manager.enforce_adhoc_budget()
        assert len(manager.model_reference) < 2, "with pins cleared, the oldest entry is evicted"


class TestTextualInversionInherits:
    def test_ti_read_only_blocks_writes(self, tmp_path: Path) -> None:
        """The TI manager inherits the read-only contract: writes and downloads raise."""
        db_path = tmp_path / "ti.json"
        writer = _ti_shell(tmp_path, db_path)
        writer.model_reference = {"neg": _make_ti_record("neg", 100)}
        writer.save_reference_to_disk()

        reader = _ti_shell(tmp_path, db_path, read_only=True)
        reader.load_model_database()

        assert reader.is_local_model("neg") is True
        with pytest.raises(ReadOnlyModelManagerError):
            reader.save_reference_to_disk()
        with pytest.raises(ReadOnlyModelManagerError):
            reader.fetch_adhoc_ti("neg")
        with pytest.raises(ReadOnlyModelManagerError):
            reader._delete_model_entry("neg", None)

    def test_ti_refresh_picks_up_external_change(self, tmp_path: Path) -> None:
        """The TI manager inherits stat-gated refresh: an external write becomes visible after refresh."""
        db_path = tmp_path / "ti.json"
        writer = _ti_shell(tmp_path, db_path)
        writer.model_reference = {"neg": _make_ti_record("neg", 100)}
        writer.save_reference_to_disk()

        reader = _ti_shell(tmp_path, db_path, read_only=True)
        reader.load_model_database()
        assert reader.refresh_reference_if_stale() is False

        writer.model_reference["pos"] = _make_ti_record("pos", 200)
        writer.save_reference_to_disk()

        assert reader.refresh_reference_if_stale() is True
        assert reader.is_local_model("pos") is True


def test_ensure_room_for_download_read_only_raises(tmp_path: Path) -> None:
    """The download-space check also refuses in read-only mode (belt-and-braces on the download path)."""
    db_path = tmp_path / "lora.json"
    reader = _lora_shell(tmp_path, db_path, read_only=True)
    target = DownloadTarget(filename="x.safetensors", url="http://x", size_mb=1.0)
    with pytest.raises(ReadOnlyModelManagerError):
        reader._ensure_room_for_download(target)
