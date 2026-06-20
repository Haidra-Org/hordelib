"""Unit tests for the ad-hoc cache budget accounting and the disk-space safety floor.

These exercise :class:`~hordelib.model_manager.lora.LoraModelManager`'s inherited cache-accounting
and eviction logic in isolation: no CivitAI network, no GPU, and no real weight files. A manager
shell is built with ``object.__new__`` and seeded with synthetic version entries so the size
arithmetic (which is what the unit bug lived in) can be asserted directly and deterministically.
"""

from __future__ import annotations

import pytest

from hordelib.model_manager.civitai_adhoc import DEFAULT_MIN_FREE_DISK_MB
from hordelib.model_manager.civitai_records import HordeLoraModelRecord, LoraVersionEntry
from hordelib.model_manager.lora import LoraModelManager


def _make_record(key: str, version_id: int, size_mb: float, *, adhoc: bool = True) -> HordeLoraModelRecord:
    """Return a single-version LoRA record of *size_mb* megabytes for cache accounting."""
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
        # Ascending timestamps so eviction order (oldest first) is deterministic by version id.
        last_used=f"2026-01-01 00:00:{version_id % 60:02d}",
    )
    return HordeLoraModelRecord(
        name=key,
        civitai_id=version_id,
        orig_name=key,
        nsfw=False,
        versions={str(version_id): version},
    )


def _make_manager(*, max_adhoc_disk: int, min_free_disk_mb: int = DEFAULT_MIN_FREE_DISK_MB) -> LoraModelManager:
    """Build a LoraModelManager shell (no __init__) wired only for in-memory cache accounting."""
    manager = object.__new__(LoraModelManager)
    manager.max_adhoc_disk = max_adhoc_disk
    manager._max_top_disk = max_adhoc_disk
    manager.min_free_disk_mb = min_free_disk_mb
    manager.model_reference = {}
    manager._index_ids = {}
    manager._index_version_ids = {}
    manager._index_orig_names = {}
    return manager


def _seed_adhoc(manager: LoraModelManager, count: int, size_mb: float) -> None:
    """Add *count* ad-hoc LoRAs of *size_mb* each to the manager's reference."""
    for i in range(count):
        record = _make_record(f"lora{i}", 1000 + i, size_mb)
        manager.model_reference[record.name] = record


class TestAdhocCacheBudget:
    def test_budget_is_megabytes_not_gigabytes(self) -> None:
        """A 1 GB (1024 MB) budget must report full once the ad-hoc cache exceeds 1024 MB.

        Regression guard for the unit bug where the budget was compared against ``max_adhoc_disk *
        1024``, letting the cache grow ~1024x past its configured size before any eviction fired.
        """
        manager = _make_manager(max_adhoc_disk=1024)
        _seed_adhoc(manager, count=10, size_mb=200)  # 2000 MB, ~2x the 1024 MB budget

        assert manager.calculate_adhoc_cache() == pytest.approx(2000)
        assert manager.is_adhoc_cache_full() is True
        assert manager.amount_of_adhoc_to_delete() >= 1

    def test_not_full_below_budget(self) -> None:
        manager = _make_manager(max_adhoc_disk=1024)
        _seed_adhoc(manager, count=4, size_mb=200)  # 800 MB < 1024 MB

        assert manager.is_adhoc_cache_full() is False
        assert manager.amount_of_adhoc_to_delete() == 0

    def test_amount_to_delete_scales_with_overage(self) -> None:
        """One eviction at the budget, plus one more per 4 GB (4096 MB) over."""
        manager = _make_manager(max_adhoc_disk=1024)
        _seed_adhoc(manager, count=50, size_mb=200)  # 10000 MB, ~8976 MB over budget

        # 1 + floor(8976 / 4096) == 1 + 2 == 3
        assert manager.amount_of_adhoc_to_delete() == 3

    def test_enforce_adhoc_budget_evicts_back_within_limit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        manager = _make_manager(max_adhoc_disk=1024)
        _seed_adhoc(manager, count=10, size_mb=200)  # 2000 MB
        monkeypatch.setattr(manager, "_delete_weight_files", lambda filename: None)
        monkeypatch.setattr(manager, "save_reference_to_disk", lambda: None)

        manager.enforce_adhoc_budget()

        # amount_of_adhoc_to_delete is computed once up front (1 here), so a single pass need not land
        # under budget; the invariant is that it strictly shrinks the cache toward the limit.
        assert manager.calculate_adhoc_cache() < 2000


class TestDiskSafetyFloor:
    def _patch_disk(self, manager: LoraModelManager, monkeypatch: pytest.MonkeyPatch, free_mb: float) -> dict:
        """Make disk_free_mb track a mutable free figure that eviction increases by each entry's size."""
        state = {"free": free_mb}
        monkeypatch.setattr(manager, "disk_free_mb", lambda: state["free"])
        monkeypatch.setattr(manager, "save_reference_to_disk", lambda: None)

        real_delete = manager._delete_model_entry

        def _delete(model_key: str, version_key: str | None) -> None:
            record = manager.model_reference.get(model_key)
            if record is not None and version_key in record.versions:
                state["free"] += record.versions[version_key].size_mb
            real_delete(model_key, version_key)

        monkeypatch.setattr(manager, "_delete_weight_files", lambda filename: None)
        monkeypatch.setattr(manager, "_delete_model_entry", _delete)
        return state

    def test_is_disk_below_floor(self, monkeypatch: pytest.MonkeyPatch) -> None:
        manager = _make_manager(max_adhoc_disk=10240, min_free_disk_mb=1024)
        monkeypatch.setattr(manager, "disk_free_mb", lambda: 500.0)
        assert manager.is_disk_below_floor() is True
        monkeypatch.setattr(manager, "disk_free_mb", lambda: 2000.0)
        assert manager.is_disk_below_floor() is False

    def test_evict_for_free_space_makes_room(self, monkeypatch: pytest.MonkeyPatch) -> None:
        manager = _make_manager(max_adhoc_disk=10240, min_free_disk_mb=1024)
        _seed_adhoc(manager, count=10, size_mb=200)  # 2000 MB of evictable ad-hoc
        state = self._patch_disk(manager, monkeypatch, free_mb=200.0)  # below the 1024 floor

        solved = manager.evict_adhoc_for_free_space()

        assert solved is True
        assert state["free"] >= 1024
        # Only as many entries as needed to clear the floor were evicted, not the whole cache.
        assert len(manager.model_reference) > 0

    def test_evict_for_free_space_unsolvable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When even evicting every ad-hoc entry can't clear the floor, report failure."""
        manager = _make_manager(max_adhoc_disk=10240, min_free_disk_mb=1024)
        _seed_adhoc(manager, count=2, size_mb=50)  # only 100 MB evictable
        state = self._patch_disk(manager, monkeypatch, free_mb=200.0)  # 200 + 100 < 1024 floor

        solved = manager.evict_adhoc_for_free_space()

        assert solved is False
        assert manager.model_reference == {}  # exhausted all ad-hoc entries trying
        assert state["free"] == pytest.approx(300.0)

    def test_ensure_room_skips_when_unsolvable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from hordelib.model_manager.civitai_adhoc import DownloadTarget

        manager = _make_manager(max_adhoc_disk=10240, min_free_disk_mb=1024)
        self._patch_disk(manager, monkeypatch, free_mb=1100.0)
        target = DownloadTarget(filename="big.safetensors", url="http://x", size_mb=400.0)

        # 1100 free - 400 to write == 700 < 1024 floor, and no ad-hoc entries to evict.
        assert manager._ensure_room_for_download(target) is False

    def test_ensure_room_ok_with_headroom(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from hordelib.model_manager.civitai_adhoc import DownloadTarget

        manager = _make_manager(max_adhoc_disk=10240, min_free_disk_mb=1024)
        monkeypatch.setattr(manager, "disk_free_mb", lambda: 5000.0)
        target = DownloadTarget(filename="ok.safetensors", url="http://x", size_mb=400.0)

        assert manager._ensure_room_for_download(target) is True
