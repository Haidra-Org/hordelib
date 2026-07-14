"""Behavioural tests for the default-LoRA seeding cycle preserving the persistent ad-hoc cache.

Default seeding must merge into the reference loaded from disk rather than wiping it. These tests build
a :class:`~hordelib.model_manager.lora.LoraModelManager` shell wired only for the seed/reset/persist
cycle (no CivitAI network, no GPU, no real downloads), seed it with ad-hoc entries whose weight files
exist, run :meth:`download_default_models`, and assert the ad-hoc entries survive the whole cycle and
that a default already present on disk is not re-enqueued for download.
"""

from __future__ import annotations

import json
import os
import threading
from collections import deque
from contextlib import nullcontext
from pathlib import Path
from typing import Any
from unittest.mock import Mock

from hordelib.model_manager.civitai_records import HordeLoraModelRecord, LoraVersionEntry
from hordelib.model_manager.lora import LoraModelManager

_FAKE_SHA = "deadbeef" * 8


def _seedable_manager(tmp_path: Path) -> LoraModelManager:
    """Build a LoraModelManager shell wired for the default-seed/reset/persist cycle only."""
    manager = object.__new__(LoraModelManager)
    manager.model_folder_path = str(tmp_path / "lora")
    os.makedirs(manager.model_folder_path, exist_ok=True)
    manager.models_db_path = tmp_path / "lora.json"

    manager.read_only = False
    manager.nsfw = True
    manager.model_reference = {}
    manager._index_ids = {}
    manager._index_version_ids = {}
    manager._index_orig_names = {}
    manager.eviction_pins = set()
    manager._default_lora_ids = []

    manager._download_wait = False
    manager._thread = None
    manager._adhoc_reset_thread = None
    manager._download_threads = {}
    manager._download_queue = deque()
    manager.stop_downloading = True
    manager._stop_all_threads = False
    manager._download_generation = 0

    manager._mutex = threading.Lock()
    manager._file_mutex = threading.Lock()
    manager._download_mutex = threading.Lock()
    manager._file_lock = nullcontext()
    manager._using_multiprocessing = False
    manager._reference_stamp = None

    # Large budget so the ad-hoc cache is never over-limit and eviction never fires during the cycle.
    manager.max_adhoc_disk = 1_000_000
    manager._max_top_disk = 1_000_000
    manager.min_free_disk_mb = 0
    return manager


def _write_weight_file(manager: LoraModelManager, filename: str, sha256: str = _FAKE_SHA) -> None:
    """Create a weight file and its checksum sidecar so a presence-and-hash check passes."""
    weight_path = os.path.join(manager.model_folder_path, filename)
    with open(weight_path, "wb") as weight_file:
        weight_file.write(b"weights")
    hash_path = f"{os.path.splitext(weight_path)[0]}.sha256"
    with open(hash_path, "w", encoding="utf-8") as hash_file:
        hash_file.write(f"{sha256} *{filename}")


def _register(manager: LoraModelManager, record: HordeLoraModelRecord) -> None:
    """Add *record* to the in-memory reference and indices."""
    manager.model_reference[record.name] = record
    manager._index_ids[record.civitai_id] = record.name
    manager._index_orig_names[record.orig_name.lower()] = record.name
    for version_id in record.versions:
        manager._index_version_ids[version_id] = record.name


def _adhoc_record(key: str, civitai_id: int, version_id: str) -> HordeLoraModelRecord:
    """Return a single-version ad-hoc LoRA record whose weight filename is derived from *key*."""
    version = LoraVersionEntry(
        filename=f"{key}_{version_id}.safetensors",
        url="http://example/adhoc",
        version_id=version_id,
        lora_key=key,
        sha256=_FAKE_SHA,
        adhoc=True,
        size_mb=10,
        triggers=[],
        base_model="SD 1.5",
        availability="Public",
        last_used="2026-01-01 00:00:00",
    )
    return HordeLoraModelRecord(
        name=key,
        civitai_id=civitai_id,
        orig_name=key,
        nsfw=True,
        versions={version_id: version},
    )


def _default_metadata_item(name: str, civitai_id: int, version_id: int) -> dict[str, Any]:
    """Return a CivitAI-shaped metadata item that parses into a single-version default LoRA record."""
    return {
        "name": name,
        "id": civitai_id,
        "nsfw": False,
        "modelVersions": [
            {
                "id": version_id,
                "trainedWords": [],
                "baseModel": "SD 1.5",
                "availability": "Public",
                "files": [
                    {
                        "primary": True,
                        "name": "default.safetensors",
                        "sizeKB": 10240,
                        "downloadUrl": "http://example/default",
                        "hashes": {"SHA256": _FAKE_SHA},
                    },
                ],
            },
        ],
    }


def test_adhoc_entries_survive_default_seed_cycle(tmp_path: Path) -> None:
    """Ad-hoc entries loaded from disk survive the seed/reset/persist cycle and stay referenced.

    Regression guard for the boot-time reference wipe: seeding cleared the whole in-memory reference
    (including persistent ad-hoc entries) before the first ``save_reference_to_disk`` persisted the
    empty state, so every worker restart destroyed the warm ad-hoc cache.
    """
    manager = _seedable_manager(tmp_path)

    # A previously-seeded default and a warm ad-hoc entry, both present on disk.
    default = _adhoc_record("prior_default", 111, "1001")
    default.versions["1001"].adhoc = False
    adhoc = _adhoc_record("warm_adhoc", 222, "2002")
    _register(manager, default)
    _register(manager, adhoc)
    _write_weight_file(manager, default.versions["1001"].filename)
    _write_weight_file(manager, adhoc.versions["2002"].filename)

    # An empty curated list mirrors the production condition: no default is enqueued, so the only path
    # that touches the reference is the reset/persist cycle.
    manager._fetch_civitai_json = Mock(return_value=[])  # pyrefly: ignore

    manager.download_default_models()
    manager.wait_for_downloads(15)
    manager.wait_for_adhoc_reset(15)

    assert "warm_adhoc" in manager.model_reference
    assert os.path.exists(os.path.join(manager.model_folder_path, adhoc.versions["2002"].filename))
    assert adhoc.versions["2002"].filename not in manager.find_unused_files()

    # delete_unused_models must not reap the still-referenced ad-hoc weight file.
    deleted = manager.delete_unused_models(15)
    assert adhoc.versions["2002"].filename not in deleted
    assert os.path.exists(os.path.join(manager.model_folder_path, adhoc.versions["2002"].filename))


def test_present_default_not_reenqueued_and_reference_keeps_both(tmp_path: Path) -> None:
    """A default already on disk and referenced is not re-enqueued, and the ad-hoc entry survives.

    Also asserts the reference persisted after the cycle carries both the default and the prior ad-hoc
    entry, so the on-disk database remains a durable superset rather than a reseeded subset.
    """
    manager = _seedable_manager(tmp_path)

    item = _default_metadata_item("Curated Default", 333, 3003)
    parsed, _ = manager._parse_civitai_item(item)
    assert parsed is not None
    _register(manager, parsed)
    _write_weight_file(manager, parsed.versions["3003"].filename)

    adhoc = _adhoc_record("warm_adhoc", 222, "2002")
    _register(manager, adhoc)
    _write_weight_file(manager, adhoc.versions["2002"].filename)

    def fake_fetch(url: str) -> Any:
        if url == LoraModelManager.LORA_DEFAULTS:
            return [333]
        return {"items": [item]}

    manager._fetch_civitai_json = Mock(side_effect=fake_fetch)  # pyrefly: ignore
    enqueue_spy = Mock()
    manager._enqueue_download = enqueue_spy  # pyrefly: ignore

    manager.download_default_models()
    manager.wait_for_downloads(15)
    manager.wait_for_adhoc_reset(15)

    enqueue_spy.assert_not_called()

    saved = json.loads(manager.models_db_path.read_text())
    assert parsed.name in saved
    assert "warm_adhoc" in saved
