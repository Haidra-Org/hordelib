"""GPU-free unit tests for the MB-budgeted component cache (``hordelib.execution.component_cache``).

These pin the cache's contract independently of the loader: LRU recency and eviction order, budget fitting,
the single-slot (``budget_mb=0``) rollback behaviour, identity/all eviction, the held-residency report, the
explicit restore lever and its statistics, and the per-kind RAM estimation fallback. The loader-level
behaviours (subset satisfaction, standalone-VAE dedup) are covered by the stubbed-comfy routing tests.
"""

from __future__ import annotations

from typing import Any

from hordelib.execution.component_cache import (
    DEFAULT_APPROX_RAM_MB,
    ComponentCache,
    ComponentCacheEntry,
    ComponentCacheKey,
    ComponentSlotKind,
    approx_ram_mb_from_bytes,
    component_cache_budget_mb,
)


class _FakeModule:
    def __init__(self, applied_uuid: object | None) -> None:
        self.current_weight_patches_uuid = applied_uuid


class _FakePatcher:
    """The minimum shape the residue probe and the restorer look for on a comfy patcher."""

    def __init__(self, *, applied_uuid: object | None, patches_uuid: object | None) -> None:
        self.model = _FakeModule(applied_uuid)
        self.patches_uuid = patches_uuid
        self.offload_device = "cpu"
        self.unpatched = False

    def unpatch_model(self, device_to: object = None, unpatch_weights: bool = True) -> None:
        self.unpatched = True
        self.model.current_weight_patches_uuid = None


def _key(identity: str, kind: ComponentSlotKind = ComponentSlotKind.CHECKPOINT) -> ComponentCacheKey:
    return ComponentCacheKey(kind, identity)


def _entry(identity: str, mb: float, *, payload: Any = None) -> ComponentCacheEntry:
    return ComponentCacheEntry(
        key=_key(identity),
        payload=(identity, None, None) if payload is None else payload,
        approx_ram_mb=mb,
        source_ckpt_path=f"/models/{identity}",
    )


def test_budget_zero_is_single_slot() -> None:
    """A zero budget keeps exactly one entry: each insert evicts every prior entry (the rollback lever)."""
    cache = ComponentCache(budget_mb=0)

    assert cache.put(_entry("a", 100)) == []
    evicted = cache.put(_entry("b", 100))

    assert [entry.key.identity for entry in evicted] == ["a"]
    assert len(cache) == 1
    assert cache.get(_key("a")) is None
    assert cache.get(_key("b")) is not None


def test_positive_budget_holds_multiple_until_full() -> None:
    """A positive budget keeps as many entries as fit; the insert that overflows evicts the coldest."""
    cache = ComponentCache(budget_mb=250)

    assert cache.put(_entry("a", 100)) == []
    assert cache.put(_entry("b", 100)) == []
    evicted = cache.put(_entry("c", 100))

    assert [entry.key.identity for entry in evicted] == ["a"]
    assert len(cache) == 2
    assert cache.get(_key("a")) is None
    assert cache.get(_key("b")) is not None
    assert cache.get(_key("c")) is not None


def test_eviction_follows_recency_not_insertion_order() -> None:
    """A get bumps recency, so the least-recently-used entry (not the oldest inserted) is evicted."""
    cache = ComponentCache(budget_mb=250)
    cache.put(_entry("a", 100))
    cache.put(_entry("b", 100))

    # Touch 'a' so 'b' becomes the coldest despite being inserted later.
    assert cache.get(_key("a")) is not None

    evicted = cache.put(_entry("c", 100))

    assert [entry.key.identity for entry in evicted] == ["b"]
    assert cache.get(_key("a")) is not None
    assert cache.get(_key("c")) is not None


def test_just_inserted_entry_is_never_evicted_even_when_oversized() -> None:
    """A single entry larger than the whole budget still loads and stays until the next insert displaces it."""
    cache = ComponentCache(budget_mb=100)

    evicted = cache.put(_entry("huge", 5000))

    assert evicted == []
    assert len(cache) == 1
    assert cache.get(_key("huge")) is not None


def test_put_same_key_replaces_payload() -> None:
    """Re-inserting a key overwrites its entry (a broader subset load replacing a narrower one)."""
    cache = ComponentCache(budget_mb=1000)
    cache.put(_entry("a", 100))

    broader = ComponentCacheEntry(
        key=_key("a"),
        payload=("broader", "clip", "vae"),
        approx_ram_mb=200,
        source_ckpt_path="/models/a",
    )
    cache.put(broader)

    assert len(cache) == 1
    served = cache.get(_key("a"))
    assert served is not None
    assert served.payload == ("broader", "clip", "vae")


def test_acquisition_never_restores_the_entry(monkeypatch) -> None:
    """Serving hands the payload over untouched, whatever the caller declares.

    Comfy unpatches a component at its next load whenever the patch set differs, so normalising on the
    way out would only eject never-patched components from the device for nothing.
    """
    restored: list[Any] = []
    monkeypatch.setattr(
        "hordelib.execution.component_restore.restore_payload",
        lambda payload: restored.append(payload) or 0,
    )

    cache = ComponentCache(budget_mb=1000)
    cache.put(_entry("a", 100))

    assert cache.get(_key("a"), will_mutate=True) is not None
    assert cache.get(_key("a")) is not None
    assert cache.get(_key("a"), will_mutate=True) is not None

    assert restored == []


def test_declaring_mutation_only_counts_towards_the_statistics() -> None:
    """``marked`` counts declared-mutator acquisitions; only the explicit lever moves the restore counts."""
    cache = ComponentCache(budget_mb=1000)
    cache.put(_entry("a", 100))

    cache.get(_key("a"), will_mutate=True)
    cache.get(_key("a"))
    cache.get(_key("a"), will_mutate=True)
    cache.get(_key("missing"), will_mutate=True)  # a miss declares nothing about a resident component

    stats = cache.restore_stats()
    assert stats.marked == 2
    assert stats.restored == 0
    assert stats.restored_bytes == 0


def test_restore_identities_counts_the_bytes_the_components_gave_up(monkeypatch) -> None:
    """The explicit lever is what moves ``restored`` and ``restored_bytes``."""
    monkeypatch.setattr("hordelib.execution.component_restore.restore_payload", lambda payload: 4096)

    cache = ComponentCache(budget_mb=1000)
    cache.put(_entry("a", 100))

    assert cache.restore_identities({"a", "missing"}) == 1

    stats = cache.restore_stats()
    assert stats.restored == 1
    assert stats.restored_bytes == 4096


def test_restore_failure_is_swallowed_and_the_entry_stays_resident(monkeypatch) -> None:
    """A restore that raises leaves the entry served as-is: failing a reclaim is worse than the residue."""

    def _boom(payload):
        raise RuntimeError("restore exploded")

    monkeypatch.setattr("hordelib.execution.component_restore.restore_payload", _boom)

    cache = ComponentCache(budget_mb=1000)
    cache.put(_entry("a", 100))

    assert cache.restore_identities({"a"}) == 1
    assert cache.restore_stats().restored == 0

    served = cache.get(_key("a"))
    assert served is not None
    assert served.payload == ("a", None, None)


def test_restore_identities_restores_whatever_it_is_asked_for_without_evicting() -> None:
    """Restoring by identity acts on the named entries, declared or not, and keeps them resident.

    Staying resident is what makes it cheaper than evicting: the next job re-uploads from host RAM
    instead of re-reading the checkpoint from disk.
    """
    patched = _FakePatcher(applied_uuid="lora", patches_uuid="base")
    untouched = _FakePatcher(applied_uuid=None, patches_uuid=None)
    cache = ComponentCache(budget_mb=1000)
    cache.put(_entry("a", 100, payload=(patched, None, None)))
    cache.put(_entry("b", 100, payload=(untouched, None, None)))

    assert cache.restore_identities({"a", "b", "missing"}) == 2

    assert patched.unpatched is True
    assert untouched.unpatched is True
    assert len(cache) == 2
    assert {snapshot.mutated for snapshot in cache.held_report()} == {False}


def test_held_report_reads_patch_residue_from_the_components() -> None:
    """Residue is reported from the module's applied-patch identity, not from a stored flag."""
    cache = ComponentCache(budget_mb=1000)
    cache.put(_entry("patched", 100, payload=(_FakePatcher(applied_uuid="lora", patches_uuid="base"), None, None)))
    cache.put(_entry("own", 100, payload=(_FakePatcher(applied_uuid="base", patches_uuid="base"), None, None)))
    cache.put(_entry("clean", 100, payload=(_FakePatcher(applied_uuid=None, patches_uuid="base"), None, None)))
    cache.put(_entry("patcherless", 100))

    residue = {snapshot.identity: snapshot.mutated for snapshot in cache.held_report()}

    assert residue == {"patched": True, "own": False, "clean": False, "patcherless": False}


def test_evict_identities_matches_by_identity_string() -> None:
    cache = ComponentCache(budget_mb=1000)
    cache.put(_entry("a", 100))
    cache.put(_entry("b", 100))

    removed = cache.evict_identities({"a", "missing"})

    assert removed == 1
    assert cache.get(_key("a")) is None
    assert cache.get(_key("b")) is not None


def test_evict_all_clears_everything() -> None:
    cache = ComponentCache(budget_mb=1000)
    cache.put(_entry("a", 100))
    cache.put(_entry("b", 100))

    cache.evict_all()

    assert len(cache) == 0
    assert cache.held_mb() == 0.0


def test_held_report_and_held_mb() -> None:
    cache = ComponentCache(budget_mb=1000)
    cache.put(_entry("a", 100))
    cache.put(_entry("b", 250))

    report = cache.held_report()
    identities = {snapshot.identity: snapshot.approx_ram_mb for snapshot in report}

    assert identities == {"a": 100.0, "b": 250.0}
    assert cache.held_mb() == 350.0


def test_approx_ram_mb_from_bytes_uses_constant_on_missing_bytes() -> None:
    """Estimation degrades to the conservative per-kind constant when no positive byte count is given."""
    for kind, constant in DEFAULT_APPROX_RAM_MB.items():
        assert approx_ram_mb_from_bytes(kind, None) == constant
        assert approx_ram_mb_from_bytes(kind, 0) == constant
        assert approx_ram_mb_from_bytes(kind, -5) == constant

    one_gib = 1024 * 1024 * 1024
    assert approx_ram_mb_from_bytes(ComponentSlotKind.VAE, one_gib) == 1024.0


def test_budget_env_default_and_override(monkeypatch) -> None:
    monkeypatch.delenv("HORDE_COMPONENT_CACHE_MB", raising=False)
    assert component_cache_budget_mb() == 0.0

    monkeypatch.setenv("HORDE_COMPONENT_CACHE_MB", "8192")
    assert component_cache_budget_mb() == 8192.0

    monkeypatch.setenv("HORDE_COMPONENT_CACHE_MB", "not-a-number")
    assert component_cache_budget_mb() == 0.0

    monkeypatch.setenv("HORDE_COMPONENT_CACHE_MB", "-100")
    assert component_cache_budget_mb() == 0.0
