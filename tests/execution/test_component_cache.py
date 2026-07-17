"""GPU-free unit tests for the MB-budgeted component cache (``hordelib.execution.component_cache``).

These pin the cache's contract independently of the loader: LRU recency and eviction order, budget fitting,
the single-slot (``budget_mb=0``) rollback behaviour, identity/all eviction, the held-residency report, the
non-reusable serve semantics, and the per-kind RAM estimation fallback. The loader-level behaviours
(subset satisfaction, standalone-VAE dedup) are covered by the stubbed-comfy routing tests.
"""

from __future__ import annotations

from hordelib.execution.component_cache import (
    DEFAULT_APPROX_RAM_MB,
    ComponentCache,
    ComponentCacheEntry,
    ComponentCacheKey,
    ComponentSlotKind,
    approx_ram_mb_from_bytes,
    component_cache_budget_mb,
    pristine_lora_serving_enabled,
)


def _key(identity: str, kind: ComponentSlotKind = ComponentSlotKind.CHECKPOINT) -> ComponentCacheKey:
    return ComponentCacheKey(kind, identity)


def _entry(identity: str, mb: float, *, reusable: bool = True) -> ComponentCacheEntry:
    return ComponentCacheEntry(
        key=_key(identity),
        payload=(identity, None, None),
        approx_ram_mb=mb,
        reusable=reusable,
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
        reusable=True,
        source_ckpt_path="/models/a",
    )
    cache.put(broader)

    assert len(cache) == 1
    served = cache.get(_key("a"))
    assert served is not None
    assert served.payload == ("broader", "clip", "vae")


def test_non_reusable_entry_is_never_served() -> None:
    """A non-reusable entry is a miss on every lookup, though it stays resident (subject to eviction)."""
    cache = ComponentCache(budget_mb=1000)
    cache.put(_entry("a", 100, reusable=False))

    assert cache.get(_key("a")) is None
    assert len(cache) == 1


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


def test_pristine_lora_serving_toggle(monkeypatch) -> None:
    monkeypatch.delenv("HORDE_COMPONENT_CACHE_PRISTINE_LORA_SERVING", raising=False)
    assert pristine_lora_serving_enabled() is True

    monkeypatch.setenv("HORDE_COMPONENT_CACHE_PRISTINE_LORA_SERVING", "0")
    assert pristine_lora_serving_enabled() is False

    monkeypatch.setenv("HORDE_COMPONENT_CACHE_PRISTINE_LORA_SERVING", "true")
    assert pristine_lora_serving_enabled() is True
