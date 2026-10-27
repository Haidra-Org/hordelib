"""GPU-free unit tests for the MB-budgeted component cache (``hordelib.execution.component_cache``).

These pin the cache's contract independently of the loader: LRU recency and eviction order, budget fitting,
the single-slot (``budget_mb=0``) rollback behaviour, the split components' lifetime across a job the host
did not retain the model past, identity/all eviction, the stop flag every eviction
path sets on the entry's weight prefetch, the held-residency report, the
explicit restore lever and its statistics, the device-weight release that keeps every entry, and the
per-kind RAM estimation fallback. The loader-level
behaviours (subset satisfaction, standalone-VAE dedup) are covered by the stubbed-comfy routing tests.
"""

from __future__ import annotations

from typing import Any

import pytest

from hordelib.execution import component_cache
from hordelib.execution.component_cache import (
    DEFAULT_APPROX_RAM_MB,
    ComponentCache,
    ComponentCacheEntry,
    ComponentCacheKey,
    ComponentSlotKind,
    approx_ram_mb_from_bytes,
    component_cache_budget_mb,
    release_component_device_weights,
)


class _FakeModule:
    def __init__(self, applied_uuid: object | None, loaded_bytes: int) -> None:
        self.current_weight_patches_uuid = applied_uuid
        self.model_loaded_weight_memory = loaded_bytes


class _FakePatcher:
    """The minimum shape the residue probe, the restorer and the device release look for on a comfy patcher.

    ``unpatch_model`` zeroes the module's loaded weight memory and records the device it was sent to, as
    ComfyUI's detach does.
    """

    def __init__(self, *, applied_uuid: object | None, patches_uuid: object | None, loaded_bytes: int = 0) -> None:
        self.model = _FakeModule(applied_uuid, loaded_bytes)
        self.patches_uuid = patches_uuid
        self.offload_device = "cpu"
        self.unpatched = False
        self.unpatched_to: object = None

    def unpatch_model(self, device_to: object = None, unpatch_weights: bool = True) -> None:
        self.unpatched = True
        self.unpatched_to = device_to
        self.model.current_weight_patches_uuid = None
        self.model.model_loaded_weight_memory = 0


class _FakeWrappedComponent:
    """Represents a comfy ``CLIP`` or ``VAE``, whose patcher is reached through ``patcher``."""

    def __init__(self, patcher: _FakePatcher) -> None:
        self.patcher = patcher


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


def _kind_entry(identity: str, kind: ComponentSlotKind) -> ComponentCacheEntry:
    return ComponentCacheEntry(
        key=_key(identity, kind),
        payload=(identity,),
        approx_ram_mb=100,
        source_ckpt_path=f"/models/{identity}",
    )


def test_budget_zero_keeps_one_entry_per_bare_kind() -> None:
    """A split-files model's three components stay resident together under the single slot."""
    cache = ComponentCache(budget_mb=0)

    assert cache.put(_kind_entry("model:unet", ComponentSlotKind.UNET)) == []
    assert cache.put(_kind_entry("text_encoders/te.safetensors", ComponentSlotKind.CLIP)) == []
    assert cache.put(_kind_entry("vae/vae.safetensors", ComponentSlotKind.VAE)) == []

    assert len(cache) == 3


def test_budget_zero_insert_displaces_its_own_kind() -> None:
    """A second text encoder replaces the first and leaves the diffusion model and VAE resident."""
    cache = ComponentCache(budget_mb=0)
    cache.put(_kind_entry("model:unet", ComponentSlotKind.UNET))
    cache.put(_kind_entry("te_a", ComponentSlotKind.CLIP))
    cache.put(_kind_entry("vae_a", ComponentSlotKind.VAE))

    evicted = cache.put(_kind_entry("te_b", ComponentSlotKind.CLIP))

    assert [entry.key.identity for entry in evicted] == ["te_a"]
    assert {entry.identity for entry in cache.held_report()} == {"model:unet", "te_b", "vae_a"}


def test_budget_zero_checkpoint_displaces_every_kind_and_is_displaced_by_any() -> None:
    """A whole checkpoint carries its own text encoder and VAE, so it shares the slot with nothing."""
    cache = ComponentCache(budget_mb=0)
    cache.put(_kind_entry("model:unet", ComponentSlotKind.UNET))
    cache.put(_kind_entry("te_a", ComponentSlotKind.CLIP))

    evicted_by_checkpoint = cache.put(_kind_entry("sd15", ComponentSlotKind.CHECKPOINT))
    assert sorted(entry.key.identity for entry in evicted_by_checkpoint) == ["model:unet", "te_a"]

    evicted_by_vae = cache.put(_kind_entry("vae_a", ComponentSlotKind.VAE))
    assert [entry.key.identity for entry in evicted_by_vae] == ["sd15"]
    assert len(cache) == 1


def test_release_single_slot_evicts_only_competitors() -> None:
    """The release before a cold load drops what that load displaces and nothing else."""
    cache = ComponentCache(budget_mb=0)
    cache.put(_kind_entry("model:unet", ComponentSlotKind.UNET))
    cache.put(_kind_entry("te_a", ComponentSlotKind.CLIP))

    released = cache.release_single_slot(ComponentSlotKind.VAE)
    assert released == []

    released = cache.release_single_slot(ComponentSlotKind.CLIP)
    assert [entry.key.identity for entry in released] == ["te_a"]
    assert cache.get(_key("model:unet", ComponentSlotKind.UNET)) is not None

    released = cache.release_single_slot(ComponentSlotKind.CHECKPOINT)
    assert [entry.key.identity for entry in released] == ["model:unet"]
    assert len(cache) == 0


_SPLIT_UNET = _key("model_a:unet", ComponentSlotKind.UNET)
_SPLIT_TEXT_ENCODER = _key("text_encoders/te_a.safetensors:qwen_image:default", ComponentSlotKind.CLIP)
_SPLIT_VAE = _key("vae/vae_a.safetensors", ComponentSlotKind.VAE)


def _cache_split_model(cache: ComponentCache) -> None:
    """Load a split-files model's three components the way its job's loaders do."""
    for key in (_SPLIT_UNET, _SPLIT_TEXT_ENCODER, _SPLIT_VAE):
        cache.put(_kind_entry(key.identity, key.kind))


def test_split_components_stay_cached_after_a_job_without_retention() -> None:
    """A retention denial frees the device only, so the next job's encoder and VAE loads are cache hits.

    The job's end performs no cache operation. Only the size rule, a full clear or a host eviction removes an
    entry.
    """
    cache = ComponentCache(budget_mb=0)
    _cache_split_model(cache)

    held = {(snapshot.kind, snapshot.identity) for snapshot in cache.held_report()}

    assert held == {(key.kind, key.identity) for key in (_SPLIT_UNET, _SPLIT_TEXT_ENCODER, _SPLIT_VAE)}
    assert cache.get(_SPLIT_TEXT_ENCODER) is not None
    assert cache.get(_SPLIT_VAE) is not None


def test_single_slot_cold_load_still_evicts_a_cached_split_component_of_its_kind() -> None:
    """Another model's text encoder displaces the cached one and leaves the diffusion model and VAE."""
    cache = ComponentCache(budget_mb=0)
    _cache_split_model(cache)

    released = cache.release_single_slot(ComponentSlotKind.CLIP)
    evicted = cache.put(_kind_entry("text_encoders/te_b.safetensors:lumina2:default", ComponentSlotKind.CLIP))

    assert [entry.key for entry in released] == [_SPLIT_TEXT_ENCODER]
    assert evicted == []
    assert cache.get(_SPLIT_TEXT_ENCODER) is None
    assert cache.get(_SPLIT_UNET) is not None
    assert cache.get(_SPLIT_VAE) is not None


def test_evict_all_clears_cached_split_components() -> None:
    cache = ComponentCache(budget_mb=0)
    _cache_split_model(cache)

    cache.evict_all()

    assert len(cache) == 0
    assert cache.get(_SPLIT_TEXT_ENCODER) is None
    assert cache.get(_SPLIT_VAE) is None


def test_release_single_slot_is_a_no_op_under_a_budget() -> None:
    cache = ComponentCache(budget_mb=1000)
    cache.put(_kind_entry("te_a", ComponentSlotKind.CLIP))

    assert cache.release_single_slot(ComponentSlotKind.CLIP) == []
    assert len(cache) == 1


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

    def _record_restore(payload: Any) -> int:
        restored.append(payload)
        return 0

    monkeypatch.setattr("hordelib.execution.component_restore.restore_payload", _record_restore)

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


def _split_entry(key: ComponentCacheKey, payload: Any) -> ComponentCacheEntry:
    return ComponentCacheEntry(key=key, payload=payload, approx_ram_mb=100, source_ckpt_path=key.identity)


def test_release_device_weights_moves_loaded_components_off_the_device_and_keeps_every_entry() -> None:
    """A split text encoder and VAE left on the device go to their offload device and stay cached.

    The diffusion model a VRAM unload already detached claims nothing and is not touched, so only the two
    components that held device weights are counted, and the explicit restore statistics do not move.
    """
    detached_unet = _FakePatcher(applied_uuid=None, patches_uuid=None)
    text_encoder = _FakePatcher(applied_uuid=None, patches_uuid=None, loaded_bytes=3 * 2**30)
    vae = _FakePatcher(applied_uuid=None, patches_uuid=None, loaded_bytes=160 * 2**20)
    cache = ComponentCache(budget_mb=0)
    cache.put(_split_entry(_SPLIT_UNET, (detached_unet, None, None)))
    cache.put(_split_entry(_SPLIT_TEXT_ENCODER, (None, _FakeWrappedComponent(text_encoder), None)))
    cache.put(_split_entry(_SPLIT_VAE, (None, None, _FakeWrappedComponent(vae))))

    assert cache.release_device_weights() == 2

    assert detached_unet.unpatched is False
    assert text_encoder.unpatched_to == text_encoder.offload_device
    assert vae.unpatched_to == vae.offload_device
    assert text_encoder.model.model_loaded_weight_memory == 0
    assert vae.model.model_loaded_weight_memory == 0
    assert len(cache) == 3
    assert all(cache.get(key) is not None for key in (_SPLIT_UNET, _SPLIT_TEXT_ENCODER, _SPLIT_VAE))
    assert cache.restore_stats().restored == 0


def test_release_device_weights_after_a_full_unload_touches_nothing() -> None:
    """Every component a full unload detached claims no device weights, so the release is free and counts zero."""
    patchers = [_FakePatcher(applied_uuid=None, patches_uuid=None) for _ in range(3)]
    cache = ComponentCache(budget_mb=0)
    for key, patcher in zip((_SPLIT_UNET, _SPLIT_TEXT_ENCODER, _SPLIT_VAE), patchers, strict=True):
        cache.put(_split_entry(key, (patcher, None, None)))

    assert cache.release_device_weights() == 0

    assert [patcher.unpatched for patcher in patchers] == [False, False, False]
    assert len(cache) == 3


def test_release_device_weights_failure_is_swallowed_and_the_entry_stays() -> None:
    """A component whose unpatch raises is not counted and stays cached, and the others are still released."""

    class _RefusingPatcher(_FakePatcher):
        def unpatch_model(self, device_to: object = None, unpatch_weights: bool = True) -> None:
            raise RuntimeError("unpatch exploded")

    refusing = _RefusingPatcher(applied_uuid=None, patches_uuid=None, loaded_bytes=2**30)
    releasing = _FakePatcher(applied_uuid=None, patches_uuid=None, loaded_bytes=2**30)
    cache = ComponentCache(budget_mb=0)
    cache.put(_split_entry(_SPLIT_TEXT_ENCODER, (refusing, None, None)))
    cache.put(_split_entry(_SPLIT_VAE, (releasing, None, None)))

    assert cache.release_device_weights() == 1

    assert releasing.unpatched is True
    assert len(cache) == 2


def test_release_component_device_weights_acts_on_the_process_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """The module function releases through this process's cache, and is zero where there is none."""
    text_encoder = _FakePatcher(applied_uuid=None, patches_uuid=None, loaded_bytes=2**30)
    cache = ComponentCache(budget_mb=0)
    cache.put(_split_entry(_SPLIT_TEXT_ENCODER, (text_encoder, None, None)))

    monkeypatch.setattr(component_cache, "process_component_cache", lambda: cache)
    assert release_component_device_weights() == 1
    assert len(cache) == 1

    monkeypatch.setattr(component_cache, "process_component_cache", lambda: None)
    assert release_component_device_weights() == 0


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


def test_budget_eviction_stops_the_evicted_entry_prefetch() -> None:
    cache = ComponentCache(budget_mb=150)
    coldest = _entry("a", 100)
    newest = _entry("b", 100)
    cache.put(coldest)

    assert cache.put(newest) == [coldest]

    assert coldest.prefetch_stop.is_set()
    assert not newest.prefetch_stop.is_set()


def test_single_slot_insert_and_release_stop_the_displaced_prefetch() -> None:
    cache = ComponentCache(budget_mb=0)
    displaced_by_insert = _entry("a", 100)
    resident = _entry("b", 100)
    cache.put(displaced_by_insert)
    cache.put(resident)

    released = cache.release_single_slot(ComponentSlotKind.CHECKPOINT)

    assert released == [resident]
    assert displaced_by_insert.prefetch_stop.is_set()
    assert resident.prefetch_stop.is_set()


def test_identity_and_full_eviction_stop_the_evicted_prefetches() -> None:
    cache = ComponentCache(budget_mb=1000)
    named = _entry("a", 100)
    remaining = _entry("b", 100)
    cache.put(named)
    cache.put(remaining)

    cache.evict_identities({"a"})
    assert named.prefetch_stop.is_set()
    assert not remaining.prefetch_stop.is_set()

    cache.evict_all()
    assert remaining.prefetch_stop.is_set()


def test_same_key_replacement_stops_the_replaced_prefetch_only() -> None:
    cache = ComponentCache(budget_mb=1000)
    narrower = _entry("a", 100)
    broader = _entry("a", 100)
    cache.put(narrower)
    cache.put(broader)
    cache.put(broader)

    assert narrower.prefetch_stop.is_set()
    assert not broader.prefetch_stop.is_set()
