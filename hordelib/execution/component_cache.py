"""Content-addressed, MB-budgeted LRU cache for the loader's in-RAM model components.

The checkpoint loader keeps loaded ``(model, clip, vae, ...)`` tuples in RAM so a later job or
disaggregated stage that needs the same component skips the multi-gigabyte disk read. This module is that
store. It replaces the historical single-slot dictionary (which wiped every prior entry on each new load)
with an LRU keyed by component identity and bounded by an approximate host-RAM budget: as many components
as fit the budget stay resident, and the coldest are evicted to make room.

Keys are :class:`ComponentCacheKey` ``(kind, identity)`` pairs. ``identity`` is content-addressed where a
component's content hash is cheaply known (a standalone VAE's ``vae@<hash>``) and otherwise a stable
per-checkpoint string (a reference sha256, or ``<name>:<size>``, or the bare model name), so two requests
that resolve to the same component share one entry.

Budgeting is deliberately approximate: :attr:`ComponentCacheEntry.approx_ram_mb` is an estimate (from a
component-identity sidecar's tensor byte counts, a file size, or a per-kind constant), not a measured
resident-set delta, so the budget bounds intent rather than guaranteeing an exact RSS ceiling. A budget of
``0`` reproduces the historical single-slot behaviour exactly (each insert evicts every other entry), which
is the rollback lever.

Entries are handed out by reference and are not normalised on the way out: a job that patches one is
patching the object the next job will be given. That is safe for the load path because ComfyUI restores
patch residue lazily, at the component's next load: a patcher whose patch set differs from the one the
module's weights currently hold triggers a full unpatch-to-offload and re-upload before the load
proceeds, so a component always reaches a job carrying that job's own patches. What the cache offers on
top is a declaration (``will_mutate``) that feeds residency reporting and the restore statistics, and
:meth:`ComponentCache.restore_identities` as an explicit lever for giving a resident component's device
memory back (see :mod:`hordelib.execution.component_restore`).

Concurrency: a :class:`threading.Lock` guards every mutation of the entry map so the recency and eviction
bookkeeping stays consistent. Restoration deliberately runs outside that lock, since it walks a model's
modules and holding the cache lock across it would serialise unrelated lookups behind one restore.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable, Collection
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from loguru import logger
from pydantic import BaseModel, ConfigDict

__all__ = [
    "ComponentCache",
    "ComponentCacheEntry",
    "ComponentCacheKey",
    "ComponentRestoreStats",
    "ComponentSlotKind",
    "DEFAULT_APPROX_RAM_MB",
    "HeldComponentSnapshot",
    "approx_ram_mb_from_bytes",
    "component_cache_budget_mb",
    "component_restore_stats",
    "evict_components",
    "held_components",
    "process_component_cache",
    "restore_components",
]

_BUDGET_ENV_VAR = "HORDE_COMPONENT_CACHE_MB"
"""Approximate host-RAM budget for the component cache, in megabytes. Unset or ``0`` keeps the historical
single-slot behaviour (one component resident at a time); a positive value opts into multi-component
residency up to that many megabytes."""

_BYTES_PER_MB = 1024 * 1024


class ComponentSlotKind(StrEnum):
    """The kind of model component a cache entry holds.

    ``CHECKPOINT`` is a full (or subset) monolithic-checkpoint load tuple; ``UNET``/``CLIP``/``VAE`` are bare
    single-component loads. The kind is part of the cache key, so a bare UNet and a bare text encoder loaded
    from the same checkpoint never alias even when their identities coincide.
    """

    UNET = "unet"
    CLIP = "clip"
    VAE = "vae"
    CHECKPOINT = "checkpoint"


@dataclass(frozen=True)
class ComponentCacheKey:
    """The content-addressed identity of a cached component: its kind plus an identity string.

    The identity is a content hash where one is cheaply available (``vae@<hash>``), otherwise a stable
    per-checkpoint string. Frozen so it is hashable and usable as a dict key.
    """

    kind: ComponentSlotKind
    identity: str


class HeldComponentSnapshot(BaseModel):
    """A serialisable summary of one resident cache entry, for reporting residency across process boundaries."""

    model_config = ConfigDict(frozen=True)

    kind: ComponentSlotKind
    identity: str
    approx_ram_mb: float
    mutated: bool = False
    """Whether the entry currently holds patch residue a restore would clear, read from the components
    themselves at report time. A reader deciding how to reclaim memory cannot choose restoring over
    evicting without seeing which entries it applies to, so residency reporting carries it. Defaults
    False so an older reader deserialises unchanged."""


@dataclass(frozen=True)
class ComponentRestoreStats:
    """Cumulative counts of the restore contract's activity in this process.

    Reported rather than logged because the useful questions are ratios over time: how many acquisitions
    declared they would patch, how often the explicit restore lever was pulled, and how much weight
    memory that returned. A silent period is ambiguous on its own, and these three make it readable:
    ``marked`` staying flat means jobs stopped declaring, while ``marked`` rising with ``restored`` flat
    means the declarations arrive and nothing is asking for the memory back.
    """

    marked: int
    """Acquisitions that declared they would patch the component they were handed."""
    restored: int
    """Entries put through an explicit restore request."""
    restored_bytes: int
    """Device weight memory those restores gave up, summed over every restored component."""


@dataclass
class ComponentCacheEntry:
    """One resident component: its key, the loader payload, an approximate RAM cost, and recency.

    ``payload`` is the loader's ``(model, clip, vae, ...)`` tuple, held by reference and never copied.
    ``last_used`` is a monotonic timestamp maintained by the cache.
    """

    key: ComponentCacheKey
    payload: Any
    approx_ram_mb: float
    source_ckpt_path: str
    last_used: float = field(default_factory=time.monotonic)


DEFAULT_APPROX_RAM_MB: dict[ComponentSlotKind, float] = {
    ComponentSlotKind.VAE: 512.0,
    ComponentSlotKind.CLIP: 1500.0,
    ComponentSlotKind.UNET: 5000.0,
    ComponentSlotKind.CHECKPOINT: 7000.0,
}
"""Conservative (deliberately high) per-kind RAM estimates used when no byte count is reachable.

Erring high makes the budget evict sooner rather than overrun host RAM, so an unknown-size component is
treated as if it were a large one of its kind.
"""


def approx_ram_mb_from_bytes(kind: ComponentSlotKind, tensor_bytes: int | None) -> float:
    """Return an approximate RAM cost in megabytes for a component of *kind*.

    Uses *tensor_bytes* when it is a positive byte count; otherwise falls back to the conservative per-kind
    constant in :data:`DEFAULT_APPROX_RAM_MB`. Never raises, so an estimation miss degrades to the constant
    rather than failing a load.
    """
    if tensor_bytes is not None and tensor_bytes > 0:
        return tensor_bytes / _BYTES_PER_MB
    return DEFAULT_APPROX_RAM_MB[kind]


def component_cache_budget_mb() -> float:
    """Return the configured component-cache budget in megabytes (``0`` when unset or unparseable).

    Read from :data:`_BUDGET_ENV_VAR`. A missing, empty, negative, or non-numeric value yields ``0``, which
    selects the single-slot rollback behaviour.
    """
    raw = os.environ.get(_BUDGET_ENV_VAR, "").strip()
    if not raw:
        return 0.0
    try:
        value = float(raw)
    except ValueError:
        logger.warning(f"Ignoring non-numeric {_BUDGET_ENV_VAR}={raw!r}; using single-slot component cache.")
        return 0.0
    return value if value > 0.0 else 0.0


def _never_carries_residue(payload: Any) -> bool:
    return False


def _patch_residue_predicate() -> Callable[[Any], bool]:
    """Return the predicate that reports whether a payload carries patch residue.

    The import is deferred so this module stays importable in a process that never loads a model backend;
    where it is unavailable no payload can be patched, so the constant answer is the accurate one.
    """
    try:
        from hordelib.execution.component_restore import has_patch_residue
    except Exception as import_error:
        logger.debug(
            "No patch-residue probe in this process: {} {}",
            type(import_error).__name__,
            import_error,
        )
        return _never_carries_residue
    return has_patch_residue


class ComponentCache:
    """An MB-budgeted LRU of loaded model components, keyed by content identity.

    A positive budget keeps as many components resident as fit within it, evicting the least-recently-used
    first. A budget of ``0`` keeps exactly one component resident (each insert evicts every other entry),
    reproducing the historical single-slot cache. All mutations are guarded by a lock; the just-inserted
    entry is never evicted to satisfy its own insertion, so a single component larger than the budget still
    loads (and stays until the next insert displaces it).
    """

    def __init__(self, budget_mb: float) -> None:
        """Create a cache bounded by *budget_mb* megabytes (``0`` selects single-slot behaviour)."""
        self._budget_mb = float(budget_mb) if budget_mb > 0 else 0.0
        self._entries: dict[ComponentCacheKey, ComponentCacheEntry] = {}
        self._lock = threading.Lock()
        # A strictly increasing recency stamp: wall-clock resolution is too coarse on some platforms to
        # order operations that happen microseconds apart, so recency is a monotonic counter, not a clock.
        self._recency = 0.0
        self._marked_count = 0
        self._restored_count = 0
        self._restored_bytes = 0

    def _next_recency_locked(self) -> float:
        self._recency += 1.0
        return self._recency

    @property
    def budget_mb(self) -> float:
        """The configured budget in megabytes (``0`` means single-slot)."""
        return self._budget_mb

    def __len__(self) -> int:
        """Return the number of resident entries."""
        with self._lock:
            return len(self._entries)

    def get(self, key: ComponentCacheKey, *, will_mutate: bool = False) -> ComponentCacheEntry | None:
        """Return the resident entry for *key* as it stands, or None on a miss.

        The payload is handed over untouched: patch residue from a previous job is cleared by ComfyUI at
        the component's next load, so an acquisition that normalised it would only be paying to eject
        never-patched components from the device. Set *will_mutate* when this caller will patch the
        component (a LoRA-bearing job does); it is counted for :meth:`restore_stats` and changes nothing
        about what is served.
        """
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            entry.last_used = self._next_recency_locked()
            if will_mutate:
                self._marked_count += 1
        return entry

    def restore_entry(self, entry: ComponentCacheEntry) -> int:
        """Return *entry*'s payload to its loaded state; report the device bytes it gave up.

        Never raises: a component that cannot be restored is logged at debug and reported as having given
        up nothing, because failing a reclaim request is worse than leaving the memory claimed. The import
        is deferred so this module stays importable in a process that never loads a model backend.
        """
        try:
            from hordelib.execution.component_restore import restore_payload

            released = restore_payload(entry.payload)
        except Exception as restore_error:
            logger.debug(
                "Could not restore component {}: {} {}",
                entry.key.identity,
                type(restore_error).__name__,
                restore_error,
            )
            return 0

        with self._lock:
            self._restored_count += 1
            self._restored_bytes += released
        logger.debug(
            "Restored resident component: identity={}, released_mb={:.1f}",
            entry.key.identity,
            released / _BYTES_PER_MB,
        )
        return released

    def restore_stats(self) -> ComponentRestoreStats:
        """Return this process's cumulative restore activity."""
        with self._lock:
            return ComponentRestoreStats(
                marked=self._marked_count,
                restored=self._restored_count,
                restored_bytes=self._restored_bytes,
            )

    def put(self, entry: ComponentCacheEntry) -> list[ComponentCacheEntry]:
        """Insert (or replace) *entry* and evict to fit the budget; return the evicted entries, coldest first.

        A same-key entry is overwritten (a broader subset load replacing a narrower one). The returned
        evictions are the caller's to log and clean up; the just-inserted entry is never among them.
        """
        with self._lock:
            entry.last_used = self._next_recency_locked()
            self._entries[entry.key] = entry
            return self._evict_to_fit_locked(protected_key=entry.key)

    def evict_identities(self, identities: Collection[str]) -> int:
        """Evict every entry whose key identity is in *identities*; return the number evicted.

        Matches on the identity string across all kinds, so a content hash shared by more than one kind (it
        should not be) would evict each. Used to drop a specific component by content identity.
        """
        wanted = set(identities)
        with self._lock:
            doomed = [key for key in self._entries if key.identity in wanted]
            for key in doomed:
                del self._entries[key]
            return len(doomed)

    def evict_all(self) -> None:
        """Drop every resident entry (the full-cache clear used at RAM-unload boundaries)."""
        with self._lock:
            self._entries.clear()

    def restore_identities(self, identities: Collection[str]) -> int:
        """Restore every entry whose key identity is in *identities*; return the number restored.

        The cheaper counterpart to :meth:`evict_identities`: it clears patch residue and the components'
        weight-memory claim while the pristine weights stay resident, so the next job re-uploads them
        rather than re-reading them from disk. An identity the cache does not hold is skipped, matching
        eviction, so a raced request is harmless.
        """
        wanted = set(identities)
        with self._lock:
            targets = [entry for key, entry in self._entries.items() if key.identity in wanted]

        for entry in targets:
            self.restore_entry(entry)
        return len(targets)

    def held_report(self) -> list[HeldComponentSnapshot]:
        """Return a serialisable snapshot of every resident entry (kind, identity, RAM, patch residue).

        Residue is read from the components themselves rather than from a stored flag, so the report
        answers what the entry holds now instead of what some earlier acquisition intended.
        """
        carries_residue = _patch_residue_predicate()
        with self._lock:
            return [
                HeldComponentSnapshot(
                    kind=entry.key.kind,
                    identity=entry.key.identity,
                    approx_ram_mb=entry.approx_ram_mb,
                    mutated=carries_residue(entry.payload),
                )
                for entry in self._entries.values()
            ]

    def held_mb(self) -> float:
        """Return the summed approximate RAM cost of all resident entries, in megabytes."""
        with self._lock:
            return self._held_mb_locked()

    def _held_mb_locked(self) -> float:
        return sum(entry.approx_ram_mb for entry in self._entries.values())

    def _evict_to_fit_locked(self, *, protected_key: ComponentCacheKey) -> list[ComponentCacheEntry]:
        evicted: list[ComponentCacheEntry] = []
        if self._budget_mb <= 0.0:
            # Single-slot rollback: only the just-inserted entry survives an insert.
            for key in [candidate for candidate in self._entries if candidate != protected_key]:
                evicted.append(self._entries.pop(key))
            return evicted

        while self._held_mb_locked() > self._budget_mb:
            victim_key = self._coldest_evictable_key_locked(protected_key)
            if victim_key is None:
                break  # only the protected entry remains; keep it even if it alone exceeds the budget
            evicted.append(self._entries.pop(victim_key))
        return evicted

    def _coldest_evictable_key_locked(self, protected_key: ComponentCacheKey) -> ComponentCacheKey | None:
        coldest_key: ComponentCacheKey | None = None
        coldest_used = 0.0
        for key, entry in self._entries.items():
            if key == protected_key:
                continue
            if coldest_key is None or entry.last_used < coldest_used:
                coldest_key = key
                coldest_used = entry.last_used
        return coldest_key


def process_component_cache() -> ComponentCache | None:
    """Return this process's component cache, or None where no model backend is loaded.

    Resolving it goes through the shared model manager, whose import pulls torch, so the import is
    deferred to the call. A process with no backend (a dry-run lane, a CPU-only helper) gets None and
    treats every residency operation as a no-op rather than importing a model stack it will never use.
    """
    try:
        from hordelib.shared_model_manager import SharedModelManager

        return getattr(SharedModelManager.manager, "_models_in_ram", None)
    except Exception as lookup_error:
        logger.debug(
            "No component cache in this process: {} {}",
            type(lookup_error).__name__,
            lookup_error,
        )
        return None


def held_components() -> list[HeldComponentSnapshot]:
    """Return a snapshot of every component resident in this process, empty where there is no cache."""
    cache = process_component_cache()
    return [] if cache is None else cache.held_report()


def restore_components(identities: Collection[str]) -> int:
    """Restore the named components in place; return how many resident entries were handled.

    The cheaper of the two reclaim actions: it clears patch residue and hands the components' weight
    memory back while the pristine weights stay resident, so the next job re-uploads rather than
    re-reading from disk. Prefer it over :func:`evict_components` where the pressure allows.
    """
    cache = process_component_cache()
    return 0 if cache is None else cache.restore_identities(identities)


def evict_components(identities: Collection[str]) -> int:
    """Drop the named components from RAM; return how many resident entries were evicted."""
    cache = process_component_cache()
    return 0 if cache is None else cache.evict_identities(identities)


def component_restore_stats() -> ComponentRestoreStats:
    """Return this process's cumulative restore activity, zeroed where there is no cache."""
    cache = process_component_cache()
    if cache is None:
        return ComponentRestoreStats(marked=0, restored=0, restored_bytes=0)
    return cache.restore_stats()
