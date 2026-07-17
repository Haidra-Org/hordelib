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

Concurrency: a :class:`threading.Lock` guards every mutation of the entry map so the recency and eviction
bookkeeping stays consistent. The cached payloads themselves are owned serially by comfy execution (one job
runs at a time in an inference process), so this cache never copies or mutates a payload; it only holds and
hands back references.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Collection
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from loguru import logger
from pydantic import BaseModel, ConfigDict

__all__ = [
    "ComponentCache",
    "ComponentCacheEntry",
    "ComponentCacheKey",
    "ComponentSlotKind",
    "DEFAULT_APPROX_RAM_MB",
    "HeldComponentSnapshot",
    "approx_ram_mb_from_bytes",
    "component_cache_budget_mb",
    "pristine_lora_serving_enabled",
]

_BUDGET_ENV_VAR = "HORDE_COMPONENT_CACHE_MB"
"""Approximate host-RAM budget for the component cache, in megabytes. Unset or ``0`` keeps the historical
single-slot behaviour (one component resident at a time); a positive value opts into multi-component
residency up to that many megabytes."""

_PRISTINE_LORA_ENV_VAR = "HORDE_COMPONENT_CACHE_PRISTINE_LORA_SERVING"
"""Set falsy to restore the historical behaviour of never sharing a base loaded by a LoRA-bearing job. The
default (enabled) serves LoRA-bearing jobs a pristine cached base, because the graph's LoRA loader clones the
base before patching it, leaving the cached weights untouched."""

_TRUTHY_VALUES = frozenset({"1", "true", "yes", "on"})

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


@dataclass
class ComponentCacheEntry:
    """One resident component: its key, the loader payload, an approximate RAM cost, and recency.

    ``payload`` is the loader's ``(model, clip, vae, ...)`` tuple, held by reference and never copied or
    mutated. ``reusable`` is False only for an entry that must never be shared with a later request (see
    :meth:`ComponentCache.get`). ``last_used`` is a monotonic timestamp maintained by the cache.
    """

    key: ComponentCacheKey
    payload: Any
    approx_ram_mb: float
    reusable: bool
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


def pristine_lora_serving_enabled() -> bool:
    """Return whether a LoRA-bearing job may be served a pristine cached base (default: enabled).

    Disabled only when :data:`_PRISTINE_LORA_ENV_VAR` is set to a falsy value, restoring the historical
    behaviour of not sharing a base that a LoRA-bearing job loaded.
    """
    raw = os.environ.get(_PRISTINE_LORA_ENV_VAR, "").strip().lower()
    if raw == "":
        return True
    return raw in _TRUTHY_VALUES


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

    def _next_recency_locked(self) -> float:
        self._recency += 1.0
        return self._recency

    @property
    def budget_mb(self) -> float:
        """The configured budget in megabytes (``0`` means single-slot)."""
        return self._budget_mb

    def __len__(self) -> int:
        """Return the number of resident entries (including any non-reusable ones)."""
        with self._lock:
            return len(self._entries)

    def get(self, key: ComponentCacheKey) -> ComponentCacheEntry | None:
        """Return the resident entry for *key* and bump its recency, or None on a miss.

        A non-reusable entry is treated as a miss (returns None) so it is never shared with a later request;
        it stays resident (subject to eviction) but only the job that loaded it ever used it.
        """
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            if not entry.reusable:
                return None
            entry.last_used = self._next_recency_locked()
            return entry

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

    def held_report(self) -> list[HeldComponentSnapshot]:
        """Return a serialisable snapshot of every resident entry (kind, identity, approximate RAM)."""
        with self._lock:
            return [
                HeldComponentSnapshot(
                    kind=entry.key.kind,
                    identity=entry.key.identity,
                    approx_ram_mb=entry.approx_ram_mb,
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
