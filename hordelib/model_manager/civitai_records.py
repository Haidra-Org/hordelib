"""Typed, mutable record subclasses for the CivitAI-sourced LoRA and TI managers.

The LoRA and Textual Inversion managers download model metadata directly from CivitAI (and,
for TI, the Hordeling embedding API) rather than from canonical horde data. ``horde_model_reference``
v4 ships :class:`~horde_model_reference.model_reference_records.LoraModelRecord` and
:class:`~horde_model_reference.model_reference_records.TextualInversionModelRecord` for exactly
this ``managed_elsewhere`` case and explicitly invites providers to subclass them.

These subclasses add the bookkeeping the managers need (CivitAI id, per-version download details,
adhoc/last-used cache state) while remaining ordinary mutable Pydantic records, so:

* the managers operate on typed records instead of ``dict[str, Any]``;
* the records satisfy the :class:`~horde_model_reference.providers.base.ModelProvider` contract and
  can be returned from a provider unchanged (they *are* ``GenericModelRecord`` subclasses); and
* cache state such as ``last_used`` can still be mutated in place during a run.

The canonical top-level fields (``baseline``, ``trigger``, ``config.download``) mirror the active
(latest) version; :attr:`HordeLoraModelRecord.versions` retains the full per-version detail that the
single-version upstream schema does not model.
"""

from __future__ import annotations

from horde_model_reference.model_reference_records import (
    LoraModelRecord,
    TextualInversionModelRecord,
    get_default_config,
)
from pydantic import BaseModel, Field


class LoraVersionEntry(BaseModel):
    """Represents a single downloaded (or downloadable) version of a LoRA.

    A LoRA on CivitAI may publish multiple versions; the horde tracks each one it has touched so it
    can serve a specific version on request and evict by least-recent use.
    """

    model_config = get_default_config()

    filename: str
    """The horde-local filename for this version's ``.safetensors`` weight file."""
    url: str
    """The fully qualified CivitAI download URL for this version."""
    version_id: str
    """The CivitAI version id, as a string (JSON object keys must be strings)."""
    lora_key: str
    """The sanitised key of the parent LoRA this version belongs to."""

    sha256: str | None = None
    """The expected SHA256 checksum of the weight file, if CivitAI provided one."""
    adhoc: bool = False
    """Whether this version was fetched ad-hoc (subject to cache eviction) rather than as a default."""
    size_mb: int = 0
    """The approximate on-disk size of the weight file in megabytes."""
    triggers: list[str] = Field(default_factory=list)
    """The trigger words/phrases which activate this version."""
    base_model: str = "SD 1.5"
    """The CivitAI ``baseModel`` string (e.g. ``"SDXL 1.0"``) for this version."""
    availability: str = "Public"
    """The CivitAI availability of this version (e.g. ``"Public"`` or ``"EarlyAccess"``)."""
    last_used: str | None = None
    """The ``"%Y-%m-%d %H:%M:%S"`` timestamp this version was last used, for cache eviction."""


class HordeLoraModelRecord(LoraModelRecord):
    """A LoRA record enriched with the horde's CivitAI download and cache bookkeeping.

    Subclasses the upstream :class:`~horde_model_reference.model_reference_records.LoraModelRecord`
    so it remains a valid provider record while carrying per-version download detail the
    single-version upstream schema cannot express.
    """

    civitai_id: int = 0
    """The CivitAI model id for this LoRA."""
    orig_name: str = ""
    """The LoRA's original (unsanitised) name as reported by CivitAI."""
    versions: dict[str, LoraVersionEntry] = Field(default_factory=dict)
    """Known versions of this LoRA, keyed by string version id."""
    last_checked: str | None = None
    """The ``"%Y-%m-%d %H:%M:%S"`` timestamp this LoRA's metadata was last refreshed from CivitAI."""


class HordeTextualInversionModelRecord(TextualInversionModelRecord):
    """A textual inversion record enriched with the horde's CivitAI/Hordeling download bookkeeping.

    Unlike LoRAs, a textual inversion is tracked as a single version, so the download detail lives
    directly on the record rather than in a versions map.
    """

    civitai_id: int = 0
    """The CivitAI model id for this textual inversion."""
    orig_name: str = ""
    """The embedding's original (unsanitised) name as reported by CivitAI."""
    filename: str = ""
    """The horde-local filename for this embedding's weight file."""
    url: str = ""
    """The download URL for this embedding (resolved via the Hordeling API at download time)."""
    sha256: str | None = None
    """The expected SHA256 checksum of the weight file, if available."""
    size_kb: int = 0
    """The approximate on-disk size of the weight file in kilobytes."""
    base_model: str = "SD 1.5"
    """The CivitAI ``baseModel`` string for this embedding."""
    version_id: int = 0
    """The CivitAI version id for this embedding."""
    adhoc: bool = False
    """Whether this embedding was fetched ad-hoc (subject to cache eviction) rather than as a default."""
    last_used: str | None = None
    """The ``"%Y-%m-%d %H:%M:%S"`` timestamp this embedding was last used, for cache eviction."""
    last_checked: str | None = None
    """The ``"%Y-%m-%d %H:%M:%S"`` timestamp this embedding's metadata was last refreshed."""
