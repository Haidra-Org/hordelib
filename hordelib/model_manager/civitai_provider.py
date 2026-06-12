"""A :class:`~horde_model_reference.providers.base.ModelProvider` exposing the horde's CivitAI data.

The LoRA and Textual Inversion managers fetch and cache model records from CivitAI at runtime. This
provider is the standardized, read-only window onto whatever those managers currently know: register
it with a :class:`~horde_model_reference.model_reference_manager.ModelReferenceManager` and consumers
can read LoRA/TI records through the same ``manager.query(category, source="civitai")`` surface used
for every other category, instead of reaching into hordelib's managers directly.

The managers remain the download/cache engine; this provider never triggers downloads. It simply
delegates :meth:`fetch_category` to the owning manager's current in-memory records.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, override

from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORY
from horde_model_reference.model_reference_records import GenericModelRecord
from horde_model_reference.providers.base import ModelProvider

CIVITAI_SOURCE_ID = "civitai"
"""The stable provider source id under which the horde's CivitAI records are registered."""


class SupportsCurrentRecords(Protocol):
    """A manager that can hand out a snapshot of its current model records.

    Implemented by :class:`~hordelib.model_manager.civitai_adhoc.CivitaiAdhocModelManager`; declared as
    a protocol here to keep this provider decoupled from the manager class hierarchy.
    """

    def current_records(self) -> Mapping[str, GenericModelRecord]:
        """Return a snapshot mapping of ``model_name -> record`` for this manager's category."""
        ...


class CivitaiModelProvider(ModelProvider):
    """Serves the horde's runtime-fetched CivitAI LoRA and TI records under the ``"civitai"`` source.

    Subclass Integration:
        Construct with a mapping of category to the manager that owns that category's records. The
        provider advertises exactly those categories and reads each manager's live snapshot on demand.
    """

    def __init__(
        self,
        managers_by_category: Mapping[MODEL_REFERENCE_CATEGORY, SupportsCurrentRecords],
    ) -> None:
        """Store the per-category managers this provider reads from.

        Args:
            managers_by_category: Mapping of category (e.g. ``MODEL_REFERENCE_CATEGORY.lora``) to the
                manager whose :meth:`SupportsCurrentRecords.current_records` should serve it.
        """
        self._managers_by_category: dict[MODEL_REFERENCE_CATEGORY, SupportsCurrentRecords] = dict(
            managers_by_category,
        )

    @property
    @override
    def source_id(self) -> str:
        """Return the stable ``"civitai"`` source id."""
        return CIVITAI_SOURCE_ID

    @override
    def provided_categories(self) -> set[MODEL_REFERENCE_CATEGORY | str]:
        """Return the categories backed by a registered manager."""
        return set(self._managers_by_category)

    @override
    def fetch_category(
        self,
        category: MODEL_REFERENCE_CATEGORY | str,
        *,
        force_refresh: bool = False,
    ) -> dict[str, GenericModelRecord] | None:
        """Return the owning manager's current records for *category*, or ``None`` if unserved.

        ``force_refresh`` is accepted for interface compatibility but ignored: the managers refresh on
        their own schedule and this provider only ever reflects their current state.
        """
        manager = self._managers_by_category.get(category)  # type: ignore[arg-type]
        if manager is None:
            return None
        return dict(manager.current_records())
