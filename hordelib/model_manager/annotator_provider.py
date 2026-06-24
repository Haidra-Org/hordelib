"""A :class:`~horde_model_reference.providers.base.ModelProvider` exposing the ControlNet annotators.

The ControlNet annotator checkpoints are fetched on the worker by the installed ``comfyui_controlnet_aux``
package (via the unified download engine, falling back to the package's own HuggingFace downloader). This
provider makes that set a first-class, queryable source: register it with a
:class:`~horde_model_reference.model_reference_manager.ModelReferenceManager` and consumers can read annotator
records through the same ``manager.query(MODEL_REFERENCE_CATEGORY.controlnet_annotator, source=...)`` surface
used for every other category, instead of the previously opaque internal download.

The authoritative annotator set lives in :mod:`horde_model_reference.annotator_records` (derived from the
pinned package). This provider simply surfaces it; it never triggers downloads.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, override

from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORY
from horde_model_reference.providers.base import ModelProvider

if TYPE_CHECKING:
    from horde_model_reference.model_reference_records import GenericModelRecord

ANNOTATOR_SOURCE_ID = "comfyui_controlnet_aux"
"""The stable provider source id under which the ControlNet annotator records are registered."""


class AnnotatorModelProvider(ModelProvider):
    """Serves the ControlNet annotator records under the ``"comfyui_controlnet_aux"`` source.

    The records come from :func:`horde_model_reference.annotator_records.annotator_records`, the verified set
    the pinned ``comfyui_controlnet_aux`` actually loads. The provider is read-only and never refreshes.
    """

    @property
    @override
    def source_id(self) -> str:
        """Return the stable ``"comfyui_controlnet_aux"`` source id."""
        return ANNOTATOR_SOURCE_ID

    @override
    def provided_categories(self) -> set[MODEL_REFERENCE_CATEGORY | str]:
        """Return the single category this provider serves (``controlnet_annotator``)."""
        return {MODEL_REFERENCE_CATEGORY.controlnet_annotator}

    @override
    def fetch_category(
        self,
        category: MODEL_REFERENCE_CATEGORY | str,
        *,
        force_refresh: bool = False,
    ) -> dict[str, GenericModelRecord] | None:
        """Return the annotator records for *category*, or ``None`` if this provider does not serve it.

        ``force_refresh`` is accepted for interface compatibility but ignored: the set is a fixed, in-package
        catalog that only changes with a ``comfyui_controlnet_aux`` pin bump (a code change, not a refresh).
        """
        if category != MODEL_REFERENCE_CATEGORY.controlnet_annotator:
            return None
        from horde_model_reference.annotator_records import annotator_records

        return dict(annotator_records())
