"""First-class model manager for the ControlNet annotator (preprocessor) checkpoints.

The annotator checkpoints are the detector weights ``comfyui_controlnet_aux`` loads to preprocess an image
before a ControlNet runs (pose/depth/edge/etc.). They were historically fetched as an opaque side-channel
(the package's own HuggingFace download, exercised by running each preprocessor). ``horde_model_reference``
now models them as proper ``controlnet_annotator`` records (one record per preprocessor, each carrying its
per-file :class:`~horde_model_reference.model_reference_records.DownloadRecord`s), so this manager surfaces
them through the exact same ``BaseModelManager`` interface every other category uses: enumeration, per-file
on-disk presence, and per-file downloads with progress and checksum verification.

The records' ``file_name`` values are rooted at the controlnet folder's ``annotators/`` subdirectory, and the
``controlnet_annotator`` category resolves to the **same on-disk folder as ``controlnet``** (see
:func:`horde_model_reference.category_folder`). So a file this manager downloads lands at
``<weights_root>/controlnet/annotators/<repo>/<sub>/<file>`` -- exactly where the detector looks, which lets
``comfyui_controlnet_aux`` find the pre-placed file and skip its own fetch.
"""

from typing import override

from horde_model_reference import MODEL_REFERENCE_CATEGORY
from horde_model_reference.model_reference_manager import ModelReferenceManager
from horde_model_reference.model_reference_records import ControlNetAnnotatorModelRecord
from horde_model_reference.source_consts import SourceSelector

from hordelib.model_manager.annotator_provider import ANNOTATOR_SOURCE_ID
from hordelib.model_manager.base import BaseModelManager


class ControlNetAnnotatorModelManager(BaseModelManager[ControlNetAnnotatorModelRecord]):
    """Manages the ControlNet annotator checkpoints as first-class ``controlnet_annotator`` records.

    Records are served by the registered ``comfyui_controlnet_aux`` provider (see
    :class:`hordelib.model_manager.annotator_provider.AnnotatorModelProvider`), so that provider must be
    registered with the :class:`~horde_model_reference.model_reference_manager.ModelReferenceManager` before
    this manager loads its database in ``__init__``.
    """

    def __init__(
        self,
        download_reference=False,
        **kwargs,
    ):
        kwargs.pop("model_category", None)  # consumed by this subclass
        super().__init__(
            model_category=MODEL_REFERENCE_CATEGORY.controlnet_annotator,
            download_reference=download_reference,
            **kwargs,
        )

    @override
    def _reference_source(self, ref_manager: ModelReferenceManager) -> SourceSelector:
        """Load annotator records from the ``comfyui_controlnet_aux`` provider, not canonical data.

        There is no canonical ``controlnet_annotator`` reference file; the records are supplied entirely by
        the registered :class:`~hordelib.model_manager.annotator_provider.AnnotatorModelProvider`, so its
        source must be named explicitly (the default beta-aware selector would read canonical-only and find
        nothing).
        """
        return ANNOTATOR_SOURCE_ID
