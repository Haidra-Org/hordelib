from horde_model_reference import MODEL_REFERENCE_CATEGORY
from horde_model_reference.model_reference_records import GenericModelRecord

from hordelib.model_manager.base import BaseModelManager


class CodeFormerModelManager(BaseModelManager[GenericModelRecord]):
    def __init__(self, download_reference=False, **kwargs):
        kwargs.pop("model_category", None)  # consumed by this subclass
        super().__init__(
            model_category=MODEL_REFERENCE_CATEGORY.codeformer,
            download_reference=download_reference,
            **kwargs,
        )
