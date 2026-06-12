from horde_model_reference.model_reference_records import GenericModelRecord

from hordelib.consts import MODEL_CATEGORY_NAMES
from hordelib.model_manager.base import BaseModelManager


class SafetyCheckerModelManager(BaseModelManager[GenericModelRecord]):  # FIXME # TODO?
    def __init__(
        self,
        download_reference=False,
        **kwargs,
    ):
        kwargs.pop("model_category_name", None)  # consumed by this subclass
        super().__init__(
            model_category_name=MODEL_CATEGORY_NAMES.safety_checker,
            download_reference=download_reference,
            **kwargs,
        )
