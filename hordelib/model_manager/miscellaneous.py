from hordelib.consts import MODEL_CATEGORY_NAMES
from hordelib.model_manager.base import BaseModelManager


class MiscellaneousModelManager(BaseModelManager):  # FIXME # TODO?
    def __init__(
        self,
        download_reference=False,
        **kwargs,
    ):
        super().__init__(
            model_category_name=MODEL_CATEGORY_NAMES.miscellaneous,
            download_reference=download_reference,
            **kwargs,
        )
