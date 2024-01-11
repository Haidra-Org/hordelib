from hordelib.consts import MODEL_CATEGORY_NAMES
from hordelib.model_manager.base import BaseModelManager


class GfpganModelManager(BaseModelManager):
    def __init__(
        self,
        download_reference=False,
        civitai_api_token: str | None = None,
        **kwargs,
    ):
        super().__init__(
            model_category_name=MODEL_CATEGORY_NAMES.gfpgan,
            download_reference=download_reference,
            **kwargs,
        )
