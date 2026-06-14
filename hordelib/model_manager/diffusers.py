from horde_model_reference import MODEL_REFERENCE_CATEGORY

from hordelib.model_manager.base import BaseModelManager


class DiffusersModelManager(BaseModelManager):
    def __init__(
        self,
        download_reference=False,
        **kwargs,
    ):
        raise NotImplementedError("Diffusers are not yet supported")

        super().__init__(
            model_category=MODEL_REFERENCE_CATEGORY.image_generation,
            download_reference=download_reference,
            **kwargs,
        )
