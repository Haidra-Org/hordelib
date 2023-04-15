# shared_model_manager.py
import builtins
from pathlib import Path
from hordelib.model_manager.hyper import ModelManager
from hordelib.cache import get_cache_directory
from hordelib.preload import download_all_controlnet_annotators


class SharedModelManager:
    _instance = None
    manager: ModelManager | None = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def loadModelManagers(
        cls,
        # aitemplate: bool = False,
        blip: bool = False,
        clip: bool = False,
        codeformer: bool = False,
        compvis: bool = False,
        controlnet: bool = False,
        diffusers: bool = False,
        esrgan: bool = False,
        gfpgan: bool = False,
        safety_checker: bool = False,
    ):
        if cls.manager is None:
            cls.manager = ModelManager()

        args_passed = locals().copy()  # XXX This is temporary
        args_passed.pop("cls")  # XXX This is temporary

        cls.manager.init_model_managers(**args_passed)
        builtins.annotator_ckpts_path = Path(get_cache_directory()).joinpath("controlnet").joinpath("annotator")
        # XXX # FIXME _PLEASE_

    @classmethod
    def preloadAnnotators(cls):
        download_all_controlnet_annotators()
