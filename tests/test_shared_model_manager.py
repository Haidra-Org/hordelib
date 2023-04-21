# test_horde.py
import glob
import pathlib

import pytest

from hordelib.cache import get_cache_directory
from hordelib.horde import HordeLib
from hordelib.shared_model_manager import SharedModelManager


class TestSharedModelManager:
    horde = HordeLib()
    default_model_manager_args: dict

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        self.horde = HordeLib()

        self.default_model_manager_args = {  # XXX # TODO
            # aitemplate
            "blip": True,
            "clip": True,
            "codeformer": True,
            "compvis": True,
            "controlnet": True,
            "diffusers": True,
            "esrgan": True,
            "gfpgan": True,
            "safety_checker": True,
        }
        SharedModelManager.loadModelManagers(**self.default_model_manager_args)
        assert SharedModelManager.manager is not None
        yield
        del self.horde
        SharedModelManager._instance = None
        SharedModelManager.manager = None

    def test_singleton(self):
        a = SharedModelManager()
        b = SharedModelManager()
        assert a.manager is b.manager

    def test_horde_model_manager_init(self):
        assert SharedModelManager.manager is not None
        # assert SharedModelManager.manager.aitemplate is not None # XXX # FIXME
        assert SharedModelManager.manager.blip is not None  # XXX
        assert SharedModelManager.manager.clip is not None
        assert SharedModelManager.manager.codeformer is not None
        assert SharedModelManager.manager.compvis is not None
        assert SharedModelManager.manager.controlnet is not None
        assert SharedModelManager.manager.diffusers is not None
        assert SharedModelManager.manager.esrgan is not None
        assert SharedModelManager.manager.gfpgan is not None
        assert SharedModelManager.manager.safety_checker is not None

    def test_horde_model_manager_reload_db(self):
        assert SharedModelManager.manager is not None
        SharedModelManager.manager.reload_database()

    def test_horde_model_manager_download_model(self):
        assert SharedModelManager.manager is not None
        result: bool | None = SharedModelManager.manager.download_model("Deliberate")
        assert result is True

    def test_horde_model_manager_validate(self):
        assert SharedModelManager.manager is not None
        SharedModelManager.manager.load("Deliberate")
        result: bool | None = SharedModelManager.manager.validate_model("Deliberate")
        assert result is True

    def test_taint_models(self):
        assert SharedModelManager.manager is not None
        SharedModelManager.manager.taint_models(["Deliberate"])
        assert "Deliberate" not in SharedModelManager.manager.get_available_models()
        assert SharedModelManager.manager.is_model_loaded("Deliberate") is False

    # XXX add a test for model missing?
    def test_horde_model_manager_unload_model(self):
        assert SharedModelManager.manager is not None
        SharedModelManager.manager.load("Deliberate")
        assert SharedModelManager.manager.is_model_loaded("Deliberate") is True
        result = SharedModelManager.manager.unload_model("Deliberate")
        assert result is True
        assert SharedModelManager.manager.is_model_loaded("Deliberate") is False

    def test_model_load_checking(self):
        assert SharedModelManager.manager is not None
        assert SharedModelManager.manager.is_model_loaded("Deliberate") is False
        assert SharedModelManager.manager.is_model_loaded("GFPGAN") is False
        assert SharedModelManager.manager.is_model_loaded("RealESRGAN_x4plus") is False
        SharedModelManager.manager.load("Deliberate")
        SharedModelManager.manager.load("GFPGAN")
        SharedModelManager.manager.load("RealESRGAN_x4plus")
        assert SharedModelManager.manager.is_model_loaded("Deliberate") is True
        assert SharedModelManager.manager.is_model_loaded("GFPGAN") is True
        assert SharedModelManager.manager.is_model_loaded("RealESRGAN_x4plus") is True

    def test_check_sha(self):
        """Check the sha256 hashes of all models. If the .sha file doesn't exist, this will write it out."""
        assert SharedModelManager.manager is not None
        for model_manager in SharedModelManager.manager.active_model_managers:
            for model in model_manager.available_models:
                model_file_details = model_manager.get_model_files(model)
                for file in model_file_details:
                    path = file.get("path")
                    if not (".pt" in path or ".ckpt" in path or ".safetensors" in path):
                        continue
                    model_manager.get_file_sha256_hash(f"{model_manager.modelFolderPath}/{path}")
        pass

    def test_check_validate_all_available_models(self):
        assert SharedModelManager.manager is not None
        for model_manager in SharedModelManager.manager.active_model_managers:
            for model in model_manager.available_models:
                assert model_manager.validate_model(model)

    def test_preload_annotators(self):
        assert SharedModelManager.preloadAnnotators()
