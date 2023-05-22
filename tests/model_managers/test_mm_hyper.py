# test_horde.py
import pytest
from PIL import Image

import hordelib

hordelib.initialise()

from hordelib.horde import HordeLib
from hordelib.model_manager.compvis import CompVisModelManager
from hordelib.model_manager.esrgan import EsrganModelManager
from hordelib.shared_model_manager import SharedModelManager


class TestHyperMM:
    @pytest.fixture(autouse=True, scope="class")
    def setup_and_teardown(self):
        TestHyperMM.horde = HordeLib()

        SharedModelManager.loadModelManagers(
            compvis=True,
            blip=True,
            clip=True,
            safety_checker=True,
            esrgan=True,
        )
        assert SharedModelManager.manager is not None
        SharedModelManager.manager.load("RealESRGAN_x4plus")
        SharedModelManager.manager.load("Deliberate")
        SharedModelManager.manager.load("safety_checker")
        yield
        del TestHyperMM.horde
        SharedModelManager._instance = None
        SharedModelManager.manager = None

    def test_get_loaded_models_names(self):
        assert "RealESRGAN_x4plus" in SharedModelManager.manager.get_loaded_models_names()
        assert "Deliberate" in SharedModelManager.manager.get_loaded_models_names()
        assert "safety_checker" in SharedModelManager.manager.get_loaded_models_names()
        assert "Deliberate" in SharedModelManager.manager.get_loaded_models_names(mm_include=["compvis"])
        assert "RealESRGAN_x4plus" not in SharedModelManager.manager.get_loaded_models_names(mm_include=["compvis"])
        assert "safety_checker" not in SharedModelManager.manager.get_loaded_models_names(mm_include=["compvis"])
        assert "Deliberate" in SharedModelManager.manager.get_loaded_models_names(
            mm_include=["compvis", "safety_checker"],
        )
        assert "RealESRGAN_x4plus" not in SharedModelManager.manager.get_loaded_models_names(
            mm_include=["compvis", "safety_checker"],
        )
        assert "safety_checker" in SharedModelManager.manager.get_loaded_models_names(
            mm_include=["compvis", "safety_checker"],
        )
        assert "Deliberate" in SharedModelManager.manager.get_loaded_models_names(mm_exclude=["esrgan"])
        assert "RealESRGAN_x4plus" not in SharedModelManager.manager.get_loaded_models_names(mm_exclude=["esrgan"])
        assert "safety_checker" in SharedModelManager.manager.get_loaded_models_names(mm_exclude=["esrgan"])

    def test_get_available_models_by_types(self):
        assert "RealESRGAN_x4plus" in SharedModelManager.manager.get_available_models_by_types()
        assert "Deliberate" in SharedModelManager.manager.get_available_models_by_types()
        assert "safety_checker" in SharedModelManager.manager.get_available_models_by_types()
        assert "Deliberate" in SharedModelManager.manager.get_available_models_by_types(mm_include=["compvis"])
        assert "RealESRGAN_x4plus" not in SharedModelManager.manager.get_available_models_by_types(
            mm_include=["compvis"],
        )
        assert "safety_checker" not in SharedModelManager.manager.get_available_models_by_types(mm_include=["compvis"])
        assert "Deliberate" in SharedModelManager.manager.get_available_models_by_types(
            mm_include=["compvis", "safety_checker"],
        )
        assert "RealESRGAN_x4plus" not in SharedModelManager.manager.get_available_models_by_types(
            mm_include=["compvis", "safety_checker"],
        )
        assert "safety_checker" in SharedModelManager.manager.get_available_models_by_types(
            mm_include=["compvis", "safety_checker"],
        )
        assert "Deliberate" in SharedModelManager.manager.get_available_models_by_types(mm_exclude=["esrgan"])
        assert "RealESRGAN_x4plus" not in SharedModelManager.manager.get_available_models_by_types(
            mm_exclude=["esrgan"],
        )
        assert "safety_checker" in SharedModelManager.manager.get_available_models_by_types(mm_exclude=["esrgan"])

    def test_get_mm_pointers(self):
        print(SharedModelManager.manager.get_mm_pointers(["compvis"]))
        assert SharedModelManager.manager.get_mm_pointers(["compvis"]) == {SharedModelManager.manager.compvis}
        # because gfpgan MM not active
        assert len(SharedModelManager.manager.get_mm_pointers(["compvis", "gfpgan"])) == 1
        assert len(SharedModelManager.manager.get_mm_pointers(["compvis", "gfpgan", "esrgan"])) == 2
        assert len(SharedModelManager.manager.get_mm_pointers(["compvis", "FAKE"])) == 1
        assert SharedModelManager.manager.get_mm_pointers(None) == set()
        # Any value other than a string or a mm pointer is ignored
        assert SharedModelManager.manager.get_mm_pointers([None]) == set()
        assert SharedModelManager.manager.get_mm_pointers(
            [None, "FAKE", "esrgan", "compvis"],
        ) == {SharedModelManager.manager.compvis, SharedModelManager.manager.esrgan}

        assert SharedModelManager.manager.get_mm_pointers(
            [EsrganModelManager, CompVisModelManager],
        ) == {SharedModelManager.manager.compvis, SharedModelManager.manager.esrgan}
        assert SharedModelManager.manager.get_mm_pointers(
            [None, "FAKE", EsrganModelManager, CompVisModelManager],
        ) == {SharedModelManager.manager.compvis, SharedModelManager.manager.esrgan}
