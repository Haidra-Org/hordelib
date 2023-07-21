# test_horde.py
import os

import pytest

from hordelib.consts import EXCLUDED_MODEL_NAMES, MODEL_CATEGORY_NAMES
from hordelib.horde import HordeLib
from hordelib.model_manager.clip import ClipModelManager
from hordelib.model_manager.compvis import CompVisModelManager
from hordelib.model_manager.esrgan import EsrganModelManager
from hordelib.shared_model_manager import SharedModelManager
from hordelib.utils.logger import HordeLog


class TestSharedModelManager:
    def test_singleton(
        self,
        shared_model_manager: type[SharedModelManager],
    ):
        _shared_test_singleton = None if shared_model_manager._instance is None else SharedModelManager()._instance
        a = SharedModelManager()
        b = SharedModelManager()
        assert a.manager is b.manager
        if _shared_test_singleton is not None:
            assert _shared_test_singleton is a
            assert _shared_test_singleton is b

    def test_unload_model_manager(
        self,
        shared_model_manager: type[SharedModelManager],
    ):
        assert shared_model_manager.manager.esrgan is not None
        shared_model_manager.unload_model_managers([EsrganModelManager])
        assert shared_model_manager.manager.esrgan is None
        shared_model_manager.unload_model_managers([CompVisModelManager, MODEL_CATEGORY_NAMES.blip, "clip"])
        assert shared_model_manager.manager.compvis is None
        assert shared_model_manager.manager.blip is None
        assert shared_model_manager.manager.clip is None
        shared_model_manager.load_model_managers(
            [EsrganModelManager, CompVisModelManager, MODEL_CATEGORY_NAMES.blip, "clip"],
        )

    def test_horde_model_manager_reload_db(
        self,
        shared_model_manager: type[SharedModelManager],
    ):
        shared_model_manager.manager.reload_database()

    def test_horde_model_manager_download_model(
        self,
        shared_model_manager: type[SharedModelManager],
    ):
        result: bool | None = shared_model_manager.manager.download_model("Deliberate")
        assert result is True

    def test_horde_model_manager_validate(
        self,
        shared_model_manager: type[SharedModelManager],
    ):
        assert shared_model_manager.manager is not None
        shared_model_manager.manager.load("Deliberate")
        result: bool | None = shared_model_manager.manager.validate_model("Deliberate")
        assert result is True
        shared_model_manager.manager.unload_model("Deliberate")
        if not shared_model_manager.manager.validate_model("Deliberate"):
            assert shared_model_manager.manager.download_model("Deliberate")
            assert shared_model_manager.manager.validate_model("Deliberate")

    def test_model_load_cycling(
        self,
        shared_model_manager: type[SharedModelManager],
    ):
        deliberate_was_loaded = "Deliberate" in shared_model_manager.manager.loaded_models
        shared_model_manager.manager.load("Deliberate")
        shared_model_manager.manager.load("GFPGAN")
        shared_model_manager.manager.load("RealESRGAN_x4plus")
        shared_model_manager.manager.load("4x_NMKD_Superscale_SP")
        assert shared_model_manager.manager.is_model_loaded("Deliberate") is True
        assert shared_model_manager.manager.is_model_loaded("GFPGAN") is True
        assert shared_model_manager.manager.is_model_loaded("RealESRGAN_x4plus") is True
        assert shared_model_manager.manager.is_model_loaded("4x_NMKD_Superscale_SP") is True

        shared_model_manager.manager.unload_model("Deliberate")
        shared_model_manager.manager.unload_model("GFPGAN")
        shared_model_manager.manager.unload_model("RealESRGAN_x4plus")
        shared_model_manager.manager.unload_model("4x_NMKD_Superscale_SP")
        assert shared_model_manager.manager.is_model_loaded("Deliberate") is False
        assert shared_model_manager.manager.is_model_loaded("GFPGAN") is False
        assert shared_model_manager.manager.is_model_loaded("RealESRGAN_x4plus") is False
        assert shared_model_manager.manager.is_model_loaded("4x_NMKD_Superscale_SP") is False
        if deliberate_was_loaded:
            shared_model_manager.manager.load("Deliberate")

    def test_model_excluding(
        self,
        shared_model_manager: type[SharedModelManager],
    ):
        for excluded_model in EXCLUDED_MODEL_NAMES:
            assert not shared_model_manager.manager.load(excluded_model)

    def test_check_sha(
        self,
        shared_model_manager: type[SharedModelManager],
    ):
        """Check the sha256 hashes of all models. If the .sha file doesn't exist, this will write it out."""
        assert shared_model_manager.manager is not None
        for model_manager in shared_model_manager.manager.active_model_managers:
            for model in model_manager.available_models:
                model_file_details = model_manager.get_model_files(model)
                for file in model_file_details:
                    path = file.get("path")
                    if not (".pt" in path or ".ckpt" in path or ".safetensors" in path):
                        continue
                    model_manager.get_file_sha256_hash(f"{model_manager.modelFolderPath}/{path}")

    def test_check_validate_all_available_models(
        self,
        shared_model_manager: type[SharedModelManager],
    ):
        if os.environ.get("TESTS_ONGOING"):
            pytest.skip(
                (
                    "Skipping test_check_validate_all_available_models because it takes too long and could tamper "
                    "with CI environment."
                ),
            )
        assert shared_model_manager.manager is not None
        for model_manager in shared_model_manager.manager.active_model_managers:
            for model in model_manager.available_models:
                if not model_manager.validate_model(model):
                    assert model_manager.download_model(model)
                    assert model_manager.validate_model(model)

    def test_preload_annotators(
        self,
        shared_model_manager: type[SharedModelManager],
    ):
        assert shared_model_manager.preloadAnnotators()

    @pytest.fixture(scope="class")
    def load_models_for_type_test(
        self,
        shared_model_manager: type[SharedModelManager],
    ):
        shared_model_manager.manager.load("RealESRGAN_x4plus")
        shared_model_manager.manager.load("Deliberate")
        shared_model_manager.manager.load("safety_checker")
        yield
        shared_model_manager.manager.unload_model("RealESRGAN_x4plus")
        shared_model_manager.manager.unload_model("Deliberate")
        shared_model_manager.manager.unload_model("safety_checker")

    def test_get_loaded_models_names(
        self,
        shared_model_manager: type[SharedModelManager],
        load_models_for_type_test,
    ):
        assert "RealESRGAN_x4plus" in shared_model_manager.manager.get_loaded_models_names()
        assert "Deliberate" in shared_model_manager.manager.get_loaded_models_names()
        assert "safety_checker" in shared_model_manager.manager.get_loaded_models_names()
        assert "Deliberate" in shared_model_manager.manager.get_loaded_models_names(mm_include=["compvis"])
        assert "RealESRGAN_x4plus" not in shared_model_manager.manager.get_loaded_models_names(mm_include=["compvis"])
        assert "safety_checker" not in shared_model_manager.manager.get_loaded_models_names(mm_include=["compvis"])
        assert "Deliberate" in shared_model_manager.manager.get_loaded_models_names(
            mm_include=["compvis", "safety_checker"],
        )
        assert "RealESRGAN_x4plus" not in shared_model_manager.manager.get_loaded_models_names(
            mm_include=["compvis", "safety_checker"],
        )
        assert "safety_checker" in shared_model_manager.manager.get_loaded_models_names(
            mm_include=["compvis", "safety_checker"],
        )
        assert "Deliberate" in shared_model_manager.manager.get_loaded_models_names(mm_exclude=["esrgan"])
        assert "RealESRGAN_x4plus" not in shared_model_manager.manager.get_loaded_models_names(mm_exclude=["esrgan"])
        assert "safety_checker" in shared_model_manager.manager.get_loaded_models_names(mm_exclude=["esrgan"])

    def test_get_available_models_by_types(
        self,
        shared_model_manager: type[SharedModelManager],
        load_models_for_type_test,
    ):
        assert "RealESRGAN_x4plus" in shared_model_manager.manager.get_available_models_by_types()
        assert "Deliberate" in shared_model_manager.manager.get_available_models_by_types()
        assert "safety_checker" in shared_model_manager.manager.get_available_models_by_types()
        assert "Deliberate" in shared_model_manager.manager.get_available_models_by_types(mm_include=["compvis"])
        assert "RealESRGAN_x4plus" not in shared_model_manager.manager.get_available_models_by_types(
            mm_include=["compvis"],
        )
        assert "safety_checker" not in shared_model_manager.manager.get_available_models_by_types(
            mm_include=["compvis"],
        )
        assert "Deliberate" in shared_model_manager.manager.get_available_models_by_types(
            mm_include=["compvis", "safety_checker"],
        )
        assert "RealESRGAN_x4plus" not in shared_model_manager.manager.get_available_models_by_types(
            mm_include=["compvis", "safety_checker"],
        )
        assert "safety_checker" in shared_model_manager.manager.get_available_models_by_types(
            mm_include=["compvis", "safety_checker"],
        )
        assert "Deliberate" in shared_model_manager.manager.get_available_models_by_types(mm_exclude=["esrgan"])
        assert "RealESRGAN_x4plus" not in shared_model_manager.manager.get_available_models_by_types(
            mm_exclude=["esrgan"],
        )
        assert "safety_checker" in shared_model_manager.manager.get_available_models_by_types(mm_exclude=["esrgan"])

    def test_get_mm_pointers(
        self,
        shared_model_manager: type[SharedModelManager],
        load_models_for_type_test,
    ):
        shared_model_manager.unload_model_managers(["gfpgan"])
        print(shared_model_manager.manager.get_mm_pointers(["compvis"]))
        assert shared_model_manager.manager.get_mm_pointers(["compvis"]) == [shared_model_manager.manager.compvis]
        # because gfpgan MM not active
        assert len(shared_model_manager.manager.get_mm_pointers(["compvis", "gfpgan"])) == 1
        assert len(shared_model_manager.manager.get_mm_pointers(["compvis", "gfpgan", "esrgan"])) == 2
        assert len(shared_model_manager.manager.get_mm_pointers(["compvis", "FAKE"])) == 1
        assert shared_model_manager.manager.get_mm_pointers(None) == []
        # Any value other than a string or a mm pointer is ignored
        assert shared_model_manager.manager.get_mm_pointers([None]) == []  # type: ignore
        assert set(
            shared_model_manager.manager.get_mm_pointers(
                [None, "FAKE", "esrgan", "compvis"],  # type: ignore
            ),
        ) == {shared_model_manager.manager.compvis, shared_model_manager.manager.esrgan}

        assert set(
            shared_model_manager.manager.get_mm_pointers(
                [EsrganModelManager, CompVisModelManager],
            ),
        ) == {shared_model_manager.manager.compvis, shared_model_manager.manager.esrgan}
        assert set(
            shared_model_manager.manager.get_mm_pointers(
                [None, "FAKE", EsrganModelManager, CompVisModelManager],  # type: ignore
            ),
        ) == {shared_model_manager.manager.compvis, shared_model_manager.manager.esrgan}
        shared_model_manager.load_model_managers(["gfpgan"])
