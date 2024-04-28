# test_horde.py
import os

import pytest

from hordelib.model_manager.compvis import CompVisModelManager
from hordelib.model_manager.esrgan import EsrganModelManager
from hordelib.shared_model_manager import SharedModelManager


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
        shared_model_manager.load_model_managers([EsrganModelManager])
        assert shared_model_manager.manager.esrgan is not None
        shared_model_manager.unload_model_managers([EsrganModelManager])
        assert shared_model_manager.manager.esrgan is None
        shared_model_manager.load_model_managers([EsrganModelManager])

    def test_horde_model_manager_reload_db(
        self,
        shared_model_manager: type[SharedModelManager],
    ):
        shared_model_manager.manager.reload_database()

    def test_horde_model_manager_download_model(
        self,
        shared_model_manager: type[SharedModelManager],
    ):
        assert shared_model_manager.manager.download_model("Deliberate")
        assert shared_model_manager.manager.is_model_available("Deliberate")

    def test_horde_model_manager_validate(
        self,
        shared_model_manager: type[SharedModelManager],
    ):
        assert shared_model_manager.manager is not None
        assert shared_model_manager.manager.download_model("Deliberate")
        if not shared_model_manager.manager.validate_model("Deliberate"):
            assert shared_model_manager.manager.download_model("Deliberate")
            assert shared_model_manager.manager.validate_model("Deliberate")

    def test_check_sha(
        self,
        shared_model_manager: type[SharedModelManager],
    ):
        """Check the sha256 hashes of all models. If the .sha file doesn't exist, this will write it out."""
        assert shared_model_manager.manager is not None
        for model_manager in shared_model_manager.manager.active_model_managers:
            for model in model_manager.available_models:
                model_file_details = model_manager.get_model_config_files(model)
                for file in model_file_details:
                    path = file.get("path")
                    if not isinstance(path, str):
                        continue
                    if not (".pt" in path or ".ckpt" in path or ".safetensors" in path):
                        continue
                    # Check if `path` is already a full path
                    if os.path.isabs(path):
                        model_manager.get_file_sha256_hash(path)
                    else:
                        model_manager.get_file_sha256_hash(f"{model_manager.model_folder_path}/{path}")

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
        assert shared_model_manager.preload_annotators()

    @pytest.fixture(scope="class")
    def load_models_for_type_test(
        self,
        shared_model_manager: type[SharedModelManager],
    ):
        shared_model_manager.manager.download_model("RealESRGAN_x4plus")
        shared_model_manager.manager.download_model("Deliberate")
        shared_model_manager.manager.download_model("safety_checker")
        yield

    def test_get_available_models(
        self,
        shared_model_manager: type[SharedModelManager],
        load_models_for_type_test,
    ):
        assert "RealESRGAN_x4plus" in shared_model_manager.manager.get_available_models()
        assert "Deliberate" in shared_model_manager.manager.get_available_models()
        assert "SDXL 1.0" in shared_model_manager.manager.get_available_models()
        assert "safety_checker" in shared_model_manager.manager.get_available_models()
        assert "Deliberate" in shared_model_manager.manager.get_available_models(mm_include=["compvis"])
        assert "RealESRGAN_x4plus" not in shared_model_manager.manager.get_available_models(mm_include=["compvis"])
        assert "safety_checker" not in shared_model_manager.manager.get_available_models(mm_include=["compvis"])
        assert "Deliberate" in shared_model_manager.manager.get_available_models(
            mm_include=["compvis", "safety_checker"],
        )
        assert "RealESRGAN_x4plus" not in shared_model_manager.manager.get_available_models(
            mm_include=["compvis", "safety_checker"],
        )
        assert "safety_checker" in shared_model_manager.manager.get_available_models(
            mm_include=["compvis", "safety_checker"],
        )
        assert "Deliberate" in shared_model_manager.manager.get_available_models(mm_exclude=["esrgan"])
        assert "RealESRGAN_x4plus" not in shared_model_manager.manager.get_available_models(mm_exclude=["esrgan"])
        assert "safety_checker" in shared_model_manager.manager.get_available_models(mm_exclude=["esrgan"])

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
        print(shared_model_manager.manager.get_model_manager_instances(["compvis"]))
        assert shared_model_manager.manager.get_model_manager_instances(["compvis"]) == [
            shared_model_manager.manager.compvis,
        ]
        # because gfpgan MM not active
        assert len(shared_model_manager.manager.get_model_manager_instances(["compvis", "gfpgan"])) == 1
        assert len(shared_model_manager.manager.get_model_manager_instances(["compvis", "gfpgan", "esrgan"])) == 2
        assert len(shared_model_manager.manager.get_model_manager_instances(["compvis", "FAKE"])) == 1
        assert shared_model_manager.manager.get_model_manager_instances(None) == []
        # Any value other than a string or a mm pointer is ignored
        assert shared_model_manager.manager.get_model_manager_instances([None]) == []  # type: ignore
        assert set(
            shared_model_manager.manager.get_model_manager_instances(
                [None, "FAKE", "esrgan", "compvis"],  # type: ignore
            ),
        ) == {shared_model_manager.manager.compvis, shared_model_manager.manager.esrgan}

        assert set(
            shared_model_manager.manager.get_model_manager_instances(
                [EsrganModelManager, CompVisModelManager],
            ),
        ) == {shared_model_manager.manager.compvis, shared_model_manager.manager.esrgan}
        assert set(
            shared_model_manager.manager.get_model_manager_instances(
                [None, "FAKE", EsrganModelManager, CompVisModelManager],  # type: ignore
            ),
        ) == {shared_model_manager.manager.compvis, shared_model_manager.manager.esrgan}
        shared_model_manager.load_model_managers(["gfpgan"])
