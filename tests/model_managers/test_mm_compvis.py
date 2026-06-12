from hordelib.shared_model_manager import SharedModelManager


class TestCompvis:
    def test_compvis_defaults(self, shared_model_manager: type[SharedModelManager]):
        compvis_model_manager = shared_model_manager.manager.compvis
        assert compvis_model_manager is not None
        assert compvis_model_manager.download_model("Deliberate")
