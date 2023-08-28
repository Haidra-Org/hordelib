from typing import Generator

import pytest

import hordelib
from hordelib.model_manager.compvis import CompVisModelManager


class TestCompvis:
    @pytest.fixture(scope="class", autouse=True)
    def compvis_model_manager(self, init_horde) -> Generator[CompVisModelManager, None, None]:
        yield CompVisModelManager()

    def test_compvis_defaults(self, compvis_model_manager: CompVisModelManager):
        assert compvis_model_manager.download_model("Deliberate")
