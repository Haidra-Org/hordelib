"""Textual-inversion resolution must reject incompatible encoders without faulting inference."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

from horde_model_reference.meta_consts import KNOWN_IMAGE_GENERATION_BASELINE
from horde_sdk.ai_horde_api.consts import METADATA_TYPE, METADATA_VALUE

from hordelib.pipeline.context import ModelContext
from hordelib.pipeline.payload import ImageGenPayload, TISpec
from hordelib.pipeline.resolution import _resolve_tis
from hordelib.shared_model_manager import SharedModelManager


def test_anima_drops_sd15_ti_and_reports_baseline_mismatch(monkeypatch) -> None:
    ti_manager = Mock()
    ti_manager.is_local_model.return_value = True
    ti_manager.get_ti_name.return_value = "EasyNegative"
    ti_manager.do_baselines_match.return_value = False
    manager = SimpleNamespace(ti=ti_manager)
    monkeypatch.setattr(SharedModelManager, "manager", manager, raising=False)

    compvis = Mock()
    compvis.get_model_reference_info.return_value = Mock(baseline=KNOWN_IMAGE_GENERATION_BASELINE.anima)
    payload = ImageGenPayload(
        prompt="1girl, firefly",
        negative_prompt="",
        tis=[TISpec(name="7808", strength=1, inject_ti="negprompt")],
        model="Anima-Turbo-v1.1",
    )
    context = ModelContext(
        horde_model_name="Anima-Turbo-v1.1",
        baseline=KNOWN_IMAGE_GENERATION_BASELINE.anima,
    )

    with patch("hordelib.pipeline.resolution._compvis_manager", return_value=compvis):
        faults = _resolve_tis(payload, context)

    assert payload.negative_prompt == ""
    assert len(faults) == 1
    assert faults[0].type_ == METADATA_TYPE.ti
    assert faults[0].value == METADATA_VALUE.baseline_mismatch
    assert faults[0].ref == "EasyNegative"
    ti_manager.get_ti_id.assert_not_called()
    ti_manager.touch_ti.assert_not_called()
