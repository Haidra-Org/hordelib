"""Custom checkpoint flags must retain their boolean meaning during normalization."""

import pytest
from pydantic import ValidationError

from hordelib.model_manager.compvis import _custom_model_entry_to_record


@pytest.mark.parametrize(
    "value,expected", [(False, False), (True, True), ("false", False), ("true", True), ("0", False)]
)
def test_custom_inpainting_flag(value: bool | str, expected: bool) -> None:
    record = _custom_model_entry_to_record(
        "custom",
        {
            "inpainting": value,
            "config": {"files": [{"path": "custom.safetensors"}]},
        },
    )
    assert record.inpainting is expected


def test_invalid_custom_inpainting_flag_is_rejected() -> None:
    with pytest.raises(ValidationError):
        _custom_model_entry_to_record(
            "custom",
            {
                "inpainting": "not-a-boolean",
                "config": {"files": [{"path": "custom.safetensors"}]},
            },
        )


def test_missing_custom_inpainting_flag_defaults_false() -> None:
    record = _custom_model_entry_to_record("custom", {"config": {"files": [{"path": "custom.safetensors"}]}})
    assert record.inpainting is False
