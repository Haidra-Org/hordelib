"""hordelib's presence check agrees with horde_model_reference's canonical one.

These run hordelib's *real* ``BaseModelManager.is_model_available`` against the same on-disk files as
``horde_model_reference.on_disk_layout.is_present`` for a Z-Image-Turbo-shaped 3-file record, guarding
that presence is answered identically by both: existence-only (no ``.sha256`` sidecar gate) and
multi-root (extra weights directories are searched). hordelib delegates presence to the canonical check;
integrity (checksums) is a separate concern handled by ``validate_model``.

The manager is built bare (``object.__new__`` + the attributes the presence path reads) so the real
code runs without ``hordelib.initialise()``, a GPU, the reference singleton, or the network.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from horde_model_reference.meta_consts import KNOWN_IMAGE_GENERATION_BASELINE, MODEL_REFERENCE_CATEGORY
from horde_model_reference.model_reference_records import (
    DownloadRecord,
    GenericModelRecordConfig,
    ImageGenerationModelRecord,
)
from horde_model_reference.on_disk_layout import is_present

from hordelib.model_manager.base import BaseModelManager

_MODEL_NAME = "Z-Image-Turbo"
_ZIMAGE_FILES: tuple[tuple[str, str], ...] = (
    ("z_image_turbo_bf16.safetensors", "unet"),
    ("ae.safetensors", "vae"),
    ("qwen_3_4b.safetensors", "text_encoders"),
)


def _zimage_record() -> ImageGenerationModelRecord:
    return ImageGenerationModelRecord(
        name=_MODEL_NAME,
        baseline=KNOWN_IMAGE_GENERATION_BASELINE.z_image_turbo,
        nsfw=True,
        description="Z-Image-Turbo presence-consistency record",
        size_on_disk_bytes=20_430_635_136,
        config=GenericModelRecordConfig(
            download=[
                DownloadRecord(
                    file_name=file_name,
                    file_url=f"https://example.com/{file_name}",
                    file_purpose=file_purpose,
                )
                for file_name, file_purpose in _ZIMAGE_FILES
            ],
        ),
    )


def _bare_compvis_manager(weights_root: Path) -> BaseModelManager:
    """A BaseModelManager with only the attributes the presence path reads (no heavy __init__)."""
    manager = object.__new__(BaseModelManager)  # type: ignore[call-overload]
    manager._model_category = MODEL_REFERENCE_CATEGORY.image_generation  # type: ignore[attr-defined]
    manager.model_reference = {_MODEL_NAME: _zimage_record()}  # type: ignore[attr-defined]
    manager.tainted_models = []
    manager._weights_root = weights_root  # type: ignore[attr-defined]
    manager.model_folder_path = weights_root / "compvis"
    return manager


def _place_all_files(root: Path) -> None:
    targets = {
        "z_image_turbo_bf16.safetensors": root / "compvis",
        "ae.safetensors": root / "vae",
        "qwen_3_4b.safetensors": root / "text_encoders",
    }
    for file_name, folder in targets.items():
        folder.mkdir(parents=True, exist_ok=True)
        (folder / file_name).write_bytes(b"x")


def test_present_without_sidecar_agrees_with_canonical(tmp_path: Path) -> None:
    """A bare weight (no .sha256 sidecar) counts present in both: presence is existence, not integrity."""
    _place_all_files(tmp_path)
    manager = _bare_compvis_manager(tmp_path)

    assert is_present(_zimage_record(), tmp_path) is True
    assert manager.is_model_available(_MODEL_NAME) is True


def test_absent_when_a_component_missing_agrees_with_canonical(tmp_path: Path) -> None:
    """Only the unet on disk is not-present in both: every declared file must exist."""
    (tmp_path / "compvis").mkdir(parents=True)
    (tmp_path / "compvis" / "z_image_turbo_bf16.safetensors").write_bytes(b"x")
    manager = _bare_compvis_manager(tmp_path)

    assert is_present(_zimage_record(), tmp_path) is False
    assert manager.is_model_available(_MODEL_NAME) is False


def test_present_via_extra_root_agrees_with_canonical(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Weights on a second model root count present when AIWORKER_EXTRA_MODEL_DIRECTORIES names it."""
    primary = tmp_path / "primary"
    extra = tmp_path / "extra"
    _place_all_files(extra)
    (primary / "compvis").mkdir(parents=True)
    manager = _bare_compvis_manager(primary)

    # Without the extra root configured, neither sees the files (single primary root is empty).
    monkeypatch.delenv("AIWORKER_EXTRA_MODEL_DIRECTORIES", raising=False)
    assert manager.is_model_available(_MODEL_NAME) is False
    assert is_present(_zimage_record(), primary) is False

    # With the extra root configured, both find the spread-out weights.
    monkeypatch.setenv("AIWORKER_EXTRA_MODEL_DIRECTORIES", str(extra))
    assert manager.is_model_available(_MODEL_NAME) is True
    assert is_present(_zimage_record(), primary, extra_roots=[extra]) is True
