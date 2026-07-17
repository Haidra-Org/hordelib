"""GPU-free tests for CompVisModelManager's component-identity sidecar creation and sweep.

These pin the download-time and sweep behaviour: a successful checkpoint download builds a sidecar and
extracts the embedded VAE into the sibling ``vae`` folder; the standalone sweep does the same for every
on-disk image checkpoint; a legitimate ``.ckpt`` pickle (which cannot be identified torch-free) is skipped
with a log rather than aborting the sweep; and a second sweep is idempotent (no rewrite of a fresh sidecar).

The manager is built via ``__new__`` with only the attributes the methods under test touch (mirroring
``test_base_download_delegation.py``), and checkpoints are synthetic safetensors written in-test, so no GPU,
network, or ModelReferenceManager initialisation is needed.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest
from horde_model_reference import MODEL_REFERENCE_CATEGORY, category_folder
from horde_model_reference.component_identity import read_sidecar, sidecar_path_for
from horde_model_reference.meta_consts import MODEL_DOMAIN, MODEL_PURPOSE, ModelClassification
from horde_model_reference.model_reference_records import (
    DownloadRecord,
    GenericModelRecordConfig,
    ImageGenerationModelRecord,
)
from loguru import logger

from hordelib.model_manager import base as base_module
from hordelib.model_manager.compvis import CompVisModelManager

_UNET = ("model.diffusion_model.x", "F16", (2,), bytes(range(40, 44)))
_VAE = [
    ("first_stage_model.decoder.conv_in.weight", "F16", (2, 2), bytes(range(1, 9))),
    ("first_stage_model.encoder.conv_out.weight", "F32", (1, 2), bytes(range(16, 24))),
]
_OTHER_UNET = ("model.diffusion_model.y", "F32", (4,), bytes(range(70, 86)))


def _build_safetensors(tensors: list[tuple[str, str, tuple[int, ...], bytes]]) -> bytes:
    header: dict[str, object] = {}
    buffer = bytearray()
    for name, dtype, shape, data in tensors:
        begin = len(buffer)
        buffer += data
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [begin, len(buffer)]}
    header_json = json.dumps(header).encode("utf-8")
    return struct.pack("<Q", len(header_json)) + header_json + bytes(buffer)


def _make_manager(weights_root: Path) -> CompVisModelManager:
    """Build a compvis manager with only the state the sidecar methods touch, bypassing __init__."""
    manager = CompVisModelManager.__new__(CompVisModelManager)
    manager._model_category = MODEL_REFERENCE_CATEGORY.image_generation
    manager._weights_root = weights_root
    folder = category_folder(MODEL_REFERENCE_CATEGORY.image_generation)
    assert folder is not None
    manager.model_folder_path = weights_root / folder
    manager.model_folder_path.mkdir(parents=True, exist_ok=True)
    manager.model_reference = {}
    manager.available_models = []
    manager.tainted_models = []
    return manager


def _record(file_name: str) -> ImageGenerationModelRecord:
    return ImageGenerationModelRecord(
        record_type=MODEL_REFERENCE_CATEGORY.image_generation,
        name="m",
        description="d",
        baseline="stable_diffusion_1",
        nsfw=False,
        inpainting=False,
        model_classification=ModelClassification(domain=MODEL_DOMAIN.image, purpose=MODEL_PURPOSE.generation),
        config=GenericModelRecordConfig(download=[DownloadRecord(file_name=file_name, file_url="")]),
    )


def _write_checkpoint(manager: CompVisModelManager, file_name: str, unet: tuple) -> Path:
    path = manager.model_folder_path / file_name
    path.write_bytes(_build_safetensors([unet, *_VAE]))
    return path


@pytest.fixture(autouse=True)
def _no_extra_roots(monkeypatch: pytest.MonkeyPatch) -> None:
    """is_model_available consults an env-driven extra-roots list; keep it empty for deterministic presence."""
    monkeypatch.delenv("AIWORKER_EXTRA_MODEL_DIRECTORIES", raising=False)


def test_download_model_builds_sidecar_and_extracts_vae(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manager = _make_manager(tmp_path)
    ckpt = _write_checkpoint(manager, "model.safetensors", _UNET)
    manager.model_reference = {"model": _record("model.safetensors")}

    # The engine is not exercised here; the file already exists, so only the post-download sidecar hook runs.
    monkeypatch.setattr(base_module.download_engine, "download_record_files", lambda *a, **k: True)
    monkeypatch.setattr(manager, "validate_model", lambda name: True)

    assert manager.download_model("model") is True

    sidecar = read_sidecar(ckpt)
    assert sidecar is not None
    vae_entry = sidecar.embedded["vae"]
    assert vae_entry.extracted_file_name is not None
    extracted = tmp_path / "vae" / vae_entry.extracted_file_name
    assert extracted.exists()


def test_sweep_builds_sidecars_for_present_models(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    ckpt_a = _write_checkpoint(manager, "a.safetensors", _UNET)
    ckpt_b = _write_checkpoint(manager, "b.safetensors", _OTHER_UNET)
    manager.model_reference = {"a": _record("a.safetensors"), "b": _record("b.safetensors")}

    manager.ensure_component_identity_sweep()

    assert sidecar_path_for(ckpt_a).exists()
    assert sidecar_path_for(ckpt_b).exists()
    # Byte-identical embedded VAEs deduplicate onto one content-addressed extracted file.
    name_a = read_sidecar(ckpt_a).embedded["vae"].extracted_file_name  # type: ignore[union-attr]
    name_b = read_sidecar(ckpt_b).embedded["vae"].extracted_file_name  # type: ignore[union-attr]
    assert name_a == name_b
    assert (tmp_path / "vae" / name_a).exists()


def test_sweep_skips_missing_models(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    # Reference names a model whose file was never written; the sweep must not create a sidecar for it.
    manager.model_reference = {"absent": _record("absent.safetensors")}

    manager.ensure_component_identity_sweep()

    assert not sidecar_path_for(manager.model_folder_path / "absent.safetensors").exists()


def test_sweep_skips_ckpt_pickle_with_log(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    # A non-safetensors pickle: a plausible-looking .ckpt that cannot be parsed torch-free.
    pickle_ckpt = manager.model_folder_path / "legacy.ckpt"
    pickle_ckpt.write_bytes(b"\x80\x04pickle-not-safetensors")
    good_ckpt = _write_checkpoint(manager, "good.safetensors", _UNET)
    manager.model_reference = {"legacy": _record("legacy.ckpt"), "good": _record("good.safetensors")}

    records: list[str] = []
    sink_id = logger.add(records.append, level="INFO", format="{message}")
    try:
        manager.ensure_component_identity_sweep()
    finally:
        logger.remove(sink_id)

    # The pickle is skipped (no sidecar), logged, and does not stop the good checkpoint from being processed.
    assert not sidecar_path_for(pickle_ckpt).exists()
    assert sidecar_path_for(good_ckpt).exists()
    assert any("non-safetensors" in message for message in records)


def test_sweep_is_idempotent(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    ckpt = _write_checkpoint(manager, "model.safetensors", _UNET)
    manager.model_reference = {"model": _record("model.safetensors")}

    manager.ensure_component_identity_sweep()
    sidecar_file = sidecar_path_for(ckpt)
    mtime_before = sidecar_file.stat().st_mtime_ns

    manager.ensure_component_identity_sweep()

    assert sidecar_file.stat().st_mtime_ns == mtime_before
