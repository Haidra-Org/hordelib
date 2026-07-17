"""GPU-free tests for the standalone-VAE routing decision (``hordelib.execution.standalone_vae``).

These pin the decision the loader delegates: a checkpoint with a fresh component-identity sidecar whose VAE
was extracted routes to the standalone file and a content-addressed cache key; an absent, stale, or
VAE-less sidecar, a not-yet-extracted VAE, or a missing extracted file all fall back (None); and the
kill-switch env var disables the path. The cross-model deduplication property is asserted directly: two
different checkpoints whose embedded VAEs are byte-identical produce the same cache key.

Synthetic safetensors checkpoints are built in-test (torch-free), mirroring
``horde_model_reference/tests/test_component_identity.py`` so no real model or GPU is needed.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from hordelib.execution.standalone_vae import (
    StandaloneVaePlan,
    plan_standalone_vae_load,
    standalone_vae_path_disabled,
)

_UNET = ("model.diffusion_model.x", "F16", (2,), bytes(range(40, 44)))
_VAE = [
    ("first_stage_model.decoder.conv_in.weight", "F16", (2, 2), bytes(range(1, 9))),
    ("first_stage_model.encoder.conv_out.weight", "F32", (1, 2), bytes(range(16, 24))),
]
# A second UNet residual with the SAME embedded VAE: a distinct checkpoint whose VAE weights are identical,
# so its extracted VAE deduplicates onto the same content hash (and thus the same cache key).
_OTHER_UNET = ("model.diffusion_model.y", "F32", (4,), bytes(range(70, 86)))


def _build_safetensors(tensors: list[tuple[str, str, tuple[int, ...], bytes]]) -> bytes:
    """Assemble a minimal valid safetensors container from ``(name, dtype, shape, data)`` tuples."""
    header: dict[str, object] = {}
    buffer = bytearray()
    for name, dtype, shape, data in tensors:
        begin = len(buffer)
        buffer += data
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [begin, len(buffer)]}
    header_json = json.dumps(header).encode("utf-8")
    return struct.pack("<Q", len(header_json)) + header_json + bytes(buffer)


def _write_checkpoint(path: Path, unet: tuple[str, str, tuple[int, ...], bytes]) -> Path:
    path.write_bytes(_build_safetensors([unet, *_VAE]))
    return path


def _locator_for(vae_dir: Path):
    """Return a locate callback resolving an extracted VAE file name under *vae_dir* (None when absent)."""

    def locate(file_name: str) -> Path | None:
        candidate = vae_dir / file_name
        return candidate if candidate.exists() else None

    return locate


def test_plan_routes_to_standalone_when_sidecar_fresh(tmp_path: Path) -> None:
    """A fresh sidecar with an extracted VAE yields a plan whose cache key is content-addressed."""
    from horde_model_reference.component_identity import ensure_sidecar

    ckpt = _write_checkpoint(tmp_path / "model.safetensors", _UNET)
    vae_dir = tmp_path / "vae"
    sidecar = ensure_sidecar(ckpt, extract_vae=True, extraction_dir=vae_dir)
    vae_hash = sidecar.embedded["vae"].content_hash

    plan = plan_standalone_vae_load(ckpt, _locator_for(vae_dir))

    assert isinstance(plan, StandaloneVaePlan)
    assert plan.content_hash == vae_hash
    assert plan.cache_key == f"vae@{vae_hash}"
    assert plan.vae_file_path.exists()
    assert plan.vae_file_path.parent == vae_dir


def test_plan_none_when_no_sidecar(tmp_path: Path) -> None:
    """No sidecar beside the checkpoint means fall back to the subset load."""
    ckpt = _write_checkpoint(tmp_path / "model.safetensors", _UNET)
    assert plan_standalone_vae_load(ckpt, _locator_for(tmp_path / "vae")) is None


def test_plan_none_when_sidecar_stale(tmp_path: Path) -> None:
    """A checkpoint that changed size after its sidecar was written is treated as having no sidecar."""
    from horde_model_reference.component_identity import ensure_sidecar

    ckpt = _write_checkpoint(tmp_path / "model.safetensors", _UNET)
    vae_dir = tmp_path / "vae"
    ensure_sidecar(ckpt, extract_vae=True, extraction_dir=vae_dir)

    # Grow the checkpoint so the sidecar's recorded size no longer matches.
    ckpt.write_bytes(_build_safetensors([_UNET, *_VAE, ("extra.w", "F16", (2,), bytes(range(4)))]))

    assert plan_standalone_vae_load(ckpt, _locator_for(vae_dir)) is None


def test_plan_none_when_extracted_file_missing(tmp_path: Path) -> None:
    """A sidecar that records a VAE whose extracted file is gone from disk falls back."""
    from horde_model_reference.component_identity import ensure_sidecar

    ckpt = _write_checkpoint(tmp_path / "model.safetensors", _UNET)
    vae_dir = tmp_path / "vae"
    ensure_sidecar(ckpt, extract_vae=True, extraction_dir=vae_dir)

    # The sidecar still names the VAE, but a locator that never finds it stands in for a deleted file.
    def locate_missing(_file_name: str) -> Path | None:
        return None

    assert plan_standalone_vae_load(ckpt, locate_missing) is None


def test_plan_none_when_vae_not_extracted(tmp_path: Path) -> None:
    """A sidecar computed without extraction records no extracted file name, so the path falls back."""
    from horde_model_reference.component_identity import ensure_sidecar

    ckpt = _write_checkpoint(tmp_path / "model.safetensors", _UNET)
    ensure_sidecar(ckpt)  # no extract_vae

    assert plan_standalone_vae_load(ckpt, _locator_for(tmp_path / "vae")) is None


def test_cross_model_vae_dedup_shares_cache_key(tmp_path: Path) -> None:
    """Two distinct checkpoints with byte-identical VAE weights resolve to the same cache key and file."""
    from horde_model_reference.component_identity import ensure_sidecar

    vae_dir = tmp_path / "vae"
    ckpt_a = _write_checkpoint(tmp_path / "model_a.safetensors", _UNET)
    ckpt_b = _write_checkpoint(tmp_path / "model_b.safetensors", _OTHER_UNET)
    ensure_sidecar(ckpt_a, extract_vae=True, extraction_dir=vae_dir)
    ensure_sidecar(ckpt_b, extract_vae=True, extraction_dir=vae_dir)

    plan_a = plan_standalone_vae_load(ckpt_a, _locator_for(vae_dir))
    plan_b = plan_standalone_vae_load(ckpt_b, _locator_for(vae_dir))

    assert plan_a is not None
    assert plan_b is not None
    assert plan_a.cache_key == plan_b.cache_key
    assert plan_a.vae_file_path == plan_b.vae_file_path


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", " On "])
def test_kill_switch_truthy_values_disable(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("HORDE_DISABLE_STANDALONE_VAE_PATH", value)
    assert standalone_vae_path_disabled() is True


@pytest.mark.parametrize("value", ["", "0", "false", "no", "off"])
def test_kill_switch_falsey_values_leave_enabled(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("HORDE_DISABLE_STANDALONE_VAE_PATH", value)
    assert standalone_vae_path_disabled() is False


def test_kill_switch_unset_leaves_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HORDE_DISABLE_STANDALONE_VAE_PATH", raising=False)
    assert standalone_vae_path_disabled() is False
