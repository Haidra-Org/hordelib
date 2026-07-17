"""GPU-free routing tests for the standalone-VAE path in ``HordeCheckpointLoader.load_checkpoint``.

These verify the loader wiring the pure ``standalone_vae`` decision test cannot reach: the disk load of the
extracted VAE, insertion into the in-RAM cache under the content-addressed ``vae@<hash>`` key, tiling
normalization, a same-process cache hit (including across two different model names that share a VAE), and
the gating that only attempts the path for a VAE-only, monolithic (``file_type is None``), non-kill-switched
request.

ComfyUI cannot be imported without a GPU-adjacent initialise, so where it is genuinely available these run as
the real integration test instead; here they stub ``comfy``/``folder_paths`` so the loader imports and the
routing runs without a GPU. The stubbing is confined to this module and only happens when ComfyUI is absent.
"""

from __future__ import annotations

import json
import struct
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

try:  # A real ComfyUI means the GPU integration test covers reality; skip the stubbed unit routing here.
    import comfy  # type: ignore  # noqa: F401

    _COMFY_AVAILABLE = True
except ImportError:
    _COMFY_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    _COMFY_AVAILABLE,
    reason="Stubbed-comfy routing runs only where ComfyUI is absent; the GPU integration test covers reality.",
)


def _install_comfy_stubs() -> None:
    """Register minimal ``comfy``/``folder_paths`` modules so the loader imports without an initialised comfy."""
    comfy_module = types.ModuleType("comfy")
    for submodule_name in ("model_management", "sd", "utils"):
        submodule = types.ModuleType(f"comfy.{submodule_name}")
        setattr(comfy_module, submodule_name, submodule)
        sys.modules[f"comfy.{submodule_name}"] = submodule
    sys.modules["comfy"] = comfy_module
    sys.modules["folder_paths"] = types.ModuleType("folder_paths")


if not _COMFY_AVAILABLE:
    _install_comfy_stubs()
    from hordelib.execution.component_cache import (
        ComponentCache,
        ComponentCacheEntry,
        ComponentCacheKey,
        ComponentSlotKind,
    )
    from hordelib.nodes import node_model_loader
    from hordelib.nodes.node_model_loader import HordeCheckpointLoader


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


class _FakeVae:
    """Stands in for a comfy VAE; ``first_stage_model`` is None so the tiling helpers are a no-op."""

    def __init__(self, state_dict: object) -> None:
        self.state_dict = state_dict
        self.first_stage_model = None


@pytest.fixture
def routing_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Wire a checkpoint + extracted VAE, a fake manager cache, and stubbed comfy loaders into the loader."""
    from horde_model_reference.component_identity import ensure_sidecar

    compvis_dir = tmp_path / "compvis"
    vae_dir = tmp_path / "vae"
    compvis_dir.mkdir()
    ckpt = compvis_dir / "model.safetensors"
    ckpt.write_bytes(_build_safetensors([_UNET, *_VAE]))
    sidecar = ensure_sidecar(ckpt, extract_vae=True, extraction_dir=vae_dir)
    vae_hash = sidecar.embedded["vae"].content_hash
    extracted_name = sidecar.embedded["vae"].extracted_file_name
    assert extracted_name is not None

    cache = ComponentCache(budget_mb=8192)

    def is_model_available(name: str) -> bool:
        return name in {"model_a", "model_b"}

    def get_model_filenames(name: str) -> list[dict]:
        return [{"file_path": Path("model.safetensors")}]

    fake_compvis = SimpleNamespace(
        is_model_available=is_model_available,
        get_model_filenames=get_model_filenames,
    )
    fake_manager = SimpleNamespace(compvis=fake_compvis, _models_in_ram=cache)

    monkeypatch.setattr(node_model_loader.SharedModelManager, "manager", fake_manager, raising=False)
    monkeypatch.setattr(node_model_loader, "log_free_ram", lambda: None)

    def get_full_path(category: str, file_name: str) -> str | None:
        if category == "vae" and file_name == extracted_name:
            return str(vae_dir / file_name)
        if category == "checkpoints" and file_name == "model.safetensors":
            return str(ckpt)
        return None

    monkeypatch.setattr(node_model_loader.folder_paths, "get_full_path", get_full_path, raising=False)

    load_torch_file = MagicMock(side_effect=lambda path: {"path": path})
    monkeypatch.setattr(node_model_loader.comfy.utils, "load_torch_file", load_torch_file, raising=False)
    vae_factory = MagicMock(side_effect=lambda sd: _FakeVae(sd))
    monkeypatch.setattr(node_model_loader.comfy.sd, "VAE", vae_factory, raising=False)

    return SimpleNamespace(
        vae_hash=vae_hash,
        cache_key=f"vae@{vae_hash}",
        vae_component_key=ComponentCacheKey(ComponentSlotKind.VAE, f"vae@{vae_hash}"),
        extracted_path=vae_dir / extracted_name,
        cache=cache,
        load_torch_file=load_torch_file,
        vae_factory=vae_factory,
    )


def test_standalone_vae_disk_load_caches_by_content_hash(routing_env) -> None:
    loader = HordeCheckpointLoader()

    result = loader._load_standalone_vae(routing_env.cache, "model_a", None, seamless_tiling_enabled=False)

    assert result is not None
    assert result[0] is None and result[1] is None and result[3] is None
    assert isinstance(result[2], _FakeVae)
    # The standalone extracted file was loaded, not the monolithic checkpoint.
    routing_env.load_torch_file.assert_called_once_with(str(routing_env.extracted_path))
    # Cached under the content-addressed VAE key, and nothing was cached under the model name.
    identities = [snapshot.identity for snapshot in routing_env.cache.held_report()]
    assert identities == [routing_env.cache_key]
    assert "model_a" not in identities


def test_standalone_vae_entry_is_stored_reusable(routing_env) -> None:
    """The cached standalone VAE is always reusable: LoRA patches attach to the UNet and text encoders,
    never the VAE, so a LoRA-bearing job's decode must share the entry rather than reload it."""
    loader = HordeCheckpointLoader()

    loader._load_standalone_vae(routing_env.cache, "model_a", None, seamless_tiling_enabled=False)

    # A reusable entry is servable on a later lookup; a non-reusable one would return None here.
    served = routing_env.cache.get(routing_env.vae_component_key)
    assert served is not None
    assert served.reusable is True


def test_standalone_vae_second_serve_is_cache_hit(routing_env) -> None:
    loader = HordeCheckpointLoader()

    first = loader._load_standalone_vae(routing_env.cache, "model_a", None, seamless_tiling_enabled=False)
    second = loader._load_standalone_vae(routing_env.cache, "model_a", None, seamless_tiling_enabled=False)

    assert first is second
    routing_env.load_torch_file.assert_called_once()  # not reloaded from disk on the hit


def test_cross_model_same_vae_hits_shared_cache(routing_env) -> None:
    loader = HordeCheckpointLoader()

    # model_a and model_b resolve to the same checkpoint (same VAE), so the second serve under a DIFFERENT
    # model name must be a content-hash cache hit, not a second disk load.
    loader._load_standalone_vae(routing_env.cache, "model_a", None, seamless_tiling_enabled=False)
    result_b = loader._load_standalone_vae(routing_env.cache, "model_b", None, seamless_tiling_enabled=False)

    assert result_b is not None
    routing_env.load_torch_file.assert_called_once()
    identities = [snapshot.identity for snapshot in routing_env.cache.held_report()]
    assert identities == [routing_env.cache_key]


def test_standalone_returns_none_without_sidecar(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    compvis_dir = tmp_path / "compvis"
    compvis_dir.mkdir()
    ckpt = compvis_dir / "model.safetensors"
    ckpt.write_bytes(_build_safetensors([_UNET, *_VAE]))  # no sidecar written

    fake_compvis = SimpleNamespace(
        is_model_available=lambda name: True,
        get_model_filenames=lambda name: [{"file_path": Path("model.safetensors")}],
    )
    monkeypatch.setattr(
        node_model_loader.SharedModelManager, "manager", SimpleNamespace(compvis=fake_compvis), raising=False
    )
    monkeypatch.setattr(node_model_loader, "log_free_ram", lambda: None)
    monkeypatch.setattr(
        node_model_loader.folder_paths,
        "get_full_path",
        lambda category, file_name: str(ckpt) if category == "checkpoints" else None,
        raising=False,
    )

    loader = HordeCheckpointLoader()
    result = loader._load_standalone_vae(ComponentCache(budget_mb=0), "model_a", None, seamless_tiling_enabled=False)
    assert result is None


class _GatingManager:
    """A manager whose cache is pre-seeded so load_checkpoint's fall-through returns without touching comfy.

    Both the full-checkpoint key (``CHECKPOINT``/``seed_model``) and the bare-component key
    (``VAE``/``seed_model:vae``) are seeded so that whichever non-standalone branch a not-eligible request
    takes is served entirely from cache, never reaching a comfy disk load.
    """

    def __init__(self) -> None:
        model = SimpleNamespace(model=MagicMock())
        vae = SimpleNamespace(first_stage_model=None)
        self.full_result = (model, MagicMock(), vae, None)
        self.component_result = (SimpleNamespace(model=MagicMock()), None, None)
        self._models_in_ram = ComponentCache(budget_mb=8192)
        self._models_in_ram.put(
            ComponentCacheEntry(
                key=ComponentCacheKey(ComponentSlotKind.CHECKPOINT, "seed_model"),
                payload=self.full_result,
                approx_ram_mb=1.0,
                reusable=True,
                source_ckpt_path="seed_model",
            ),
        )
        self._models_in_ram.put(
            ComponentCacheEntry(
                key=ComponentCacheKey(ComponentSlotKind.VAE, "seed_model:vae"),
                payload=self.component_result,
                approx_ram_mb=1.0,
                reusable=True,
                source_ckpt_path="seed_model:vae",
            ),
        )
        self.compvis = SimpleNamespace(is_model_available=lambda name: True)


@pytest.mark.parametrize(
    ("output_model", "output_vae", "output_clip", "file_type", "disabled", "should_attempt"),
    [
        (False, True, False, None, False, True),  # VAE-only, monolithic, enabled -> attempt
        (False, True, False, None, True, False),  # kill-switch disables
        (True, True, True, None, False, False),  # full load is not VAE-only
        (False, True, False, "vae", False, False),  # component load (file_type set) is not the monolithic path
    ],
)
def test_gating_attempts_standalone_only_when_eligible(
    monkeypatch: pytest.MonkeyPatch,
    output_model: bool,
    output_vae: bool,
    output_clip: bool,
    file_type: str | None,
    disabled: bool,
    should_attempt: bool,
) -> None:
    manager = _GatingManager()
    monkeypatch.setattr(node_model_loader.SharedModelManager, "manager", manager, raising=False)
    monkeypatch.setattr(node_model_loader, "log_free_ram", lambda: None)
    monkeypatch.setattr(node_model_loader, "standalone_vae_path_disabled", lambda: disabled)

    sentinel = (None, None, "standalone-vae", None)
    attempted = {"called": False}

    def fake_standalone(self, *args, **kwargs):
        attempted["called"] = True
        return sentinel

    monkeypatch.setattr(HordeCheckpointLoader, "_load_standalone_vae", fake_standalone, raising=False)

    loader = HordeCheckpointLoader()
    result = loader.load_checkpoint(
        will_load_loras=False,
        seamless_tiling_enabled=False,
        horde_model_name="seed_model",
        file_type=file_type,
        output_model=output_model,
        output_vae=output_vae,
        output_clip=output_clip,
    )

    assert attempted["called"] is should_attempt
    if should_attempt:
        assert result is sentinel
    else:
        # Not eligible: the standalone method was never consulted and the request was served from the
        # pre-seeded cache (never reaching a comfy disk load).
        assert result in (manager.full_result, manager.component_result)
