"""GPU-free tests for the weight prefetch ``HordeCheckpointLoader`` starts on each component it serves.

A served component may be retained across jobs with its weights still mapped from the checkpoint file, so the
loader starts a page-in of every component the request asks for, on a cache hit as well as a cold load. These
seed a ``ComponentCache`` with comfy-shaped fakes carrying tiny ``torch.nn.Module``s and record the prefetch
calls, so they check which modules are read, in what order and under which labels.

ComfyUI cannot be imported without a GPU-adjacent initialise, so ``comfy``/``folder_paths`` are stubbed for the
duration of this module, only when ComfyUI is absent.
"""

from __future__ import annotations

import importlib
import sys
import threading
import types
from collections.abc import Generator
from types import SimpleNamespace
from typing import Any

import pytest
import torch

ComponentCache: Any = None
ComponentCacheEntry: Any = None
ComponentCacheKey: Any = None
ComponentSlotKind: Any = None
node_model_loader: Any = None
HordeCheckpointLoader: Any = None


def _install_comfy_stubs() -> None:
    """Register minimal ``comfy``/``folder_paths`` modules so the loader imports without an initialised comfy."""
    comfy_module = types.ModuleType("comfy")
    for submodule_name in ("model_management", "sd", "utils"):
        submodule = types.ModuleType(f"comfy.{submodule_name}")
        setattr(comfy_module, submodule_name, submodule)
        sys.modules[f"comfy.{submodule_name}"] = submodule
    sys.modules["comfy"] = comfy_module
    sys.modules["folder_paths"] = types.ModuleType("folder_paths")


@pytest.fixture(scope="module", autouse=True)
def _load_stubbed_loader() -> Generator[None, None, None]:
    """Install test doubles only at execution time and remove every imported module afterward."""
    try:
        comfy_module = importlib.import_module("comfy")
    except ImportError:
        comfy_module = None
    if comfy_module is not None and hasattr(comfy_module, "__path__"):
        pytest.skip("The real ComfyUI package is loaded; GPU integration covers the loader serve paths.")

    global ComponentCache, ComponentCacheEntry, ComponentCacheKey, ComponentSlotKind
    global node_model_loader, HordeCheckpointLoader

    missing = object()
    stub_names = ("comfy", "comfy.model_management", "comfy.sd", "comfy.utils", "folder_paths")
    previous_stubs: dict[str, types.ModuleType | None] = {name: sys.modules.get(name) for name in stub_names}
    previous_node_modules = {name for name in sys.modules if name.startswith("hordelib.nodes")}
    nodes_package = sys.modules.get("hordelib.nodes")
    previous_loader_attribute = getattr(nodes_package, "node_model_loader", missing)
    try:
        _install_comfy_stubs()
        component_cache = importlib.import_module("hordelib.execution.component_cache")
        node_model_loader = importlib.import_module("hordelib.nodes.node_model_loader")
        ComponentCache = component_cache.ComponentCache
        ComponentCacheEntry = component_cache.ComponentCacheEntry
        ComponentCacheKey = component_cache.ComponentCacheKey
        ComponentSlotKind = component_cache.ComponentSlotKind
        HordeCheckpointLoader = node_model_loader.HordeCheckpointLoader
        yield
    finally:
        for name in [name for name in sys.modules if name.startswith("hordelib.nodes")]:
            if name not in previous_node_modules:
                sys.modules.pop(name, None)
        if nodes_package is not None:
            package_namespace = vars(nodes_package)
            if previous_loader_attribute is missing:
                package_namespace.pop("node_model_loader", None)
            else:
                package_namespace["node_model_loader"] = previous_loader_attribute
        for name, previous in previous_stubs.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


class _FakePatcher:
    """Stands in for a comfy ``ModelPatcher``: the wrapped module is ``model``."""

    def __init__(self, module: torch.nn.Module) -> None:
        self.model = module


class _FakeClip:
    """Stands in for ``comfy.sd.CLIP``: ``cond_stage_model`` and ``patcher.model`` are the same module."""

    def __init__(self, module: torch.nn.Module) -> None:
        self.cond_stage_model = module
        self.patcher = _FakePatcher(module)


class _FakeVae:
    """Stands in for ``comfy.sd.VAE``; ``module=None`` is a VAE built without recognised weights (no patcher)."""

    def __init__(self, module: torch.nn.Module | None) -> None:
        self.first_stage_model = module
        if module is not None:
            self.patcher = _FakePatcher(module)


class _PrefetchRecorder:
    """Records each ``prefetch_module_weights_async`` call as ``(module, label)`` and its stop flag apart."""

    def __init__(self) -> None:
        self.calls: list[tuple[torch.nn.Module, str]] = []
        self.stops: list[threading.Event | None] = []

    def __call__(
        self,
        module: torch.nn.Module,
        *,
        label: str = "",
        stop: threading.Event | None = None,
    ) -> None:
        self.calls.append((module, label))
        self.stops.append(stop)


@pytest.fixture
def serve_env(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """A budgeted cache, a recorder patched over the loader's prefetch, and three distinct component modules."""
    cache = ComponentCache(budget_mb=8192)
    recorder = _PrefetchRecorder()
    monkeypatch.setattr(node_model_loader, "prefetch_module_weights_async", recorder)
    monkeypatch.setattr(node_model_loader, "log_free_ram", lambda: None)
    return SimpleNamespace(
        cache=cache,
        recorder=recorder,
        unet_module=torch.nn.Linear(2, 2),
        text_encoder_module=torch.nn.Linear(2, 2),
        vae_module=torch.nn.Linear(2, 2),
    )


def _seed_checkpoint(cache: Any, horde_model_name: str, payload: tuple[Any, ...]) -> None:
    cache.put(
        ComponentCacheEntry(
            key=ComponentCacheKey(ComponentSlotKind.CHECKPOINT, horde_model_name),
            payload=payload,
            approx_ram_mb=1.0,
            source_ckpt_path="model.safetensors",
        ),
    )


def _serve_monolithic(cache: Any, *, output_model: bool, output_clip: bool, output_vae: bool) -> Any:
    return HordeCheckpointLoader()._load_monolithic_checkpoint(
        cache,
        "model_a",
        None,
        output_model=output_model,
        output_vae=output_vae,
        output_clip=output_clip,
        seamless_tiling_enabled=False,
        will_mutate=False,
    )


def test_monolithic_hit_prefetches_text_encoder_then_unet_then_vae(serve_env: SimpleNamespace) -> None:
    payload = (
        _FakePatcher(serve_env.unet_module),
        _FakeClip(serve_env.text_encoder_module),
        _FakeVae(serve_env.vae_module),
        None,
    )
    _seed_checkpoint(serve_env.cache, "model_a", payload)

    served = _serve_monolithic(serve_env.cache, output_model=True, output_clip=True, output_vae=True)

    assert served is payload
    assert serve_env.recorder.calls == [
        (serve_env.text_encoder_module, "model_a:text_encoder"),
        (serve_env.unet_module, "model_a:unet"),
        (serve_env.vae_module, "model_a:vae"),
    ]


def test_monolithic_vae_only_hit_prefetches_only_the_vae(serve_env: SimpleNamespace) -> None:
    """A decode served from a cached full tuple must not page in the UNet or the text encoder."""
    payload = (
        _FakePatcher(serve_env.unet_module),
        _FakeClip(serve_env.text_encoder_module),
        _FakeVae(serve_env.vae_module),
        None,
    )
    _seed_checkpoint(serve_env.cache, "model_a", payload)

    served = _serve_monolithic(serve_env.cache, output_model=False, output_clip=False, output_vae=True)

    assert served is payload
    assert serve_env.recorder.calls == [(serve_env.vae_module, "model_a:vae")]


def test_bare_component_hit_prefetches_its_component(serve_env: SimpleNamespace) -> None:
    payload = (_FakePatcher(serve_env.unet_module), None, None)
    serve_env.cache.put(
        ComponentCacheEntry(
            key=ComponentCacheKey(ComponentSlotKind.UNET, "model_a:unet"),
            payload=payload,
            approx_ram_mb=1.0,
            source_ckpt_path="model_unet.safetensors",
        ),
    )

    served = HordeCheckpointLoader()._load_bare_component(
        serve_env.cache,
        "model_a",
        None,
        "unet",
        None,
        seamless_tiling_enabled=False,
        will_mutate=False,
    )

    assert served is payload
    assert serve_env.recorder.calls == [(serve_env.unet_module, "model_a:unet")]


def test_hit_prefetch_carries_the_entry_stop_flag(serve_env: SimpleNamespace) -> None:
    """A prefetch started for a cache hit is the one the entry's eviction stops."""
    payload = (_FakePatcher(serve_env.unet_module), None, None)
    entry = ComponentCacheEntry(
        key=ComponentCacheKey(ComponentSlotKind.UNET, "model_a:unet"),
        payload=payload,
        approx_ram_mb=1.0,
        source_ckpt_path="model_unet.safetensors",
    )
    serve_env.cache.put(entry)

    HordeCheckpointLoader()._load_bare_component(
        serve_env.cache,
        "model_a",
        None,
        "unet",
        None,
        seamless_tiling_enabled=False,
        will_mutate=False,
    )

    assert serve_env.recorder.stops == [entry.prefetch_stop]


def test_component_without_a_module_is_skipped_and_the_payload_served(serve_env: SimpleNamespace) -> None:
    """A VAE built without recognised weights has no ``first_stage_model``; the other components still read."""
    payload = (
        _FakePatcher(serve_env.unet_module),
        _FakeClip(serve_env.text_encoder_module),
        _FakeVae(None),
        None,
    )
    _seed_checkpoint(serve_env.cache, "model_a", payload)

    served = _serve_monolithic(serve_env.cache, output_model=True, output_clip=True, output_vae=True)

    assert served is payload
    assert serve_env.recorder.calls == [
        (serve_env.text_encoder_module, "model_a:text_encoder"),
        (serve_env.unet_module, "model_a:unet"),
    ]


def test_none_slots_are_skipped_without_raising(serve_env: SimpleNamespace) -> None:
    node_model_loader._prefetch_served_components(
        (None, None, None, None),
        "model_a",
        output_model=True,
        output_clip=True,
        output_vae=True,
    )

    assert serve_env.recorder.calls == []


def test_prefetch_failure_does_not_fail_the_serve(
    serve_env: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failing_prefetch(
        module: torch.nn.Module,
        *,
        label: str = "",
        stop: threading.Event | None = None,
    ) -> None:
        raise RuntimeError("prefetch unavailable")

    monkeypatch.setattr(node_model_loader, "prefetch_module_weights_async", failing_prefetch)
    payload = (_FakePatcher(serve_env.unet_module), None, None)
    serve_env.cache.put(
        ComponentCacheEntry(
            key=ComponentCacheKey(ComponentSlotKind.UNET, "model_a:unet"),
            payload=payload,
            approx_ram_mb=1.0,
            source_ckpt_path="model_unet.safetensors",
        ),
    )

    served = HordeCheckpointLoader()._load_bare_component(
        serve_env.cache,
        "model_a",
        None,
        "unet",
        None,
        seamless_tiling_enabled=False,
        will_mutate=False,
    )

    assert served is payload
