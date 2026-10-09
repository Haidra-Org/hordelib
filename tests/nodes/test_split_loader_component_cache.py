"""GPU-free tests for serving ComfyUI's split-files ``CLIPLoader`` and ``VAELoader`` through the component cache.

The stock loaders build a new object, and so a new ModelPatcher, on every call. ComfyUI tracks loaded models by
patcher identity, so on a process whose diffusion model is retained across jobs each job's fresh text encoder
and VAE were uploaded beside the previous copies. These check that a repeated load of the same file returns the
resident object (one copy, read from disk once), that the hijacks pass ComfyUI's node inputs through, and that
under the single-slot budget the split components stay resident beside the diffusion model. The preload rows
check that a preloaded model is cached under the key its run's main loader asks for, so the run reads the file
once.

ComfyUI cannot be imported without a GPU-adjacent initialise, so ``comfy``/``folder_paths`` are stubbed for the
duration of this module, only when ComfyUI is absent.
"""

from __future__ import annotations

import importlib
import sys
import types
from collections.abc import Callable, Generator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

ComponentCache: Any = None
ComponentCacheEntry: Any = None
ComponentCacheKey: Any = None
ComponentSlotKind: Any = None
node_model_loader: Any = None


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
        pytest.skip("The real ComfyUI package is loaded; GPU integration covers the split loaders.")

    global ComponentCache, ComponentCacheEntry, ComponentCacheKey, ComponentSlotKind, node_model_loader

    missing = object()
    stub_names = ("comfy", "comfy.model_management", "comfy.sd", "comfy.utils", "folder_paths")
    previous_stubs = {name: sys.modules.get(name, missing) for name in stub_names}
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
            if previous is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


_RESOLVABLE_FILES = {
    ("text_encoders", "qwen_3_06b_base.safetensors"): "/models/text_encoders/qwen_3_06b_base.safetensors",
    ("vae", "qwen_image_vae.safetensors"): "/models/vae/qwen_image_vae.safetensors",
}


class _CountingLoader:
    """Stands in for a stock ComfyUI loader. Each call builds a new component object, as the real one does."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, loader: object, *args: Any, **kwargs: Any) -> tuple[object]:
        self.calls.append({"args": args, "kwargs": kwargs})
        return (object(),)


def _cache_env(monkeypatch: pytest.MonkeyPatch, *, budget_mb: float) -> SimpleNamespace:
    cache = ComponentCache(budget_mb=budget_mb)
    monkeypatch.setattr(node_model_loader, "process_component_cache", lambda: cache)
    monkeypatch.setattr(node_model_loader, "trim_host_after_component_release", lambda: None)
    monkeypatch.setattr(node_model_loader, "_release_device_cache_after_eviction", lambda: None)
    monkeypatch.setattr(
        node_model_loader.folder_paths,
        "get_full_path",
        lambda folder, name: _RESOLVABLE_FILES.get((folder, name)),
        raising=False,
    )
    monkeypatch.setattr(node_model_loader, "_safe_file_size", lambda path: 1024 * 1024)
    return SimpleNamespace(cache=cache)


@pytest.fixture
def budgeted(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    return _cache_env(monkeypatch, budget_mb=8192)


@pytest.fixture
def single_slot(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    return _cache_env(monkeypatch, budget_mb=0)


def _serve_text_encoder(stock: _CountingLoader, *, clip_type_name: str = "qwen_image") -> tuple[Any, ...]:
    clip_name = "qwen_3_06b_base.safetensors"
    return node_model_loader.serve_split_text_encoder(
        lambda: stock(None, clip_name, type=clip_type_name, device="default"),
        clip_name=clip_name,
        clip_type_name=clip_type_name,
        device="default",
    )


def _serve_vae(stock: _CountingLoader, vae_name: str = "qwen_image_vae.safetensors") -> tuple[Any, ...]:
    return node_model_loader.serve_split_vae(lambda: stock(None, vae_name), vae_name=vae_name)


@pytest.mark.parametrize("serve", [_serve_text_encoder, _serve_vae], ids=["text_encoder", "vae"])
def test_repeated_load_of_one_file_serves_one_copy(
    budgeted: SimpleNamespace,
    serve: Callable[[_CountingLoader], tuple[Any, ...]],
) -> None:
    """Two consecutive loads of the same file read it once and hand back the same component object."""
    stock = _CountingLoader()

    first = serve(stock)
    second = serve(stock)

    assert len(stock.calls) == 1
    assert second is first
    assert second[0] is first[0]
    assert len(budgeted.cache) == 1


def test_text_encoder_identity_includes_its_clip_type(budgeted: SimpleNamespace) -> None:
    """The same file wrapped as a different CLIP type is a different module, so it never aliases."""
    stock = _CountingLoader()

    qwen_image = _serve_text_encoder(stock, clip_type_name="qwen_image")
    lumina2 = _serve_text_encoder(stock, clip_type_name="lumina2")

    assert len(stock.calls) == 2
    assert qwen_image[0] is not lumina2[0]


def test_unresolved_name_goes_to_the_stock_loader_every_time(budgeted: SimpleNamespace) -> None:
    """A name with no file behind it (``pixel_space``, a tiny autoencoder) is never cached."""
    stock = _CountingLoader()

    _serve_vae(stock, "pixel_space")
    _serve_vae(stock, "pixel_space")

    assert len(stock.calls) == 2
    assert len(budgeted.cache) == 0


def test_single_slot_keeps_split_components_beside_the_retained_diffusion_model(
    single_slot: SimpleNamespace,
) -> None:
    """Under the zero budget a job's text encoder and VAE loads leave the cached diffusion model resident.

    The second job's loads are then all hits. A single slot shared by every kind would evict the diffusion
    model at a job's first cold load and make each component displace the others within every job.
    """
    unet_key = ComponentCacheKey(ComponentSlotKind.UNET, "Anima-Turbo-v1.1:unet")
    single_slot.cache.put(
        ComponentCacheEntry(key=unet_key, payload=(object(), None, None), approx_ram_mb=4000.0, source_ckpt_path="u"),
    )
    text_encoder_stock = _CountingLoader()
    vae_stock = _CountingLoader()

    first_vae = _serve_vae(vae_stock)
    first_text_encoder = _serve_text_encoder(text_encoder_stock)
    second_vae = _serve_vae(vae_stock)
    second_text_encoder = _serve_text_encoder(text_encoder_stock)

    assert single_slot.cache.get(unet_key) is not None
    assert len(single_slot.cache) == 3
    assert (len(vae_stock.calls), len(text_encoder_stock.calls)) == (1, 1)
    assert second_vae is first_vae
    assert second_text_encoder is first_text_encoder


def _seed_retained_unet(cache: Any) -> Any:
    unet_key = ComponentCacheKey(ComponentSlotKind.UNET, "Anima-Turbo-v1.1:unet")
    cache.put(
        ComponentCacheEntry(key=unet_key, payload=(object(), None, None), approx_ram_mb=4000.0, source_ckpt_path="u"),
    )
    return unet_key


def test_job_end_without_retention_releases_split_components(single_slot: SimpleNamespace) -> None:
    """A job the host does not retain past drops its split text encoder and VAE and keeps the diffusion model."""
    unet_key = _seed_retained_unet(single_slot.cache)
    _serve_vae(_CountingLoader())
    _serve_text_encoder(_CountingLoader())

    released = single_slot.cache.release_unretained()

    assert sorted(entry.key.kind for entry in released) == sorted([ComponentSlotKind.CLIP, ComponentSlotKind.VAE])
    assert single_slot.cache.get(unet_key) is not None
    assert len(single_slot.cache) == 1


def test_retained_jobs_keep_one_split_component_per_kind(single_slot: SimpleNamespace) -> None:
    """Across retained jobs, nothing releases the split components, so each kind stays one resident copy."""
    _seed_retained_unet(single_slot.cache)
    vae_stock = _CountingLoader()
    text_encoder_stock = _CountingLoader()

    for _job in range(3):
        _serve_vae(vae_stock)
        _serve_text_encoder(text_encoder_stock)

    held_kinds = sorted(snapshot.kind for snapshot in single_slot.cache.held_report())
    assert held_kinds == sorted([ComponentSlotKind.UNET, ComponentSlotKind.CLIP, ComponentSlotKind.VAE])
    assert (len(vae_stock.calls), len(text_encoder_stock.calls)) == (1, 1)


def test_comfy_hijacks_pass_node_inputs_through_and_reuse_the_copy(
    budgeted: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ``CLIPLoader``/``VAELoader`` hijacks call the captured stock method with ComfyUI's input names."""
    comfy_patches = importlib.import_module("hordelib.execution.comfy_patches")
    clip_stock = _CountingLoader()
    vae_stock = _CountingLoader()
    monkeypatch.setitem(comfy_patches._originals, "clip_loader_load_clip", clip_stock)
    monkeypatch.setitem(comfy_patches._originals, "vae_loader_load_vae", vae_stock)
    loader_node = object()

    first_clip = comfy_patches._clip_loader_load_clip_hijack(
        loader_node,
        clip_name="qwen_3_06b_base.safetensors",
        type="qwen_image",
        device="default",
    )
    second_clip = comfy_patches._clip_loader_load_clip_hijack(
        loader_node,
        clip_name="qwen_3_06b_base.safetensors",
        type="qwen_image",
        device="default",
    )
    first_vae = comfy_patches._vae_loader_load_vae_hijack(loader_node, vae_name="qwen_image_vae.safetensors")
    second_vae = comfy_patches._vae_loader_load_vae_hijack(loader_node, vae_name="qwen_image_vae.safetensors")

    assert clip_stock.calls == [
        {"args": ("qwen_3_06b_base.safetensors",), "kwargs": {"type": "qwen_image", "device": "default"}},
    ]
    assert vae_stock.calls == [{"args": ("qwen_image_vae.safetensors",), "kwargs": {}}]
    assert second_clip is first_clip
    assert second_vae is first_vae


class _FakeDiffusionPatcher:
    """Stands in for a comfy ``ModelPatcher`` around a freshly loaded diffusion model."""

    def __init__(self) -> None:
        self.model = torch.nn.Linear(2, 2)


class _FakeCompvis:
    """The record lookups the loader makes on a cold load: one model whose files are declared by type."""

    def __init__(self, file_entries: list[dict[str, Any]]) -> None:
        self._file_entries = file_entries

    def is_model_available(self, horde_model_name: str) -> bool:
        return True

    def get_model_filenames(self, horde_model_name: str) -> list[dict[str, Any]]:
        return self._file_entries


class _RecordingGraph:
    """Collects the inputs a patch step sets, keyed ``<node>.<input>`` as the graph receives them."""

    def __init__(self) -> None:
        self.inputs: dict[str, Any] = {}

    def set_inputs(self, updates: dict[str, Any]) -> None:
        self.inputs.update(updates)


_KREA2_MODEL = "Krea2-Turbo_fp8"
_KREA2_UNET_FILE = "krea2_turbo_fp8_scaled.safetensors"
_SDXL_MODEL = "AlbedoBase XL (SDXL)"
_SDXL_FILE = "albedobase_xl.safetensors"


def _model_context(horde_model_name: str, baseline: Any, main_file: str) -> Any:
    context_module = importlib.import_module("hordelib.pipeline.context")
    return context_module.ModelContext(horde_model_name=horde_model_name, baseline=baseline, main_file=main_file)


def _preload_env(
    monkeypatch: pytest.MonkeyPatch,
    *,
    horde_model_name: str,
    baseline: Any,
    main_file: str,
    file_type: str | None,
) -> SimpleNamespace:
    """A single-slot cache behind a stubbed record and disk, counting each comfy load call."""
    env = _cache_env(monkeypatch, budget_mb=0)
    context = _model_context(horde_model_name, baseline, main_file)
    compvis = _FakeCompvis([{"file_path": Path(main_file), "file_type": file_type}])
    monkeypatch.setattr(
        node_model_loader,
        "SharedModelManager",
        SimpleNamespace(manager=SimpleNamespace(compvis=compvis, _models_in_ram=env.cache)),
    )
    resolution = importlib.import_module("hordelib.pipeline.resolution")
    monkeypatch.setattr(resolution, "resolve_image_model", lambda name: context)
    monkeypatch.setattr(
        node_model_loader.folder_paths,
        "get_full_path",
        lambda folder, name: f"/models/{folder}/{name}",
        raising=False,
    )
    monkeypatch.setattr(node_model_loader.folder_paths, "get_folder_paths", lambda folder: [], raising=False)
    monkeypatch.setattr(node_model_loader, "prefetch_module_weights_async", lambda *args, **kwargs: None)
    monkeypatch.setattr(node_model_loader, "log_free_ram", lambda: None)
    monkeypatch.setattr(node_model_loader, "_estimate_checkpoint_ram_mb", lambda path, payload: 1.0)
    disk_loads: list[str] = []

    def load_diffusion_model(ckpt_path: str, model_options: dict[str, Any]) -> _FakeDiffusionPatcher:
        disk_loads.append(ckpt_path)
        return _FakeDiffusionPatcher()

    def load_checkpoint_guess_config(ckpt_path: str, **kwargs: Any) -> tuple[Any, ...]:
        disk_loads.append(ckpt_path)
        text_encoder = SimpleNamespace() if kwargs["output_clip"] else None
        vae = SimpleNamespace() if kwargs["output_vae"] else None
        return (_FakeDiffusionPatcher(), text_encoder, vae, None)

    monkeypatch.setattr(node_model_loader.comfy.sd, "load_diffusion_model", load_diffusion_model, raising=False)
    monkeypatch.setattr(
        node_model_loader.comfy.sd,
        "load_checkpoint_guess_config",
        load_checkpoint_guess_config,
        raising=False,
    )
    env.context = context
    env.disk_loads = disk_loads
    return env


def _run_main_model_load(context: Any) -> tuple[Any, ...]:
    """Load the main model the way a run's graph does, from the inputs ``apply_main_model`` sets."""
    steps = importlib.import_module("hordelib.pipeline.families.image_gen.steps")
    graph = _RecordingGraph()
    unused_payload: Any = None
    steps.apply_main_model(graph, unused_payload, context)
    return node_model_loader.HordeCheckpointLoader().load_checkpoint(
        will_load_loras=graph.inputs["model_loader.will_load_loras"],
        seamless_tiling_enabled=False,
        horde_model_name=graph.inputs["model_loader.horde_model_name"],
        ckpt_name=graph.inputs["model_loader.ckpt_name"],
        file_type=graph.inputs["model_loader.file_type"],
    )


def _preload(horde_model_name: str, *, diffusion_model_only: bool) -> tuple[Any, ...]:
    return node_model_loader.HordeCheckpointLoader().preload(
        horde_model_name,
        will_load_loras=False,
        seamless_tiling_enabled=False,
        diffusion_model_only=diffusion_model_only,
    )


def _held_keys(cache: Any) -> list[tuple[Any, str]]:
    return [(snapshot.kind, snapshot.identity) for snapshot in cache.held_report()]


def test_split_files_preload_then_run_maps_the_file_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """A run after a split-files preload is served the preloaded diffusion model, with no second disk load.

    Under the checkpoint key the run missed and mapped the file again beside the preload's copy, which a host
    near its commit limit cannot hold.
    """
    meta_consts = importlib.import_module("horde_model_reference.meta_consts")
    env = _preload_env(
        monkeypatch,
        horde_model_name=_KREA2_MODEL,
        baseline=meta_consts.KNOWN_IMAGE_GENERATION_BASELINE.krea2_turbo,
        main_file=_KREA2_UNET_FILE,
        file_type="unet",
    )

    preloaded = _preload(_KREA2_MODEL, diffusion_model_only=True)
    preloaded_keys = _held_keys(env.cache)
    served = _run_main_model_load(env.context)

    assert preloaded_keys == [(ComponentSlotKind.UNET, f"{_KREA2_MODEL}:unet")]
    assert _held_keys(env.cache) == preloaded_keys
    assert env.disk_loads == [f"/models/checkpoints/{_KREA2_UNET_FILE}"]
    assert served is preloaded


def test_checkpoint_preload_then_run_maps_the_file_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """A whole checkpoint keeps the checkpoint key, so its run is served the preloaded tuple."""
    meta_consts = importlib.import_module("horde_model_reference.meta_consts")
    env = _preload_env(
        monkeypatch,
        horde_model_name=_SDXL_MODEL,
        baseline=meta_consts.KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_xl,
        main_file=_SDXL_FILE,
        file_type=None,
    )

    preloaded = _preload(_SDXL_MODEL, diffusion_model_only=False)
    preloaded_keys = _held_keys(env.cache)
    served = _run_main_model_load(env.context)

    assert preloaded_keys == [(ComponentSlotKind.CHECKPOINT, _SDXL_MODEL)]
    assert _held_keys(env.cache) == preloaded_keys
    assert env.disk_loads == [f"/models/checkpoints/{_SDXL_FILE}"]
    assert served is preloaded


@pytest.mark.parametrize(
    ("baseline_name", "expected_file_type"),
    [("krea2_turbo", "unet"), ("qwen_image", "unet"), ("stable_diffusion_xl", None), ("flux_1", None)],
)
def test_preload_and_run_read_one_file_type(baseline_name: str, expected_file_type: str | None) -> None:
    """The run's graph gives the main loader the file type the preload derives, for every baseline shape."""
    meta_consts = importlib.import_module("horde_model_reference.meta_consts")
    baselines = importlib.import_module("hordelib.pipeline.families.image_gen.baselines")
    steps = importlib.import_module("hordelib.pipeline.families.image_gen.steps")
    baseline = meta_consts.KNOWN_IMAGE_GENERATION_BASELINE(baseline_name)
    graph = _RecordingGraph()
    unused_payload: Any = None

    steps.apply_main_model(graph, unused_payload, _model_context("model_a", baseline, "model_a.safetensors"))

    assert baselines.main_loader_file_type(baseline) == expected_file_type
    assert graph.inputs["model_loader.file_type"] == expected_file_type
