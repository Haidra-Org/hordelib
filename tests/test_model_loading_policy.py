"""GPU-free unit tests for the force-load skip policy and multi-file component placement.

These guard the two robustness fixes that let large multi-file models (e.g. Qwen-Image) load:
  * the force-load skip list is keyed on comfy *class names* (resolved by identity at runtime),
    not a fragile lowercased substring of ``str(type(model))``; and
  * vae/text-encoder components are routed to their sibling ComfyUI folders, matching both the
    loaders and the pre-refactor on-disk layout.

The drift tripwire is also exercised here against a *faked* ``comfy.model_base`` so it stays
GPU-free; the real check runs at ``hordelib.initialise()`` time.
"""

import asyncio
import sys
import types
from collections.abc import Iterator
from pathlib import Path

import pytest
from horde_model_reference import component_relative_path
from horde_model_reference.meta_consts import KNOWN_IMAGE_GENERATION_BASELINE

from hordelib.execution import comfy_patches


def _all_referenced_class_names() -> set[str]:
    names = set(comfy_patches.FORCE_LOAD_SKIP_CLASS_NAMES)
    for class_names in comfy_patches._baseline_class_names().values():
        names.update(class_names)
    return names


def _install_fake_comfy_model_base(
    monkeypatch: pytest.MonkeyPatch,
    class_names: set[str],
) -> types.ModuleType:
    """Put a stand-in ``comfy`` (``model_base`` classes + a recording ``model_management``) into sys.modules.

    Returns the fake ``model_management`` module; its ``free_memory`` appends each requested byte count
    to ``free_memory_requests`` so tests can observe what a scoped cap let through.
    """
    model_base = types.ModuleType("comfy.model_base")
    for name in class_names:
        setattr(model_base, name, type(name, (), {}))
    model_management = types.ModuleType("comfy.model_management")
    model_management.free_memory_requests = []  # type: ignore[attr-defined]

    def free_memory(memory_required, device, *args, **kwargs):
        model_management.free_memory_requests.append(memory_required)  # type: ignore[attr-defined]

    model_management.free_memory = free_memory  # type: ignore[attr-defined]
    comfy_pkg = types.ModuleType("comfy")
    comfy_pkg.model_base = model_base  # type: ignore[attr-defined]
    comfy_pkg.model_management = model_management  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "comfy", comfy_pkg)
    monkeypatch.setitem(sys.modules, "comfy.model_base", model_base)
    monkeypatch.setitem(sys.modules, "comfy.model_management", model_management)
    return model_management


class TestForceLoadSkipResolution:
    def test_default_list_uses_comfy_class_names(self) -> None:
        # The class names must match comfy.model_base exactly; the old "qwen_image" fragment
        # never matched the comfy class QwenImage (str(type()).lower() == "...qwenimage...").
        assert "QwenImage" in comfy_patches.models_not_to_force_load
        assert "Krea2" in comfy_patches.models_not_to_force_load
        assert "Anima" in comfy_patches.models_not_to_force_load
        assert "MOSSImage" in comfy_patches.models_not_to_force_load
        assert "Flux" in comfy_patches.models_not_to_force_load
        assert "qwen_image" not in comfy_patches.models_not_to_force_load

    def test_baseline_resolves_to_class_names(self) -> None:
        resolved = comfy_patches.resolve_force_load_skip_entries(
            [KNOWN_IMAGE_GENERATION_BASELINE.qwen_image],
        )
        assert resolved == ["QwenImage"]

    def test_cascade_baseline_resolves_to_both_stages(self) -> None:
        resolved = comfy_patches.resolve_force_load_skip_entries(
            [KNOWN_IMAGE_GENERATION_BASELINE.stable_cascade],
        )
        assert resolved == ["StableCascade_C", "StableCascade_B"]

    @pytest.mark.parametrize(
        ("baseline", "expected"),
        [
            (KNOWN_IMAGE_GENERATION_BASELINE.krea2_turbo, ["Krea2"]),
            (KNOWN_IMAGE_GENERATION_BASELINE.anima, ["Anima"]),
            (KNOWN_IMAGE_GENERATION_BASELINE.moss_image, ["MOSSImage"]),
        ],
    )
    def test_graph_compatible_baselines_keep_distinct_load_policies(self, baseline, expected) -> None:
        assert comfy_patches.resolve_force_load_skip_entries([baseline]) == expected

    def test_raw_strings_pass_through_and_dedupe(self) -> None:
        resolved = comfy_patches.resolve_force_load_skip_entries(
            [
                KNOWN_IMAGE_GENERATION_BASELINE.flux_1,
                KNOWN_IMAGE_GENERATION_BASELINE.flux_dev,  # also -> "Flux"; must dedupe
                "SomeCustomClass",
            ],
        )
        assert resolved == ["Flux", "SomeCustomClass"]


class TestComponentRelativePath:
    def test_vae_is_redirected_to_sibling_folder(self) -> None:
        assert component_relative_path("qwen_image_vae.safetensors", "vae") == Path(
            "../vae/qwen_image_vae.safetensors"
        )

    def test_text_encoders_redirected(self) -> None:
        assert component_relative_path("te.safetensors", "text_encoders") == Path("../text_encoders/te.safetensors")
        # tolerate the singular spelling used by the node loader's COMPONENT_FILE_TYPES
        assert component_relative_path("te.safetensors", "text_encoder") == Path("../text_encoders/te.safetensors")

    def test_unet_and_unknown_stay_in_manager_folder(self) -> None:
        assert component_relative_path("qwen_image_fp8.safetensors", "unet") == Path("qwen_image_fp8.safetensors")
        assert component_relative_path("model.safetensors", None) == Path("model.safetensors")
        assert component_relative_path("model.safetensors", "checkpoint") == Path("model.safetensors")


class TestForceLoadClassNameDriftTripwire:
    """`assert_force_load_class_names_exist` is the fail-fast for the hard-coded comfy class names."""

    def test_baseline_map_and_flat_list_agree(self) -> None:
        # Internal source-of-truth consistency, checkable without comfy: the baseline map may
        # only reference names declared in FORCE_LOAD_SKIP_CLASS_NAMES, and vice versa.
        baseline_names = {n for names in comfy_patches._baseline_class_names().values() for n in names}
        assert baseline_names == set(comfy_patches.FORCE_LOAD_SKIP_CLASS_NAMES)

    def test_passes_when_all_classes_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_comfy_model_base(monkeypatch, _all_referenced_class_names())
        comfy_patches.assert_force_load_class_names_exist()  # must not raise

    def test_raises_when_a_referenced_class_is_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        drifted = _all_referenced_class_names()
        drifted.discard("QwenImage")  # simulate ComfyUI renaming/removing the class
        _install_fake_comfy_model_base(monkeypatch, drifted)
        with pytest.raises(RuntimeError, match="QwenImage"):
            comfy_patches.assert_force_load_class_names_exist()


class _FakePatcher:
    """Stand-in ModelPatcher exposing just what the load hijack inspects."""

    def __init__(self, model: object, size: int) -> None:
        self.model = model
        self._size = size

    def model_size(self) -> int:
        return self._size


class TestSmallModelWorkingClamp:
    """VAE-sized loads must not evict co-resident diffusion weights to satisfy a working estimate.

    Callers pass a worst-case working-memory estimate (a 1MP decode estimates several GB) as
    ``memory_required``; ComfyUI frees that much up front, evicting a resident diffusion model to
    host a few hundred MB of VAE. The hijack clamps the estimate for small non-diffusion loads,
    relying on ComfyUI's tiled-decode OOM fallback as the backstop.
    """

    def _fake_base_model_cls(self, monkeypatch: pytest.MonkeyPatch) -> type:
        _install_fake_comfy_model_base(monkeypatch, _all_referenced_class_names() | {"BaseModel"})
        import comfy.model_base

        return comfy.model_base.BaseModel

    def test_small_support_models_qualify(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._fake_base_model_cls(monkeypatch)
        vae = _FakePatcher(model=object(), size=160 * 1024**2)
        assert comfy_patches._small_support_models_only([vae]) is True

    def test_diffusion_model_disqualifies(self, monkeypatch: pytest.MonkeyPatch) -> None:
        base_model_cls = self._fake_base_model_cls(monkeypatch)
        unet = _FakePatcher(model=base_model_cls(), size=160 * 1024**2)
        assert comfy_patches._small_support_models_only([unet]) is False

    def test_oversize_support_model_disqualifies(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._fake_base_model_cls(monkeypatch)
        big = _FakePatcher(model=object(), size=comfy_patches.SMALL_MODEL_WORKING_CLAMP_MAX_BYTES + 1)
        assert comfy_patches._small_support_models_only([big]) is False

    def test_sizing_failure_fails_closed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._fake_base_model_cls(monkeypatch)

        class _Unsizable:
            model = object()

            def model_size(self) -> int:
                raise RuntimeError("no size")

        assert comfy_patches._small_support_models_only([_Unsizable()]) is False

    def test_hijack_clamps_vae_working_estimate(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._fake_base_model_cls(monkeypatch)
        captured: dict = {}

        def capture(models, **kwargs):
            captured.update(kwargs)

        monkeypatch.setitem(comfy_patches._originals, "load_models_gpu", capture)
        vae = _FakePatcher(model=object(), size=160 * 1024**2)

        comfy_patches._load_models_gpu_hijack([vae], memory_required=5 * 1024**3)

        assert captured["force_full_load"] is True
        assert captured["memory_required"] == 0

    def test_hijack_caps_free_memory_during_small_load(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A free_memory call made while the load runs is capped near the weights, not comfy's floor.

        Zeroing ``memory_required`` still leaves ComfyUI freeing its ~1GB inference reserve, which on a
        packed card evicts a resident diffusion model; the scoped cap is what actually prevents that.
        """
        self._fake_base_model_cls(monkeypatch)
        import comfy.model_management as fake_model_management

        weights = 160 * 1024**2
        expected_cap = int(weights * 1.1) + 256 * 1024**2

        def capture(models, **kwargs):
            fake_model_management.free_memory(10 * 1024**3, None)

        monkeypatch.setitem(comfy_patches._originals, "load_models_gpu", capture)
        vae = _FakePatcher(model=object(), size=weights)

        comfy_patches._load_models_gpu_hijack([vae], memory_required=5 * 1024**3)

        assert fake_model_management.free_memory_requests == [expected_cap]
        # Outside the scope the original is restored: an uncapped request passes through unchanged.
        fake_model_management.free_memory(10 * 1024**3, None)
        assert fake_model_management.free_memory_requests[-1] == 10 * 1024**3

    def test_hijack_preserves_estimate_for_diffusion_models(self, monkeypatch: pytest.MonkeyPatch) -> None:
        base_model_cls = self._fake_base_model_cls(monkeypatch)
        captured: dict = {}

        def capture(models, **kwargs):
            captured.update(kwargs)

        monkeypatch.setitem(comfy_patches._originals, "load_models_gpu", capture)
        unet = _FakePatcher(model=base_model_cls(), size=5 * 1024**3)

        comfy_patches._load_models_gpu_hijack([unet], memory_required=7 * 1024**3)

        assert captured["force_full_load"] is True
        assert captured["memory_required"] == 7 * 1024**3


MB = 1024**2
_CUDA = "cuda:0"
_RESERVE_BYTES = 1434 * MB
"""The worker's ``--reserve-vram`` figure, which ComfyUI adds to every load's request."""


class _DiffusionModel:
    """Stands in for a diffusion model's module. The capped line reports its class name."""


class _TextEncoder:
    """Stands in for a text encoder's module."""


class _Autoencoder:
    """Stands in for a small resident VAE's module."""


class _SimPatcher:
    """A stand-in ModelPatcher. The simulated ``free_memory`` lowers its device residency."""

    def __init__(self, model: object, *, size_bytes: int, loaded_bytes: int, is_clip: bool = False) -> None:
        self.model = model
        self.size_bytes = size_bytes
        self.loaded_bytes = loaded_bytes
        if is_clip:
            self.is_clip = True

    def model_size(self) -> int:
        return self.size_bytes

    def loaded_size(self) -> int:
        return self.loaded_bytes

    def is_clone(self, other: object) -> bool:
        return getattr(other, "model", None) is self.model


class _SimLoadedModel:
    """A stand-in for ComfyUI's ``LoadedModel`` entry in ``current_loaded_models``."""

    def __init__(self, patcher: _SimPatcher) -> None:
        self.model = patcher
        self.device = _CUDA

    def model_loaded_memory(self) -> int:
        return self.model.loaded_bytes


def _install_simulated_card(
    monkeypatch: pytest.MonkeyPatch,
    *,
    free_bytes: int,
    residents: list[_SimLoadedModel],
) -> types.ModuleType:
    """Fake ``comfy.model_management`` with a card whose ``free_memory`` unloads residents in list order.

    Each ``free_memory`` call is recorded as ``(target, keep_loaded)``. A resident not kept gives up what the
    target still lacks, as ComfyUI's partial unload does, without the offload buffer.
    """
    model_management = _install_fake_comfy_model_base(monkeypatch, {"BaseModel"})
    card = types.SimpleNamespace(free_bytes=free_bytes)
    model_management.card = card  # type: ignore[attr-defined]
    model_management.free_memory_calls = []  # type: ignore[attr-defined]
    model_management.current_loaded_models = residents  # type: ignore[attr-defined]
    model_management.get_torch_device = lambda: _CUDA  # type: ignore[attr-defined]
    model_management.get_free_memory = lambda dev=None, torch_free_too=False: card.free_bytes  # type: ignore[attr-defined]
    model_management.extra_reserved_memory = lambda: _RESERVE_BYTES  # type: ignore[attr-defined]
    model_management.minimum_inference_memory = lambda: 1024**3 * 0.8 + _RESERVE_BYTES  # type: ignore[attr-defined]

    def free_memory(memory_required, device, keep_loaded=(), **kwargs):
        model_management.free_memory_calls.append((memory_required, list(keep_loaded)))  # type: ignore[attr-defined]
        for loaded_model in residents:
            if any(loaded_model is kept for kept in keep_loaded):
                continue
            shortfall = memory_required - card.free_bytes
            if shortfall <= 0:
                continue
            freed = min(loaded_model.model.loaded_bytes, int(shortfall))
            loaded_model.model.loaded_bytes -= freed
            card.free_bytes += freed

    model_management.free_memory = free_memory  # type: ignore[attr-defined]
    return model_management


def _simulated_load_models_gpu(models, memory_required=0, **kwargs) -> None:
    """Make ComfyUI's two ``free_memory`` requests for a load that is not forced, then stop before the load."""
    import comfy.model_management as model_management

    extra_bytes = max(model_management.minimum_inference_memory(), memory_required + _RESERVE_BYTES)
    weight_bytes = sum(patcher.model_size() - patcher.loaded_size() for patcher in models)
    model_management.free_memory(
        weight_bytes * 1.1 + extra_bytes, _CUDA, for_dynamic=False, pins_required=weight_bytes
    )
    if model_management.get_free_memory(_CUDA) < extra_bytes:
        model_management.free_memory(extra_bytes, _CUDA, for_dynamic=False)


class TestTextEncoderLoadBesideJobDiffusionModel:
    """A text encoder that does not fit beside the job's kept diffusion model must not page that model out.

    ComfyUI's request for the encoder is its weights times 1.1 plus the inference floor and the reserve, and
    it takes the room from the diffusion model for the same job's next sample, which is then uploaded again.
    The figures are the measured ones: 1744 MB free, a 242 MB VAE, a 12533 MB diffusion model and a 4999 MB
    encoder.
    """

    @pytest.fixture
    def info_lines(self) -> Iterator[list[str]]:
        from loguru import logger

        messages: list[str] = []
        handler_id = logger.add(lambda message: messages.append(message.record["message"]), level="INFO")
        yield messages
        logger.remove(handler_id)

    @pytest.fixture
    def overflow_branch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(comfy_patches, "_do_not_force_load_model_in_patcher", lambda patcher: False)
        monkeypatch.setattr(comfy_patches, "_force_full_load_would_overflow_vram", lambda models: True)
        monkeypatch.setitem(comfy_patches._originals, "load_models_gpu", _simulated_load_models_gpu)

    @staticmethod
    def _diffusion() -> _SimLoadedModel:
        return _SimLoadedModel(_SimPatcher(_DiffusionModel(), size_bytes=12533 * MB, loaded_bytes=12533 * MB))

    @staticmethod
    def _vae() -> _SimLoadedModel:
        return _SimLoadedModel(_SimPatcher(_Autoencoder(), size_bytes=242 * MB, loaded_bytes=242 * MB))

    @staticmethod
    def _encoder() -> _SimPatcher:
        return _SimPatcher(_TextEncoder(), size_bytes=4999 * MB, loaded_bytes=0, is_clip=True)

    @staticmethod
    def _capped_lines(messages: list[str]) -> list[str]:
        return [message for message in messages if message.startswith(comfy_patches.TEXT_ENCODER_EVICTION_CAPPED_TEXT)]

    def test_the_jobs_diffusion_model_stays_whole_and_the_vae_may_go(
        self,
        monkeypatch: pytest.MonkeyPatch,
        overflow_branch: None,
        info_lines: list[str],
    ) -> None:
        """The diffusion model is visited first, so a cap alone would page part of it out. It must be kept."""
        diffusion = self._diffusion()
        vae = self._vae()
        model_management = _install_simulated_card(monkeypatch, free_bytes=1744 * MB, residents=[diffusion, vae])
        lora_clone = _SimPatcher(diffusion.model.model, size_bytes=12533 * MB, loaded_bytes=0)
        monkeypatch.setattr(comfy_patches, "_job_diffusion_model_patchers", lambda: [lora_clone])

        comfy_patches._load_models_gpu_hijack([self._encoder()], memory_required=0)

        assert diffusion.model.loaded_bytes == 12533 * MB
        assert vae.model.loaded_bytes == 0
        assert self._capped_lines(info_lines) == [
            "Text encoder load capped beside the job's diffusion model: mode=keep_diffusion protected=_DiffusionModel"
            " requested_mb=7752,2253 cap_mb=1986 free_mb=1744 room_mb=1986 floor_mb=819"
        ]
        assert all(target <= 1986 * MB for target, _ in model_management.free_memory_calls)

    def test_both_free_memory_calls_are_capped_and_keep_the_diffusion_model(
        self,
        monkeypatch: pytest.MonkeyPatch,
        overflow_branch: None,
    ) -> None:
        """The room left is under ComfyUI's minimum, so it calls ``free_memory`` a second time inside the load."""
        diffusion = self._diffusion()
        model_management = _install_simulated_card(monkeypatch, free_bytes=1744 * MB, residents=[diffusion])
        monkeypatch.setattr(comfy_patches, "_job_diffusion_model_patchers", lambda: [diffusion.model])

        comfy_patches._load_models_gpu_hijack([self._encoder()], memory_required=0)

        calls = model_management.free_memory_calls
        assert len(calls) == 2
        assert [target for target, _ in calls] == [1744 * MB, 1744 * MB]
        assert all(any(kept is diffusion for kept in keep_loaded) for _, keep_loaded in calls)
        assert diffusion.model.loaded_bytes == 12533 * MB
        # Outside the load the module function is ComfyUI's again.
        model_management.free_memory(10 * 1024**3, _CUDA)
        assert model_management.free_memory_calls[-1] == (10 * 1024**3, [])

    def test_a_stale_model_from_another_job_is_evicted_as_before(
        self,
        monkeypatch: pytest.MonkeyPatch,
        overflow_branch: None,
        info_lines: list[str],
    ) -> None:
        stale = self._diffusion()
        vae = self._vae()
        model_management = _install_simulated_card(monkeypatch, free_bytes=1744 * MB, residents=[vae, stale])
        this_jobs_model = _SimPatcher(_DiffusionModel(), size_bytes=12533 * MB, loaded_bytes=0)
        monkeypatch.setattr(comfy_patches, "_job_diffusion_model_patchers", lambda: [this_jobs_model])

        comfy_patches._load_models_gpu_hijack([self._encoder()], memory_required=0)

        request_bytes = 4999 * MB * 1.1 + 1024**3 * 0.8 + _RESERVE_BYTES
        assert model_management.free_memory_calls[0] == (request_bytes, [])
        assert stale.model.loaded_bytes == 12533 * MB - int(request_bytes - 1986 * MB)
        assert self._capped_lines(info_lines) == []

    def test_a_room_under_the_floor_bounds_the_eviction_at_the_shortfall(
        self,
        monkeypatch: pytest.MonkeyPatch,
        overflow_branch: None,
        info_lines: list[str],
    ) -> None:
        """An encoder declaring 3000 MB of working memory beside 1000 MB of room cannot stream safely."""
        diffusion = self._diffusion()
        model_management = _install_simulated_card(monkeypatch, free_bytes=1000 * MB, residents=[diffusion])
        monkeypatch.setattr(comfy_patches, "_job_diffusion_model_patchers", lambda: [diffusion.model])

        comfy_patches._load_models_gpu_hijack([self._encoder()], memory_required=3000 * MB)

        todays_request_bytes = 4999 * MB * 1.1 + 3000 * MB + _RESERVE_BYTES
        shortfall_target_bytes = 4999 * MB + 3000 * MB
        assert model_management.free_memory_calls == [(shortfall_target_bytes, [])]
        assert shortfall_target_bytes < todays_request_bytes
        assert diffusion.model.loaded_bytes == 12533 * MB - (shortfall_target_bytes - 1000 * MB)
        assert self._capped_lines(info_lines) == [
            "Text encoder load capped beside the job's diffusion model: mode=shortfall protected=_DiffusionModel"
            " requested_mb=9933 cap_mb=7999 free_mb=1000 room_mb=1000 floor_mb=3000"
        ]

    def test_the_shortfall_cap_never_raises_a_smaller_request(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _install_simulated_card(monkeypatch, free_bytes=0, residents=[])
        import comfy.model_management as model_management

        with comfy_patches._free_memory_capped(8000 * MB) as requested_targets:
            model_management.free_memory(2000 * MB, _CUDA)

        assert requested_targets == [2000 * MB]
        assert model_management.free_memory_calls == [(2000 * MB, [])]

    @pytest.mark.parametrize(
        "case",
        [
            pytest.param("diffusion_load", id="a-non-encoder-load"),
            pytest.param("no_job_model", id="no-diffusion-model-of-the-job"),
        ],
    )
    def test_other_loads_are_unchanged(
        self,
        monkeypatch: pytest.MonkeyPatch,
        overflow_branch: None,
        info_lines: list[str],
        case: str,
    ) -> None:
        resident = self._diffusion()
        model_management = _install_simulated_card(monkeypatch, free_bytes=1744 * MB, residents=[resident])
        if case == "diffusion_load":
            monkeypatch.setattr(comfy_patches, "_job_diffusion_model_patchers", lambda: [resident.model])
            loading = _SimPatcher(_DiffusionModel(), size_bytes=4999 * MB, loaded_bytes=0)
        else:
            monkeypatch.setattr(comfy_patches, "_job_diffusion_model_patchers", list)
            loading = self._encoder()

        comfy_patches._load_models_gpu_hijack([loading], memory_required=0)

        request_bytes = 4999 * MB * 1.1 + 1024**3 * 0.8 + _RESERVE_BYTES
        assert model_management.free_memory_calls[0] == (request_bytes, [])
        assert resident.model.loaded_bytes == 12533 * MB - int(request_bytes - 1744 * MB)
        assert self._capped_lines(info_lines) == []


class _FakeDynamicPrompt:
    """A stand-in for ComfyUI's ``DynamicPrompt``: node id to the prompt's node dict."""

    def __init__(self, nodes: dict[str, dict]) -> None:
        self._nodes = nodes

    def all_node_ids(self) -> list[str]:
        return list(self._nodes)

    def get_node(self, node_id: str) -> dict:
        return self._nodes[node_id]


class TestJobDiffusionModelPatchers:
    """The job's diffusion model is the cache entry the running prompt's model loader is served from."""

    def test_only_the_model_loaders_cache_entry_is_the_jobs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import hordelib.nodes
        from hordelib.execution import component_cache

        job_patcher = object()
        stage_patcher = object()
        served = {
            ("unet", "Krea2-Turbo:unet"): types.SimpleNamespace(payload=(job_patcher, None, None)),
            ("checkpoint", "Encode-Only"): types.SimpleNamespace(payload=(stage_patcher, None, None)),
        }
        fake_cache = types.SimpleNamespace(get=lambda key: served.get(key))
        fake_loader_module = types.ModuleType("hordelib.nodes.node_model_loader")
        fake_loader_module.NODE_CLASS_MAPPINGS = {"HordeCheckpointLoader": object}  # type: ignore[attr-defined]
        fake_loader_module.main_loader_cache_key = lambda name, file_type: (  # type: ignore[attr-defined]
            ("unet", f"{name}:{file_type}") if file_type == "unet" else ("checkpoint", name)
        )
        monkeypatch.setitem(sys.modules, "hordelib.nodes.node_model_loader", fake_loader_module)
        monkeypatch.setattr(hordelib.nodes, "node_model_loader", fake_loader_module, raising=False)
        monkeypatch.setattr(component_cache, "process_component_cache", lambda: fake_cache)
        dynprompt = _FakeDynamicPrompt(
            {
                "model_loader": {
                    "class_type": "HordeCheckpointLoader",
                    "inputs": {"horde_model_name": "Krea2-Turbo", "file_type": "unet"},
                },
                "stage_loader": {
                    "class_type": "HordeCheckpointLoader",
                    "inputs": {"horde_model_name": "Encode-Only", "file_type": None, "output_model": False},
                },
                "clip_loader": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen.safetensors"}},
                "linked_loader": {
                    "class_type": "HordeCheckpointLoader",
                    "inputs": {"horde_model_name": ["upstream", 0], "file_type": None},
                },
            },
        )
        monkeypatch.setattr(comfy_patches, "_running_prompt_graph", lambda: dynprompt)

        assert comfy_patches._job_diffusion_model_patchers() == [job_patcher]

    def test_outside_a_prompt_run_no_model_is_the_jobs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(comfy_patches, "_running_prompt_graph", lambda: None)

        assert comfy_patches._job_diffusion_model_patchers() == []


_ROOM_BESIDE_FOREIGN_MB = 15600
"""The card's room for the worker and other tenants, after the CUDA contexts. Less the foreign share, it is the
room shared by this process's weights, buffers and free memory, which is 14400 MB when other applications hold
1200 MB."""

_SAMPLING_RESERVE_MB = 1519
"""The activation reserve in the measured grants."""

_ENCODE_BUFFERS_MB = 800
"""The allocator growth of a streamed Krea2 encode, which the end-of-node buffer release returns to the driver."""


class _StreamedEncodeCard:
    """A card whose free memory is the room less the foreign share, the diffusion weights and the encode's buffers.

    ComfyUI's reading is honest here, so the free-VRAM clamp's ceiling decides every figure read by the loads.
    """

    def __init__(self, *, foreign_mb: int, diffusion: _SimPatcher) -> None:
        self.foreign_bytes = foreign_mb * MB
        self.diffusion = diffusion
        self.buffer_bytes = 0
        self.host_figures: list[tuple[float, float]] = []

    def reserved(self) -> int:
        return self.diffusion.loaded_bytes + self.buffer_bytes

    def device_free_mb(self) -> float:
        return (_ROOM_BESIDE_FOREIGN_MB * MB - self.foreign_bytes - self.reserved()) / MB


def _simulated_comfy_load(
    models,
    memory_required=0,
    force_patch_weights=False,
    minimum_memory_required=None,
    force_full_load=False,
) -> None:
    """ComfyUI's ``load_models_gpu`` sizing, by bytes instead of by module.

    A forced load places the whole model. Otherwise the model ends at the free memory plus what it holds, less
    ComfyUI's minimum (the caller's minimum plus the reserve, at least the inference floor), which trims a model
    when the free memory is under the minimum. A text encoder is streamed and does not place weights, as the measured
    ``loaded partially; ... 0.00 MB loaded, 4999.10 MB offloaded`` line shows.
    """
    import comfy.model_management as model_management

    device = model_management.get_torch_device()
    inference_bytes = model_management.minimum_inference_memory()
    extra_bytes = max(inference_bytes, memory_required + _RESERVE_BYTES)
    minimum_bytes = (
        extra_bytes
        if minimum_memory_required is None
        else max(inference_bytes, minimum_memory_required + _RESERVE_BYTES)
    )
    still_to_load = sum(patcher.model_size() - patcher.loaded_size() for patcher in models)
    model_management.free_memory(still_to_load * 1.1 + extra_bytes, device)
    for patcher in models:
        if getattr(patcher, "is_clip", False):
            continue
        if force_full_load:
            patcher.loaded_bytes = patcher.size_bytes
            continue
        loaded_bytes = patcher.loaded_bytes
        current_free = model_management.get_free_memory(device) + loaded_bytes
        usable = max(
            0,
            current_free - minimum_bytes,
            min(current_free * 0.4, current_free - inference_bytes),
        )
        patcher.loaded_bytes = int(min(patcher.size_bytes, max(0, usable)))


class TestKeptDiffusionModelAfterAStreamedEncode:
    """After a streamed encode, the sample-time load restores the kept diffusion model to the largest size that fits.

    The encoder streams beside the kept model and its cast buffers sit on the device during the encode. The host
    measured the grant's figure then, so the figure counts the buffers, which the end-of-node release returns to
    the driver before the sample. A rebase on that figure would size the sample-time load short by the buffers on
    every job. The figures are the measured ones: a 12533 MB diffusion model kept at 10044 MB, a 4999 MB encoder, a
    1519 MB activation reserve and ComfyUI's 1434 MB reserve.
    """

    @pytest.fixture
    def debug_lines(self) -> Iterator[list[str]]:
        from loguru import logger

        messages: list[str] = []
        handler_id = logger.add(lambda message: messages.append(message.record["message"]), level="DEBUG")
        yield messages
        logger.remove(handler_id)

    @staticmethod
    def _install(monkeypatch: pytest.MonkeyPatch, *, foreign_mb: int, kept_mb: int) -> types.SimpleNamespace:
        """Install the fake ComfyUI, allocator, clock and lease state for a run of jobs on one card."""
        from hordelib.execution import sampling_lease

        device = types.SimpleNamespace(type="cuda")
        diffusion = _SimPatcher(_DiffusionModel(), size_bytes=12533 * MB, loaded_bytes=kept_mb * MB)
        encoder = _SimPatcher(_TextEncoder(), size_bytes=4999 * MB, loaded_bytes=0, is_clip=True)
        card = _StreamedEncodeCard(foreign_mb=foreign_mb, diffusion=diffusion)
        clock = [0.0]

        model_management = _install_fake_comfy_model_base(monkeypatch, {"BaseModel"})
        resident_entries = [_SimLoadedModel(diffusion), _SimLoadedModel(encoder)]
        for entry in resident_entries:
            entry.device = device
        model_management.current_loaded_models = resident_entries  # type: ignore[attr-defined]
        model_management.get_torch_device = lambda: device  # type: ignore[attr-defined]
        model_management.extra_reserved_memory = lambda: _RESERVE_BYTES  # type: ignore[attr-defined]
        model_management.minimum_inference_memory = lambda: 1024**3 * 0.8 + _RESERVE_BYTES  # type: ignore[attr-defined]
        model_management.STREAM_CAST_BUFFERS = {}  # type: ignore[attr-defined]

        def get_free_memory(dev=None, torch_free_too=False):
            free_bytes = card.device_free_mb() * MB
            return (free_bytes, 0) if torch_free_too else free_bytes

        def reset_cast_buffers() -> None:
            model_management.STREAM_CAST_BUFFERS.clear()  # type: ignore[attr-defined]
            card.buffer_bytes = 0

        model_management.get_free_memory = get_free_memory  # type: ignore[attr-defined]
        model_management.reset_cast_buffers = reset_cast_buffers  # type: ignore[attr-defined]
        memory_management = types.ModuleType("comfy.memory_management")
        memory_management.aimdo_enabled = False  # type: ignore[attr-defined]
        sys.modules["comfy"].memory_management = memory_management  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "comfy.memory_management", memory_management)

        async def streamed_encode(*args, **kwargs):
            clock[0] += 1.0
            model_management.STREAM_CAST_BUFFERS[None] = object()  # type: ignore[attr-defined]
            card.buffer_bytes = _ENCODE_BUFFERS_MB * MB
            clock[0] += 1.0
            card.host_figures.append((card.device_free_mb(), clock[0]))
            clock[0] += 1.0

        monkeypatch.setattr(comfy_patches.torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(comfy_patches.torch.cuda, "memory_reserved", lambda dev=None: card.reserved())
        monkeypatch.setattr(comfy_patches, "_monotonic", lambda: clock[0])
        monkeypatch.setattr(comfy_patches, "_active_free_view_clamp", None)
        monkeypatch.setitem(comfy_patches._originals, "load_models_gpu", _simulated_comfy_load)
        monkeypatch.setitem(comfy_patches._originals, "get_output_data", streamed_encode)
        monkeypatch.setattr(comfy_patches, "_job_diffusion_model_patchers", lambda: [diffusion])
        monkeypatch.setattr(sampling_lease._lease_scope, "sampling_reserve_mb", None, raising=False)
        guider = types.SimpleNamespace(model_patcher=types.SimpleNamespace(load_device=device))
        return types.SimpleNamespace(
            card=card,
            clock=clock,
            diffusion=diffusion,
            encoder=encoder,
            guider=guider,
            lease_scope=sampling_lease._lease_scope,
            release=sampling_lease._release_text_encoders_before_wait,
        )

    @staticmethod
    def _run_job(env: types.SimpleNamespace) -> int:
        """Run one job's encode, release, grant and both sample-time loads. Return the diffusion MB loaded."""
        env.clock[0] += 10.0
        run_start_figure_mb = env.card.device_free_mb()
        env.clock[0] += 1.0
        with comfy_patches.free_memory_view_clamped(run_start_figure_mb):
            env.lease_scope.sampling_reserve_mb = None
            env.clock[0] += 1.0
            comfy_patches._load_models_gpu_hijack([env.encoder], memory_required=0)
            asyncio.run(comfy_patches._get_output_data_releasing_cast_buffers("prompt", "prompt", None, {}))
            env.release(env.guider)
            grant_figure_mb, measured_at = env.card.host_figures[-1]
            env.clock[0] += 1.0
            for _ in ("stage", "sample"):
                comfy_patches.rebase_free_memory_view_clamp(grant_figure_mb, measured_at=measured_at)
                env.lease_scope.sampling_reserve_mb = _SAMPLING_RESERVE_MB
                env.clock[0] += 1.0
                comfy_patches._load_models_gpu_hijack(
                    [env.diffusion],
                    memory_required=_SAMPLING_RESERVE_MB * MB,
                    minimum_memory_required=_SAMPLING_RESERVE_MB * MB,
                    force_full_load=False,
                )
            env.lease_scope.sampling_reserve_mb = None
        return env.diffusion.loaded_bytes // MB

    @pytest.mark.parametrize(
        ("foreign_mb", "expected_mb"),
        [
            pytest.param(1200, 12533, id="1200MB-foreign-loads-completely"),
            pytest.param(1800, 13800 - 2953, id="1800MB-foreign-loads-the-room-less-comfys-minimum"),
        ],
    )
    def test_the_sample_restores_the_kept_model_to_the_room_and_holds_it_over_a_streak(
        self,
        monkeypatch: pytest.MonkeyPatch,
        debug_lines: list[str],
        foreign_mb: int,
        expected_mb: int,
    ) -> None:
        """With 1200 MB foreign the room is 14400 MB, and the whole model leaves its 1519 MB reserve plus 5%.

        With 1800 MB foreign the room is 13800 MB, under the whole model plus that reserve, so ComfyUI sizes a
        partial load that leaves its minimum free. The minimum is the 1519 MB reserve plus ComfyUI's 1434 MB, or
        2953 MB. A rebase on the figures measured during the encode has 800 MB less room and ends at 10647 MB and
        10047 MB.
        """
        env = self._install(monkeypatch, foreign_mb=foreign_mb, kept_mb=10044)

        loaded_mb_per_job = [self._run_job(env) for _ in range(4)]

        assert loaded_mb_per_job == [expected_mb] * 4
        assert env.encoder.loaded_bytes == 0
        kept_lines = [line for line in debug_lines if line.startswith("Kept comfy's free-VRAM clamp")]
        assert len(kept_lines) == 8

    def test_a_figure_measured_after_the_streamed_node_is_used(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A sibling's release after the encode is still credited, from the checkpoint taken after the buffers left."""
        env = self._install(monkeypatch, foreign_mb=1800, kept_mb=10044)
        with comfy_patches.free_memory_view_clamped(env.card.device_free_mb()):
            env.clock[0] += 1.0
            asyncio.run(comfy_patches._get_output_data_releasing_cast_buffers("prompt", "prompt", None, {}))
            env.clock[0] += 1.0
            env.card.foreign_bytes -= 500 * MB
            later_figure_mb = env.card.device_free_mb()

            assert comfy_patches.rebase_free_memory_view_clamp(later_figure_mb, measured_at=env.clock[0])

            import comfy.model_management as model_management

            assert model_management.get_free_memory() / MB == pytest.approx(later_figure_mb)

    def test_a_streamed_node_logs_its_allocator_high_water_once(
        self,
        monkeypatch: pytest.MonkeyPatch,
        debug_lines: list[str],
    ) -> None:
        """The line records the buffers before release, and a node that did not stream weights does not log one."""
        env = self._install(monkeypatch, foreign_mb=1800, kept_mb=10044)

        class CLIPTextEncode:
            """Stands in for the encode node's instance. The line reports its class name."""

        async def resident_node(*args, **kwargs):
            env.clock[0] += 1.0

        with comfy_patches.free_memory_view_clamped(env.card.device_free_mb()):
            asyncio.run(
                comfy_patches._get_output_data_releasing_cast_buffers("prompt", "prompt", CLIPTextEncode(), {})
            )
            monkeypatch.setitem(comfy_patches._originals, "get_output_data", resident_node)
            asyncio.run(comfy_patches._get_output_data_releasing_cast_buffers("prompt", "sampler", object(), {}))

        high_water_lines = [
            line for line in debug_lines if line.startswith(comfy_patches.STREAMED_NODE_HIGH_WATER_TEXT)
        ]
        assert high_water_lines == [
            "Streamed node allocator high-water: node=prompt class=CLIPTextEncode growth_mb=800"
            " reserved_start_mb=10044 reserved_before_release_mb=10844 free_view_at_start_mb=3756"
        ]

    def test_a_node_that_streamed_nothing_records_no_span(self, monkeypatch: pytest.MonkeyPatch) -> None:
        env = self._install(monkeypatch, foreign_mb=1800, kept_mb=10044)

        async def resident_node(*args, **kwargs):
            env.clock[0] += 1.0
            env.card.host_figures.append((env.card.device_free_mb(), env.clock[0]))

        monkeypatch.setitem(comfy_patches._originals, "get_output_data", resident_node)
        with comfy_patches.free_memory_view_clamped(env.card.device_free_mb()):
            asyncio.run(comfy_patches._get_output_data_releasing_cast_buffers("prompt", "prompt", None, {}))
            figure_mb, measured_at = env.card.host_figures[-1]

            assert comfy_patches._active_free_view_clamp.own_loads == []
            assert comfy_patches.rebase_free_memory_view_clamp(figure_mb, measured_at=measured_at)
