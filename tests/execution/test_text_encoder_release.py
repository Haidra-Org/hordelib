"""GPU-free seam tests for the text encoder release the sampling lease hook makes before its grant wait.

The hook runs ComfyUI's real ``free_memory`` and ``LoadedModel`` over stand-in patchers that report their weights as
loaded. The encoder's ``detach`` goes through hordelib's ``unpatch_model`` hijack, and the restore it calls is
recorded, so the release is observed taking the CPU-origin path. The weights stay on the CPU here, where the restore
moves nothing. The device round trip itself is covered by ``test_weight_prefetch_and_cpu_retention``. Whether a later
node takes the encoder is read from a real ``DynamicPrompt`` and progress registry under ComfyUI's executing context.
"""

import weakref
from collections.abc import Iterator
from typing import Any

import pytest
import torch
from loguru import logger

from hordelib.execution import comfy_patches, cpu_weight_retention, sampling_lease, weight_prefetch

_PROMPT_ID = "text-encoder-release"
_DIFFUSION_LOAD_BYTES = 1024
"""What the stand-in diffusion load asks ``free_memory`` for. It is less than the encoder holds, so a resident
encoder is unloaded partially, as the diffusion load unloads a large one."""


class _StandInPatcher:
    """The ``ModelPatcher`` surface ``LoadedModel`` and ``free_memory`` read, reporting its weights as loaded."""

    def __init__(self, call_log: list[str], *, label: str, is_clip: bool) -> None:
        self.model = torch.nn.Sequential(torch.nn.Linear(64, 64), torch.nn.Linear(64, 64))
        self.on_device = True
        self.is_clip = is_clip
        self.parent = None
        self.backup: dict[str, Any] = {}
        self.load_device = torch.device("cpu")
        self.offload_device = torch.device("cpu")
        self.partial_unloads = 0
        self.label = label
        self._call_log = call_log

    def model_size(self) -> int:
        return sum(param.numel() * param.element_size() for param in self.model.parameters())

    def loaded_size(self) -> int:
        return self.model_size() if self.on_device else 0

    def is_dynamic(self) -> bool:
        return False

    def partially_unload(self, device_to: torch.device, memory_to_free: int) -> int:
        self.partial_unloads += 1
        self._call_log.append(f"{self.label} partially unloaded")
        return memory_to_free

    def detach(self, unpatch_all: bool = True) -> torch.nn.Module:
        self._call_log.append(f"{self.label} unloaded")
        comfy_patches._model_patcher_unpatch_model_hijack(self, self.offload_device, unpatch_all)
        self.on_device = False
        return self.model


class _GuiderPatcher:
    """The diffusion ``ModelPatcher`` surface ``CFGGuider.sample`` reads before it reaches ``outer_sample``."""

    def __init__(self) -> None:
        self.model = torch.nn.Linear(1, 1)
        self.model_options: dict[str, Any] = {"transformer_options": {}}
        self.wrappers: dict[str, Any] = {}
        self.callbacks: dict[str, Any] = {}
        self.hook_mode: object = None
        self.load_device = torch.device("cpu")
        self.offload_device = torch.device("cpu")

    def is_dynamic(self) -> bool:
        return False

    def register_all_hook_patches(self, *args: object, **kwargs: object) -> None:
        return None

    def restore_hook_patches(self) -> None:
        return None


class _LoggingLease:
    """A semaphore-shaped lease that logs each acquire and release in the shared call log."""

    def __init__(self, call_log: list[str], name: str) -> None:
        self._call_log = call_log
        self._name = name

    def acquire(self, block: bool = True, timeout: float | None = None) -> bool:
        self._call_log.append(f"{self._name} acquire")
        return True

    def release(self) -> None:
        self._call_log.append(f"{self._name} release")


class _Card:
    """ComfyUI's loaded-model list for one test, holding a stand-in diffusion model and text encoder."""

    def __init__(self, call_log: list[str], model_management: Any) -> None:
        self.call_log = call_log
        self.restored_modules: list[torch.nn.Module] = []
        self.diffusion = _StandInPatcher(call_log, label="diffusion", is_clip=False)
        self.encoder = _StandInPatcher(call_log, label="encoder", is_clip=True)
        self._model_management = model_management
        for patcher in (self.diffusion, self.encoder):
            loaded_model = model_management.LoadedModel(patcher)
            # What ``LoadedModel.model_load`` leaves behind, which ``free_memory`` and ``model_unload`` read.
            loaded_model.real_model = weakref.ref(patcher.model)
            loaded_model.model_finalizer = weakref.finalize(patcher.model, lambda: None)
            model_management.current_loaded_models.append(loaded_model)

    def loaded_patchers(self) -> list[object]:
        return [loaded_model.model for loaded_model in self._model_management.current_loaded_models]

    def diffusion_load(self) -> None:
        """Free room for the diffusion model as its load does, keeping the diffusion model itself."""
        kept = [
            loaded_model
            for loaded_model in self._model_management.current_loaded_models
            if loaded_model.model is self.diffusion
        ]
        self._model_management.free_memory(_DIFFUSION_LOAD_BYTES, self.diffusion.load_device, keep_loaded=kept)


@pytest.fixture
def card(init_horde: None, monkeypatch: pytest.MonkeyPatch) -> _Card:
    """Install the hook over a fresh loaded-model list, with ``outer_sample`` running the diffusion load's free."""
    import comfy.model_management as model_management
    import comfy.sampler_helpers as sampler_helpers
    import comfy.samplers as samplers
    from comfy_execution import progress

    call_log: list[str] = []
    monkeypatch.setattr(model_management, "current_loaded_models", [])
    # A fixed device-free reading, so the diffusion load's free asks for exactly its requirement.
    monkeypatch.setattr(
        model_management,
        "get_free_memory",
        lambda dev=None, torch_free_too=False: (0, 0) if torch_free_too else 0,
    )
    # With smart memory on, as the worker runs ComfyUI, a free smaller than a resident model unloads it partially.
    monkeypatch.setattr(model_management, "DISABLE_SMART_MEMORY", False)
    monkeypatch.setattr(progress, "global_progress_registry", None)
    # The hijack hands every backed-up key to ComfyUI's own unpatch after its restore; the stand-ins have none.
    monkeypatch.setitem(comfy_patches._originals, "model_patcher_unpatch_model", lambda *args, **kwargs: None)
    real_restore_cpu_origins = cpu_weight_retention.restore_cpu_origins
    monkeypatch.setattr(weight_prefetch, "prefetch_module_weights_async", lambda module, **kwargs: None)
    monkeypatch.setattr(samplers.CFGGuider, "sample", samplers.CFGGuider.sample)
    monkeypatch.setattr(sampler_helpers, "estimate_memory", sampler_helpers.estimate_memory)
    monkeypatch.setattr(sampling_lease, "_installed", False)
    monkeypatch.setattr(sampling_lease, "_sampling_lease", None)
    monkeypatch.setattr(sampling_lease, "_stage_lease", None)
    monkeypatch.setattr(sampling_lease, "_acquire_timeout_seconds", 120.0)
    monkeypatch.setattr(
        sampling_lease,
        "_stage_sampling_model",
        lambda guider, noise, conds, model_options: call_log.append("stage upload"),
    )
    test_card = _Card(call_log, model_management)

    def _recording_restore(module: torch.nn.Module, **kwargs: Any) -> tuple[int, int]:
        test_card.restored_modules.append(module)
        return real_restore_cpu_origins(module, **kwargs)

    monkeypatch.setattr(cpu_weight_retention, "restore_cpu_origins", _recording_restore)

    def _outer_sample(
        self: object, noise: torch.Tensor, latent_image: torch.Tensor, *args: object, **kwargs: object
    ) -> torch.Tensor:
        test_card.diffusion_load()
        call_log.append("sample")
        return latent_image

    monkeypatch.setattr(samplers.CFGGuider, "outer_sample", _outer_sample)
    sampling_lease.install_sampling_lease_hook()
    return test_card


@pytest.fixture
def release_lines() -> Iterator[list[str]]:
    """Collect the DEBUG lines the lease module logs about text encoders."""
    messages: list[str] = []
    handler_id = logger.add(
        lambda message: messages.append(message.record["message"]),
        level="DEBUG",
        filter=lambda record: record["name"] == sampling_lease.__name__ and "ext encoder" in record["message"],
    )
    yield messages
    logger.remove(handler_id)


def _single_pass_graph() -> dict[str, dict[str, Any]]:
    return {
        "loader": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "model.safetensors"}},
        "positive": {"class_type": "CLIPTextEncode", "inputs": {"text": "a cat", "clip": ["loader", 1]}},
        "negative": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["loader", 1]}},
        "latent": {"class_type": "EmptyLatentImage", "inputs": {"width": 64, "height": 64, "batch_size": 1}},
        "sampler": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["loader", 0],
                "positive": ["positive", 0],
                "negative": ["negative", 0],
                "latent_image": ["latent", 0],
                "seed": 1,
                "steps": 1,
                "cfg": 1.0,
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 1.0,
            },
        },
        "decode": {"class_type": "VAEDecode", "inputs": {"samples": ["sampler", 0], "vae": ["loader", 2]}},
    }


_SINGLE_PASS_FINISHED = ("loader", "positive", "negative", "latent")


def _second_pass_encode_graph() -> dict[str, dict[str, Any]]:
    """A two-pass graph whose second sampler takes a prompt encoded only after the first pass."""
    graph = _single_pass_graph()
    graph["second_positive"] = {"class_type": "CLIPTextEncode", "inputs": {"text": "detail", "clip": ["loader", 1]}}
    graph["second_sampler"] = {
        "class_type": "KSampler",
        "inputs": {
            **graph["sampler"]["inputs"],
            "positive": ["second_positive", 0],
            "latent_image": ["sampler", 0],
        },
    }
    graph["decode"]["inputs"]["samples"] = ["second_sampler", 0]
    return graph


def _sample_in_prompt(graph: dict[str, dict[str, Any]], finished: tuple[str, ...]) -> None:
    """Sample through a guider as the ``sampler`` node of a running prompt whose ``finished`` nodes are done."""
    from comfy_execution import progress
    from comfy_execution.graph import DynamicPrompt
    from comfy_execution.utils import CurrentNodeContext

    progress.reset_progress_state(_PROMPT_ID, DynamicPrompt(graph))
    for node_id in finished:
        progress.get_progress_state().finish_progress(node_id)
    with CurrentNodeContext(_PROMPT_ID, "sampler"):
        _sample_outside_a_prompt()


def _sample_outside_a_prompt() -> None:
    import comfy.samplers as samplers

    guider = samplers.CFGGuider(_GuiderPatcher())
    guider.set_conds([], [])
    guider.sample(torch.zeros(1, 4, 8, 8), torch.zeros(1, 4, 8, 8), None, torch.tensor([1.0, 0.0]))


def test_encoder_released_through_its_cpu_origins_before_the_stage_acquire(
    card: _Card,
    release_lines: list[str],
) -> None:
    sampling_lease.set_gpu_sampling_lease(_LoggingLease(card.call_log, "sampling"))
    sampling_lease.set_gpu_stage_lease(_LoggingLease(card.call_log, "stage"))
    encoder_bytes = card.encoder.loaded_size()

    _sample_in_prompt(_single_pass_graph(), _SINGLE_PASS_FINISHED)

    assert card.call_log == [
        "encoder unloaded",
        "stage acquire",
        "stage upload",
        "stage release",
        "sampling acquire",
        "sample",
        "sampling release",
    ]
    assert card.restored_modules == [card.encoder.model], "the release takes the CPU-origin restore"
    assert card.loaded_patchers() == [card.diffusion]
    assert release_lines == [
        f"Released text encoder(s) Sequential ({encoder_bytes // 2**20} MB) to their CPU origins before the "
        "sampling lease wait",
    ]


def test_encoder_released_before_the_single_sampling_acquire(card: _Card) -> None:
    sampling_lease.set_gpu_sampling_lease(_LoggingLease(card.call_log, "sampling"))

    _sample_in_prompt(_single_pass_graph(), _SINGLE_PASS_FINISHED)

    assert card.call_log == ["encoder unloaded", "sampling acquire", "sample", "sampling release"]
    assert card.restored_modules == [card.encoder.model]


def test_diffusion_load_finds_no_encoder_to_partially_unload(card: _Card) -> None:
    sampling_lease.set_gpu_sampling_lease(_LoggingLease(card.call_log, "sampling"))

    _sample_in_prompt(_single_pass_graph(), _SINGLE_PASS_FINISHED)

    assert card.encoder.partial_unloads == 0
    assert card.diffusion.partial_unloads == 0
    assert "diffusion unloaded" not in card.call_log


def test_pipeline_encoding_after_the_sampler_keeps_the_encoder(card: _Card, release_lines: list[str]) -> None:
    sampling_lease.set_gpu_sampling_lease(_LoggingLease(card.call_log, "sampling"))

    _sample_in_prompt(_second_pass_encode_graph(), _SINGLE_PASS_FINISHED)

    assert "encoder unloaded" not in card.call_log
    # Kept on the device, the encoder is what the diffusion load's free unloads partially.
    assert card.encoder.partial_unloads == 1
    assert card.encoder in card.loaded_patchers()
    assert release_lines == ["Text encoder kept on the device: a node of this prompt still to run takes it"]


def test_second_pass_encode_already_run_releases_the_encoder(card: _Card) -> None:
    sampling_lease.set_gpu_sampling_lease(_LoggingLease(card.call_log, "sampling"))

    _sample_in_prompt(_second_pass_encode_graph(), (*_SINGLE_PASS_FINISHED, "second_positive"))

    assert card.call_log[0] == "encoder unloaded"
    assert card.restored_modules == [card.encoder.model]


def test_link_from_an_unknown_node_class_keeps_the_encoder(card: _Card) -> None:
    sampling_lease.set_gpu_sampling_lease(_LoggingLease(card.call_log, "sampling"))
    graph = _single_pass_graph()
    graph["unknown"] = {"class_type": "NotARegisteredNode", "inputs": {}}
    graph["decode"]["inputs"]["vae"] = ["unknown", 0]

    _sample_in_prompt(graph, (*_SINGLE_PASS_FINISHED, "unknown"))

    assert "encoder unloaded" not in card.call_log


def test_outside_a_prompt_run_the_encoder_stays(card: _Card) -> None:
    sampling_lease.set_gpu_sampling_lease(_LoggingLease(card.call_log, "sampling"))

    _sample_outside_a_prompt()

    assert "encoder unloaded" not in card.call_log
    assert card.encoder in card.loaded_patchers()


def test_no_sampling_lease_leaves_the_encoder(card: _Card) -> None:
    _sample_in_prompt(_single_pass_graph(), _SINGLE_PASS_FINISHED)

    assert "encoder unloaded" not in card.call_log
    assert card.encoder in card.loaded_patchers()


def test_failed_release_still_samples(card: _Card, monkeypatch: pytest.MonkeyPatch) -> None:
    import comfy.model_management as model_management

    def _raising_free_memory(*args: object, **kwargs: object) -> list[object]:
        raise RuntimeError("free boom")

    sampling_lease.set_gpu_sampling_lease(_LoggingLease(card.call_log, "sampling"))
    monkeypatch.setattr(card, "diffusion_load", lambda: None)
    monkeypatch.setattr(model_management, "free_memory", _raising_free_memory)

    _sample_in_prompt(_single_pass_graph(), _SINGLE_PASS_FINISHED)

    assert card.call_log == ["sampling acquire", "sample", "sampling release"]
