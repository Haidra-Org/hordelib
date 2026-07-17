"""Real-GPU tests for the MB-budgeted component cache.

Two properties that only a real load can prove:

- **LoRA does not poison the cached base.** A LoRA-bearing job is served the pristine cached base, and the
  graph's LoRA loader clones that base before patching, so the cached weights must be untouched after the
  LoRA job runs (patches materialise on the clone at GPU load and revert at unload). The gate loads a base,
  checksums the cached state dict, runs a LoRA-bearing generation, and asserts the checksum is unchanged and
  that a re-run of the identical non-LoRA job reproduces its pre-LoRA output byte-for-byte (same process,
  same cached load).
- **Multiple components stay resident within the budget.** With a budget large enough for two checkpoints,
  alternating two models leaves both resident, so a second pass over both serves entirely from cache with no
  disk load.

These are real-GPU tests, marked ``slow`` plus the checkpoints' model markers (matching
``tests/test_stage_disaggregation.py``), and are deselected by the CI default ``-m "not slow"``. Run manually
and serially, for example::

    uv run --no-sync pytest tests/test_component_cache_gpu.py -m slow
"""

from __future__ import annotations

import hashlib

import pytest
import torch
from PIL import Image

from hordelib.execution.component_cache import (
    ComponentCache,
    ComponentCacheKey,
    ComponentSlotKind,
)
from hordelib.horde import HordeLib, ResultingImageReturn
from hordelib.metrics import get_metrics_collector
from hordelib.shared_model_manager import SharedModelManager

_SEED = 1234567890


def _state_dict_checksum(model_patcher) -> str:
    """Return a sha256 over the base model's state dict, device- and order-independent.

    Hashes each parameter's raw bytes on the CPU in sorted key order, so the digest reflects only the weight
    values, not their current device or the dict's iteration order.
    """
    digest = hashlib.sha256()
    state_dict = model_patcher.model.state_dict()
    for key in sorted(state_dict):
        tensor = state_dict[key].detach().to("cpu").contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _txt2img_job(model_name: str, *, lora_name: str | None = None) -> dict:
    """Build a minimal deterministic txt2img job, optionally carrying a single full-strength LoRA."""
    job: dict = {
        "sampler_name": "k_euler",
        "cfg_scale": 7.5,
        "denoising_strength": 1.0,
        "seed": _SEED,
        "height": 512,
        "width": 512,
        "karras": False,
        "tiling": False,
        "hires_fix": False,
        "clip_skip": 1,
        "prompt": "a dark magical crystal, 8K resolution",
        "ddim_steps": 20,
        "n_iter": 1,
        "model": model_name,
    }
    if lora_name is not None:
        job["loras"] = [{"name": lora_name, "model": 1.0, "clip": 1.0}]
    return job


class TestComponentCacheLoRaPoisoningGate:
    @pytest.mark.slow
    @pytest.mark.default_sd15_model
    def test_lora_run_does_not_poison_cached_base(
        self,
        shared_model_manager: type[SharedModelManager],
        hordelib_instance: HordeLib,
        stable_diffusion_model_name_for_testing: str,
        lora_GlowingRunesAI: str,
    ) -> None:
        """A LoRA-bearing generation must leave the cached pristine base weights and its output unchanged.

        Uses SD1.5 with a known-applicable LoRA (GlowingRunesAI): the clone-before-patch invariant being
        gated is model-agnostic, and this is the suite's proven-applicable LoRA. A LoRA whose keys did not
        match the base would apply no patches and vacuously pass, so a real patch must be exercised.
        """
        assert shared_model_manager.manager.lora
        model_name = stable_diffusion_model_name_for_testing
        cache = shared_model_manager.manager._models_in_ram
        cache.evict_all()

        checkpoint_key = ComponentCacheKey(ComponentSlotKind.CHECKPOINT, model_name)

        before: ResultingImageReturn = hordelib_instance.basic_inference_single_image(_txt2img_job(model_name))
        assert isinstance(before.image, Image.Image)
        assert len(before.faults) == 0

        base_entry = cache.get(checkpoint_key)
        assert base_entry is not None, "base checkpoint should be resident after a non-LoRA generation"
        base_model = base_entry.payload[0]
        checksum_before = _state_dict_checksum(base_model)

        lora_result: ResultingImageReturn = hordelib_instance.basic_inference_single_image(
            _txt2img_job(model_name, lora_name=lora_GlowingRunesAI),
        )
        assert isinstance(lora_result.image, Image.Image)
        assert len(lora_result.faults) == 0

        after_entry = cache.get(checkpoint_key)
        assert after_entry is not None, "the LoRA job must be served the same pristine base, still resident"
        assert after_entry.payload[0] is base_model, "the cached base object must be reused, not replaced"
        checksum_after = _state_dict_checksum(base_model)
        assert checksum_after == checksum_before, "the LoRA run mutated the cached base weights (poisoning)"

        rerun: ResultingImageReturn = hordelib_instance.basic_inference_single_image(_txt2img_job(model_name))
        assert isinstance(rerun.image, Image.Image)
        assert len(rerun.faults) == 0
        assert rerun.image.tobytes() == before.image.tobytes(), (
            "the identical non-LoRA job produced different output after the LoRA run; the shared base or its "
            "load path was poisoned"
        )


class TestComponentCacheMultiEntryResidency:
    @pytest.mark.slow
    @pytest.mark.default_sd15_model
    @pytest.mark.default_sdxl_model
    def test_two_models_stay_resident_within_budget(
        self,
        init_horde,
        shared_model_manager: type[SharedModelManager],
        stable_diffusion_model_name_for_testing: str,
        sdxl_1_0_base_model_name: str,
    ) -> None:
        """With a budget sized for both, alternating two models serves the second pass entirely from cache."""
        from hordelib.nodes.node_model_loader import HordeCheckpointLoader

        compvis = shared_model_manager.manager.compvis
        assert compvis is not None
        model_a = stable_diffusion_model_name_for_testing
        model_b = sdxl_1_0_base_model_name
        for model_name in (model_a, model_b):
            if model_name not in compvis.available_models:
                pytest.skip(f"{model_name} checkpoint is not available on disk")
            assert compvis.download_model(model_name)

        loader = HordeCheckpointLoader()
        collector = get_metrics_collector()
        original_cache = shared_model_manager.manager._models_in_ram
        try:
            # A budget comfortably larger than an SD1.5 + SDXL pair so neither displaces the other.
            shared_model_manager.manager._models_in_ram = ComponentCache(budget_mb=32000)

            for model_name in (model_a, model_b):
                loader.load_checkpoint(
                    will_load_loras=False,
                    seamless_tiling_enabled=False,
                    horde_model_name=model_name,
                    file_type=None,
                )

            resident = shared_model_manager.manager._models_in_ram.held_report()
            assert len(resident) == 2, f"expected both checkpoints resident, held: {resident}"

            collector.snapshot_and_reset_job()  # discard the warm-pass counters and disk-load events

            for model_name in (model_a, model_b):
                loader.load_checkpoint(
                    will_load_loras=False,
                    seamless_tiling_enabled=False,
                    horde_model_name=model_name,
                    file_type=None,
                )

            snapshot = collector.snapshot_and_reset_job()
            disk_loads = [event for event in snapshot.model_loads if event.phase == "disk_to_ram"]
            assert disk_loads == [], f"second pass hit disk: {disk_loads}"
            assert snapshot.component_cache_hits >= 2
            assert snapshot.component_cache_misses == 0
        finally:
            shared_model_manager.manager._models_in_ram = original_cache
