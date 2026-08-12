"""Real-GPU proof that the host's VRAM retention grant holds the card between jobs.

``Comfy_Horde._run_pipeline`` evicts explicitly at the end of every job unless the host defers, so
the card comes back regardless of ComfyUI's memory mode. The other half is that a *granted* job can
keep its model: that requires smart memory enabled, because under ``--disable-smart-memory`` the
executor unloads everything at the end of each prompt and ``free_memory`` unloads unconditionally
(both pinned in ``tests/test_comfy_contract_drift.py``). Both are call-time reads of
``comfy.model_management.DISABLE_SMART_MEMORY``, so the smart-memory regime is entered here by
setting that global for the duration of the test rather than by launching another process.

Real-GPU test, marked ``slow`` plus the checkpoint's model marker (matching
``tests/test_component_cache_gpu.py``). Run manually and serially, for example::

    uv run --no-sync pytest tests/test_retention_window_gpu.py -m slow
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from hordelib.comfy_horde import unload_all_models_vram
from hordelib.execution.component_cache import ComponentCacheKey, ComponentSlotKind
from hordelib.horde import HordeLib
from hordelib.shared_model_manager import SharedModelManager

_SEED = 1234567890

_RESIDENT_FLOOR_BYTES = 512 * 1024 * 1024
"""A resident SD1.5 UNet is well above this; anything smaller is not a held checkpoint."""

_BASELINE_TOLERANCE_BYTES = 256 * 1024 * 1024
"""Slack over the empty-card baseline for allocator residue that is not model weights."""


def _txt2img_job(model_name: str) -> dict[str, Any]:
    """Build a minimal deterministic txt2img job."""
    return {
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
        "ddim_steps": 8,
        "n_iter": 1,
        "model": model_name,
    }


def _resident_weight_bytes() -> int:
    """Return the weight bytes ComfyUI currently holds on the device across all loaded entries."""
    import comfy.model_management as mm

    return sum(entry.model_loaded_memory() for entry in mm.current_loaded_models if not entry.is_dead())


def _allocated_bytes() -> int:
    """Return torch's own allocation total for the inference device (truthful under WDDM)."""
    import comfy.model_management as mm

    return int(torch.cuda.memory_allocated(mm.get_torch_device()))


def _return_the_card() -> None:
    """Free every model and release the allocator, leaving the session's later tests a clean card."""
    import comfy.model_management as mm

    mm.unload_all_models()
    unload_all_models_vram()


class TestRetentionWindow:
    @pytest.mark.slow
    @pytest.mark.default_sd15_model
    def test_grant_holds_the_model_and_the_next_ungranted_job_clears_it(
        self,
        shared_model_manager: type[SharedModelManager],
        hordelib_instance: HordeLib,
        stable_diffusion_model_name_for_testing: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Ungranted jobs return the card; a granted job leaves its own checkpoint resident.

        Three jobs on the same model: without a grant, without a grant plus the grant, and without a
        grant again. Residency is read from ComfyUI's loaded-model accounting and from torch's own
        allocation total, and the retained entry is identified as the job's checkpoint through the
        component cache rather than being inferred from the allocation alone.
        """
        import comfy.model_management as mm

        if not torch.cuda.is_available():
            pytest.skip("retention windows are a device-level property; no accelerator available")

        model_name = stable_diffusion_model_name_for_testing
        compvis = shared_model_manager.manager.compvis
        assert compvis is not None
        if model_name not in compvis.available_models:
            pytest.skip(f"{model_name} checkpoint is not available on disk")

        # Smart memory on: the executor stops unloading at the end of each prompt and free_memory
        # frees by shortfall, which is what leaves room for a grant to mean anything.
        monkeypatch.setattr(mm, "DISABLE_SMART_MEMORY", False)

        try:
            _return_the_card()
            baseline_allocated = _allocated_bytes()
            assert _resident_weight_bytes() == 0, "the card was not clean at the start of the window"

            hordelib_instance.basic_inference(_txt2img_job(model_name), defer_vram_unload=False)

            after_first = _resident_weight_bytes()
            assert after_first == 0, (
                f"an ungranted job left {after_first} bytes of weights resident; the explicit end-of-job "
                "eviction did not free the card"
            )
            allocated_after_first = _allocated_bytes()
            assert allocated_after_first <= baseline_allocated + _BASELINE_TOLERANCE_BYTES, (
                f"allocation after an ungranted job is {allocated_after_first} against a baseline of "
                f"{baseline_allocated}; the card was not returned"
            )

            hordelib_instance.basic_inference(_txt2img_job(model_name), defer_vram_unload=True)

            after_granted = _resident_weight_bytes()
            assert after_granted > _RESIDENT_FLOOR_BYTES, (
                f"a granted job left only {after_granted} bytes resident; the retention grant did not hold "
                "the model on the card"
            )
            allocated_after_granted = _allocated_bytes()
            assert allocated_after_granted >= baseline_allocated + _RESIDENT_FLOOR_BYTES, (
                f"allocation after a granted job is {allocated_after_granted} against a baseline of "
                f"{baseline_allocated}; the weights comfy reports resident are not on the device"
            )

            # The retained entry must be this job's checkpoint. Comfy loads a clone of the cached base,
            # so identity runs through the torch module both share.
            cached = shared_model_manager.manager._models_in_ram.get(
                ComponentCacheKey(ComponentSlotKind.CHECKPOINT, model_name),
            )
            assert cached is not None, "the job's checkpoint is not in the component cache"
            base_module = cached.payload[0].model
            resident_modules = [entry.model.model for entry in mm.current_loaded_models if not entry.is_dead()]
            assert any(module is base_module for module in resident_modules), (
                "the models held after a granted job are not this job's checkpoint; the retention window "
                "is holding something else"
            )

            hordelib_instance.basic_inference(_txt2img_job(model_name), defer_vram_unload=False)

            after_last = _resident_weight_bytes()
            assert after_last == 0, (
                f"the ungranted job following a grant left {after_last} bytes resident; a retention window "
                "is not closed by the next job that is denied one"
            )
            allocated_after_last = _allocated_bytes()
            assert allocated_after_last <= baseline_allocated + _BASELINE_TOLERANCE_BYTES, (
                f"allocation after the window closed is {allocated_after_last} against a baseline of "
                f"{baseline_allocated}; the retained model's memory was not returned"
            )
        finally:
            _return_the_card()
