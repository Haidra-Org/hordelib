"""Real-GPU tests for the MB-budgeted component cache.

Two properties that only a real load can prove:

- **A LoRA job's patches never reach the next job's output.** ComfyUI's ``ModelPatcher.clone`` shares both
  the underlying module and the patch backup with the base, so a LoRA-bearing job does patch the object the
  cache holds. The residue sits on the entry until something clears it: comfy unpatches at the component's
  next load whenever the applied patch set differs, and the restore lever clears it on demand. The gate
  loads a base, checksums it, runs a LoRA-bearing generation, asserts the residency report shows the
  residue, asserts the restore lever returns the weights to the pristine checksum, and asserts a re-run of
  the identical non-LoRA job reproduces its pre-LoRA output byte-for-byte (same process, same cached load).
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
        # The clone lifts weights comfy wrote under torch.inference_mode into normal tensors; .numpy()
        # refuses inference tensors, and a same-device .to() hands back the original.
        tensor = state_dict[key].detach().to("cpu").contiguous().clone()
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
        """A LoRA run's residue is reported, clears on restore, and never reaches the next job's output.

        Uses SD1.5 with a known-applicable LoRA (GlowingRunesAI), because a LoRA whose keys did not match
        the base would apply no patches and pass vacuously, so a real patch must be exercised.

        The harness starts ComfyUI with smart memory disabled, and under that flag the executor unloads
        (and thereby unpatches) every model at the end of each prompt, so a pipeline run cannot leave the
        patched-resident state behind for the assertions to see. The worker's default serving mode does
        the same (it passes ``--disable-smart-memory`` unless the operator opts into
        ``comfy_smart_memory``); with that opt-in a LoRA job's clone stays loaded between jobs and the
        shared base carries its patches at rest. That opt-in state is recreated explicitly: patch the
        cached base through comfy's own LoRA loader and load the clone, exactly what the graph's LoRA
        node does, then assert the residue is reported, the restore lever returns the weights to their
        pristine values, and an identical non-LoRA job reproduces its earlier output.
        """
        import comfy.model_management
        import comfy.sd
        import comfy.utils
        import folder_paths

        from hordelib.api import restore_components

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
        base_clip = base_entry.payload[1]
        checksum_before = _state_dict_checksum(base_model)

        lora_result: ResultingImageReturn = hordelib_instance.basic_inference_single_image(
            _txt2img_job(model_name, lora_name=lora_GlowingRunesAI),
        )
        assert isinstance(lora_result.image, Image.Image)
        assert len(lora_result.faults) == 0

        after_entry = cache.get(checkpoint_key)
        assert after_entry is not None, "the LoRA job must be served the same base, still resident"
        assert after_entry.payload[0] is base_model, "the cached base object must be reused, not replaced"

        lora_filename = shared_model_manager.manager.lora.get_lora_filename(lora_GlowingRunesAI)
        assert lora_filename is not None
        lora_state_dict = comfy.utils.load_torch_file(
            folder_paths.get_full_path("loras", lora_filename),
            safe_load=True,
        )
        model_lora, _clip_lora = comfy.sd.load_lora_for_models(base_model, base_clip, lora_state_dict, 1.0, 1.0)
        try:
            # Loading the clone bakes its patches into the shared module, the patched-resident state a
            # smart-memory deployment holds between jobs.
            comfy.model_management.load_models_gpu([model_lora], force_full_load=True)

            checksum_patched = _state_dict_checksum(base_model)
            assert checksum_patched != checksum_before, (
                "loading the LoRA clone left the base weights unchanged; the LoRA applied no patches and "
                "the pristine assertion below would be measuring luck"
            )

            # The cache serves entries by reference and does not normalise them, so the residue is
            # visible on the resident entry.
            residue = {snapshot.identity: snapshot.mutated for snapshot in cache.held_report()}
            assert residue.get(model_name) is True, "the patched base's residue is not reported"

            assert restore_components([model_name]) == 1
            cleared = {snapshot.identity: snapshot.mutated for snapshot in cache.held_report()}
            assert cleared.get(model_name) is False, "restoring did not clear the residue from the cached base"
            checksum_after = _state_dict_checksum(base_model)
            assert checksum_after == checksum_before, "restoring the patched base did not return it pristine"
        finally:
            comfy.model_management.unload_all_models()

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


def _vram_mb() -> tuple[float, float]:
    """Return (torch-allocated MB, device-used MB).

    The two move independently and that difference is the point: releasing a model's weights returns
    their blocks to the torch caching allocator, which holds them reserved rather than handing them back,
    so the card's used figure does not move until the allocator cache is emptied.
    """
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated() / 1024**2
    free, total = torch.cuda.mem_get_info()
    return allocated, (total - free) / 1024**2


class TestComponentRestoreReclaimsDeviceMemory:
    """What restoring a resident component actually reclaims, and what it costs the next job.

    These pin the semantics the worker's VRAM reclaim ladder relies on. Restoring is meant to sit below a
    whole-model VRAM unload: give up the device memory, keep the pristine weights in host RAM, so the next
    job for the same model re-uploads instead of re-reading the checkpoint from disk.
    """

    @pytest.mark.slow
    @pytest.mark.default_sd15_model
    def test_restore_releases_weights_but_only_the_allocator_release_reaches_the_card(
        self,
        init_horde,
        shared_model_manager: type[SharedModelManager],
        hordelib_instance: HordeLib,
        stable_diffusion_model_name_for_testing: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Restoring frees the weights; the card only sees it once the allocator cache is released.

        This is the whole reason the restore rung must pair with an allocator release. A rung that stopped
        after restoring would report a reclaim the arbiter could not measure, because the card's free
        memory would be unchanged.

        Aggressive unloading is turned off for the duration so the model is still on the card when the
        measurement starts. That is the state the reclaim ladder acts on: a lane holding a resident model.
        With aggressive unloading on, the job's own teardown has already cleared the card and there is
        nothing for any rung to reclaim.
        """
        from hordelib.api import restore_components
        from hordelib.comfy_horde import _comfy_soft_empty_cache

        model_name = stable_diffusion_model_name_for_testing
        cache = shared_model_manager.manager._models_in_ram
        cache.evict_all()

        # Residency is established explicitly rather than left to whatever the job's teardown happens to
        # leave behind. The session fixture starts ComfyUI with --reserve-vram and asks for most of VRAM
        # and RAM to be left free, so under it a finished job leaves nothing on the card; those are
        # start-up settings that cannot be undone mid-session. Loading the cached patcher directly puts
        # the process into the one state the reclaim rung ever acts on, a lane holding a resident model,
        # without depending on harness policy to produce it.
        import comfy.model_management

        hordelib_instance.basic_inference_single_image(_txt2img_job(model_name))

        entry = cache.get(ComponentCacheKey(ComponentSlotKind.CHECKPOINT, model_name))
        assert entry is not None, "the checkpoint should be resident in RAM after a generation"
        patcher = entry.payload[0]
        comfy.model_management.load_models_gpu([patcher], force_full_load=True)

        allocated_before, device_before = _vram_mb()
        assert allocated_before > 500, (
            f"the model did not reach the card even when loaded explicitly, saw {allocated_before:.0f}MB"
        )
        try:
            assert restore_components([model_name]) == 1
            allocated_after, device_after = _vram_mb()

            # The weights are genuinely released...
            assert allocated_after < allocated_before / 2, (
                f"restoring did not release the model's device allocation "
                f"({allocated_before:.0f}MB -> {allocated_after:.0f}MB)"
            )
            # ...into torch's reserve, where the card cannot see the reclaim.
            assert device_after >= device_before - 64, (
                "the card reported a reclaim before the allocator cache was released; if torch's allocator "
                "started returning blocks eagerly, the restore rung no longer needs to pair with a release"
            )

            _comfy_soft_empty_cache()
            _, device_released = _vram_mb()
            assert device_released < device_before - 500, (
                f"releasing the allocator cache after a restore did not return device memory "
                f"({device_before:.0f}MB -> {device_released:.0f}MB)"
            )
        finally:
            comfy.model_management.unload_all_models()
            _comfy_soft_empty_cache()

    @pytest.mark.slow
    @pytest.mark.default_sd15_model
    def test_a_restored_component_stays_warm_for_the_next_job(
        self,
        init_horde,
        shared_model_manager: type[SharedModelManager],
        hordelib_instance: HordeLib,
        stable_diffusion_model_name_for_testing: str,
    ) -> None:
        """After a restore the model is still in RAM, so the next job for it does not touch the disk.

        This is what makes restoring the cheaper rung: an eviction would leave the next job re-reading a
        multi-gigabyte checkpoint, and a restore leaves it re-uploading from host RAM.
        """
        from hordelib.api import restore_components

        model_name = stable_diffusion_model_name_for_testing
        cache = shared_model_manager.manager._models_in_ram
        cache.evict_all()

        first = hordelib_instance.basic_inference_single_image(_txt2img_job(model_name))
        assert isinstance(first.image, Image.Image)

        restore_components([model_name])

        resident = {snapshot.identity for snapshot in cache.held_report()}
        assert model_name in resident, "restoring must not evict the entry; that is the rung above it"

        collector = get_metrics_collector()
        collector.snapshot_and_reset_job()
        second = hordelib_instance.basic_inference_single_image(_txt2img_job(model_name))
        snapshot = collector.snapshot_and_reset_job()

        disk_loads = [
            event
            for event in (getattr(snapshot, "model_load_events", None) or [])
            if getattr(event, "phase", None) == "disk_to_ram"
        ]
        assert disk_loads == [], "the job after a restore re-read the checkpoint from disk instead of RAM"
        assert isinstance(second.image, Image.Image)
        assert second.image.tobytes() == first.image.tobytes(), (
            "the identical job produced different output after a restore; the restored weights differ "
            "from the ones the first job used"
        )


def _load_lora_state_dict(shared_model_manager: type[SharedModelManager], lora_name: str) -> dict:
    """Return the raw state dict of a LoRA the manager already has on disk."""
    import comfy.utils
    import folder_paths

    assert shared_model_manager.manager.lora is not None
    lora_filename = shared_model_manager.manager.lora.get_lora_filename(lora_name)
    assert lora_filename is not None, f"{lora_name} is not resolvable to a file on disk"
    return comfy.utils.load_torch_file(folder_paths.get_full_path("loras", lora_filename), safe_load=True)


class TestLoraServingCostPins:
    """What serving a LoRA costs on an already-resident base, and what the bypass loader avoids.

    ComfyUI identifies a patch set by ``ModelPatcher.patches_uuid``, a fresh uuid per ``add_patches`` call
    with no relation to the patch content. A second job applying the same LoRA at the same strength
    therefore cannot match the uuid the shared module records, and ``partially_load`` responds by unpatching
    the weights back to the offload device and re-uploading them. These pin that cost on real weights, and
    pin that ``load_bypass_lora_for_models`` sidesteps it by never writing the base weights at all.

    The harness starts ComfyUI with smart memory disabled, so the executor unloads (and unpatches)
    everything at the end of each prompt and a pipeline run cannot leave the patched-resident state behind.
    That state is manufactured directly here, the same way ``TestComponentCacheLoRaPoisoningGate`` does it.
    """

    @pytest.mark.slow
    @pytest.mark.default_sd15_model
    def test_an_identical_repeat_lora_still_pays_a_full_unpatch_and_reload(
        self,
        shared_model_manager: type[SharedModelManager],
        hordelib_instance: HordeLib,
        stable_diffusion_model_name_for_testing: str,
        lora_GlowingRunesAI: str,
    ) -> None:
        """Re-applying the identical LoRA at identical strength does not hit the zero-cost warm path.

        The second clone carries the same patch content as the one already baked into the shared module, yet
        its ``patches_uuid`` differs, so ``partially_load`` unpatches the weights (giving up the whole
        resident footprint) before loading them again.
        """
        import comfy.model_management
        import comfy.sd

        model_name = stable_diffusion_model_name_for_testing
        cache = shared_model_manager.manager._models_in_ram
        cache.evict_all()

        first: ResultingImageReturn = hordelib_instance.basic_inference_single_image(_txt2img_job(model_name))
        assert isinstance(first.image, Image.Image)
        assert len(first.faults) == 0

        base_entry = cache.get(ComponentCacheKey(ComponentSlotKind.CHECKPOINT, model_name))
        assert base_entry is not None, "the base checkpoint should be resident after a generation"
        base_model = base_entry.payload[0]
        base_clip = base_entry.payload[1]

        lora_state_dict = _load_lora_state_dict(shared_model_manager, lora_GlowingRunesAI)

        try:
            clone_one, _ = comfy.sd.load_lora_for_models(base_model, base_clip, lora_state_dict, 1.0, 1.0)
            comfy.model_management.load_models_gpu([clone_one], force_full_load=True)
            baked_uuid = base_model.model.current_weight_patches_uuid
            assert baked_uuid == clone_one.patches_uuid, "the first clone's patch set must be the baked one"
            assert base_model.model.model_loaded_weight_memory > 0, "the first clone must leave the base resident"

            clone_two, _ = comfy.sd.load_lora_for_models(base_model, base_clip, lora_state_dict, 1.0, 1.0)
            assert len(clone_two.patches) > 0, "the LoRA matched no keys, so the repeat below would prove nothing"
            assert base_model.model.current_weight_patches_uuid != clone_two.patches_uuid, (
                "comfy now recognises an identical patch set as already applied; the repeat-LoRA reload cost "
                "this pins no longer exists and the worker's serving cost model should be re-derived"
            )

            unpatch_calls: list[tuple[bool, int]] = []
            real_unpatch = clone_two.unpatch_model

            def recording_unpatch(device_to: object = None, unpatch_weights: bool = True) -> object:
                unpatch_calls.append((unpatch_weights, clone_two.model.model_loaded_weight_memory))
                return real_unpatch(device_to, unpatch_weights=unpatch_weights)

            clone_two.unpatch_model = recording_unpatch

            comfy.model_management.load_models_gpu([clone_two], force_full_load=True)

            weight_unpatches = [resident for unpatch_weights, resident in unpatch_calls if unpatch_weights]
            assert weight_unpatches, (
                f"loading an identically-patched clone no longer unpatches the weights first "
                f"(unpatch calls: {unpatch_calls}); the down-and-up this pins is gone"
            )
            assert max(weight_unpatches) > 0, (
                "the weight unpatch happened against an already-empty residency, so no re-upload was paid; "
                "the cost this pins is not where it was believed to be"
            )
            assert base_model.model.current_weight_patches_uuid == clone_two.patches_uuid, (
                "the second clone's load did not end with its own patch set recorded on the shared module"
            )
        finally:
            comfy.model_management.unload_all_models()

    @pytest.mark.slow
    @pytest.mark.default_sd15_model
    def test_the_bypass_loader_serves_a_lora_without_writing_the_base_weights(
        self,
        shared_model_manager: type[SharedModelManager],
        hordelib_instance: HordeLib,
        stable_diffusion_model_name_for_testing: str,
        lora_GlowingRunesAI: str,
    ) -> None:
        """``load_bypass_lora_for_models`` injects the LoRA into the forward pass, leaving weights pristine.

        Nothing is written to the shared module, so there is no backup to restore, no residue to report, and
        no patch identity for the next load to mismatch against. That is the whole difference from the baked
        path: the repeat cost the sibling test pins never arises.
        """
        import comfy.model_management
        import comfy.sd

        from hordelib.execution.component_restore import has_patch_residue

        model_name = stable_diffusion_model_name_for_testing
        cache = shared_model_manager.manager._models_in_ram
        cache.evict_all()

        first: ResultingImageReturn = hordelib_instance.basic_inference_single_image(_txt2img_job(model_name))
        assert isinstance(first.image, Image.Image)
        assert len(first.faults) == 0

        base_entry = cache.get(ComponentCacheKey(ComponentSlotKind.CHECKPOINT, model_name))
        assert base_entry is not None, "the base checkpoint should be resident after a generation"
        base_model = base_entry.payload[0]
        base_clip = base_entry.payload[1]

        lora_state_dict = _load_lora_state_dict(shared_model_manager, lora_GlowingRunesAI)
        checksum_before = _state_dict_checksum(base_model)

        try:
            bypass_model, _bypass_clip = comfy.sd.load_bypass_lora_for_models(
                base_model,
                base_clip,
                lora_state_dict,
                1.0,
                1.0,
            )
            assert bypass_model is not None
            assert "bypass_lora" in bypass_model.injections, (
                "the bypass loader registered no forward-pass injection, so the LoRA matched nothing and the "
                "pristine-weight assertions below would be measuring an empty apply"
            )
            comfy.model_management.load_models_gpu([bypass_model], force_full_load=True)

            assert _state_dict_checksum(base_model) == checksum_before, (
                "the bypass loader wrote the base weights; it is no longer a no-bake path and cannot be used "
                "to avoid the repeat-LoRA reload cost"
            )
            assert has_patch_residue(base_entry.payload) is False, (
                "the bypass load left patch residue on the cached component, so it does bake a foreign patch "
                "set into the shared module"
            )
            applied_uuid = base_model.model.current_weight_patches_uuid
            assert applied_uuid is None or applied_uuid == bypass_model.patches_uuid, (
                "the shared module records a patch identity that is neither absent nor the loading patcher's "
                "own; the bypass path baked something"
            )
        finally:
            comfy.model_management.unload_all_models()

    @pytest.mark.slow
    @pytest.mark.default_sd15_model
    def test_the_horde_lora_node_lets_an_identical_repeat_reuse_the_resident_bake(
        self,
        shared_model_manager: type[SharedModelManager],
        hordelib_instance: HordeLib,
        stable_diffusion_model_name_for_testing: str,
        lora_GlowingRunesAI: str,
    ) -> None:
        """Going through ``HordeLoraLoader``, the repeat's patch identity matches what is already baked.

        The sibling pin above shows raw ``comfy.sd.load_lora_for_models`` cannot match: its ``patches_uuid``
        is a fresh uuid4 per call. The node derives that uuid from the incoming patcher, the lora file, and
        the strength instead, so an identical repeat compares equal to the shared module's
        ``current_weight_patches_uuid`` and ``partially_load`` returns without unpatching the weights down to
        the offload device and re-uploading them.
        """
        import comfy.model_management
        import folder_paths

        from hordelib.nodes.node_lora_loader import HordeLoraLoader

        model_name = stable_diffusion_model_name_for_testing
        cache = shared_model_manager.manager._models_in_ram
        cache.evict_all()

        first: ResultingImageReturn = hordelib_instance.basic_inference_single_image(_txt2img_job(model_name))
        assert isinstance(first.image, Image.Image)
        assert len(first.faults) == 0

        base_entry = cache.get(ComponentCacheKey(ComponentSlotKind.CHECKPOINT, model_name))
        assert base_entry is not None, "the base checkpoint should be resident after a generation"
        base_model = base_entry.payload[0]
        base_clip = base_entry.payload[1]

        assert shared_model_manager.manager.lora is not None
        lora_filename = shared_model_manager.manager.lora.get_lora_filename(lora_GlowingRunesAI)
        assert lora_filename is not None, f"{lora_GlowingRunesAI} is not resolvable to a file on disk"
        assert lora_filename in folder_paths.get_filename_list("loras"), (
            "the node rejects any lora_name outside the folder_paths listing, so the calls below would "
            "silently return the unpatched inputs"
        )

        try:
            clone_one, _ = HordeLoraLoader().load_lora(base_model, base_clip, lora_filename, 1.0, 1.0)
            assert clone_one is not base_model, "the node returned its input, so no lora was applied"
            comfy.model_management.load_models_gpu([clone_one], force_full_load=True)

            baked_uuid = base_model.model.current_weight_patches_uuid
            assert baked_uuid == clone_one.patches_uuid, "the first clone's patch set must be the baked one"
            assert base_model.model.model_loaded_weight_memory > 0, "the first clone must leave the base resident"

            clone_two, _ = HordeLoraLoader().load_lora(base_model, base_clip, lora_filename, 1.0, 1.0)
            assert len(clone_two.patches) > 0, "the LoRA matched no keys, so the repeat below would prove nothing"
            assert clone_two.patches_uuid == baked_uuid, (
                "the node's derived patch identity does not reproduce across identical invocations, so an "
                "identical repeat still cannot be recognised as already baked"
            )

            unpatch_calls: list[tuple[bool, int]] = []
            real_unpatch = clone_two.unpatch_model

            def recording_unpatch(device_to: object = None, unpatch_weights: bool = True) -> object:
                unpatch_calls.append((unpatch_weights, clone_two.model.model_loaded_weight_memory))
                return real_unpatch(device_to, unpatch_weights=unpatch_weights)

            clone_two.unpatch_model = recording_unpatch

            comfy.model_management.load_models_gpu([clone_two], force_full_load=True)

            costly_unpatches = [
                resident for unpatch_weights, resident in unpatch_calls if unpatch_weights and resident
            ]
            assert not costly_unpatches, (
                f"the identical repeat still unpatched a resident model (unpatch calls: {unpatch_calls}); "
                f"matching patch identities did not reach partially_load's zero-cost return"
            )
            assert base_model.model.model_loaded_weight_memory > 0, "the repeat gave up the resident footprint"
            assert base_model.model.current_weight_patches_uuid == clone_two.patches_uuid

            # The guard side: a stack that would bake different weights must still pay the down-and-up.
            clone_three, _ = HordeLoraLoader().load_lora(base_model, base_clip, lora_filename, 0.5, 1.0)
            assert clone_three.patches_uuid != base_model.model.current_weight_patches_uuid, (
                "a different strength derived the same patch identity, so the resident bake would be served "
                "for a job that asked for different weights"
            )

            guard_unpatch_calls: list[tuple[bool, int]] = []
            real_guard_unpatch = clone_three.unpatch_model

            def recording_guard_unpatch(device_to: object = None, unpatch_weights: bool = True) -> object:
                guard_unpatch_calls.append((unpatch_weights, clone_three.model.model_loaded_weight_memory))
                return real_guard_unpatch(device_to, unpatch_weights=unpatch_weights)

            clone_three.unpatch_model = recording_guard_unpatch

            comfy.model_management.load_models_gpu([clone_three], force_full_load=True)

            assert [resident for unpatch_weights, resident in guard_unpatch_calls if unpatch_weights and resident], (
                f"a differently-strengthed stack did not unpatch the resident weights "
                f"(unpatch calls: {guard_unpatch_calls}); it was served the previous bake"
            )
            assert base_model.model.current_weight_patches_uuid == clone_three.patches_uuid
        finally:
            comfy.model_management.unload_all_models()
