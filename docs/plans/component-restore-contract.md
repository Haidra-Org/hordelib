# Component restore contract

Status: acquire-side restoration was implemented, measured against the pinned ComfyUI's own
mechanisms, and retracted. What remains in place: the restore registry and explicit restore lever,
live residue reporting on the residency surface, the `hordelib.api` residency operations, and the
padding capture/reset. The reclaim-ladder placement of a restore rung is still open.

## The problem

`ComponentCache` serves loaded model components by reference. A job that mutates one is mutating an
object the next job will be handed. Two kinds of mutation reach a cached component:

| Mutation | Mechanism | Restoration |
| --- | --- | --- |
| Seamless tiling | `model.model.apply(make_circular / make_regular)` directly on the shared `nn.Module`; no patcher and no backup | Re-asserted on every read path; `make_regular` resets each convolution to its recorded as-constructed padding |
| LoRA | `comfy.sd.load_lora_for_models`, which clones the `ModelPatcher` and patches through it | ComfyUI's own, lazily at the component's next load |

## What ComfyUI actually guarantees

Properties of `ModelPatcher` in the pinned ComfyUI, confirmed by direct probe and by reading the
pinned source. They hold on both the current and previous pin.

- `clone()` shares the underlying `nn.Module` **and** the `backup` dict by reference. A clone is a
  handle onto the same weights, so "clone before patch" does not isolate the parent.
- `unpatch_model(device_to, unpatch_weights=True)` restores only the backed-up (patched) keys in
  place, then **unconditionally moves the entire module** to `device_to` and zeroes
  `model_loaded_weight_memory` and `current_weight_patches_uuid`. The whole-module move runs even
  when the backup is empty, so calling it on a never-patched component is a full device ejection for
  nothing.
- `partially_load` compares the module's `current_weight_patches_uuid` against the loading patcher's
  own `patches_uuid`. On a mismatch it does a **full unpatch to the offload device followed by a full
  re-upload**; on a match with the model fully loaded it returns having moved nothing. Pristine at
  next load is therefore a real guarantee, and the clean path (same patch set, still resident) costs
  zero bytes.
- `add_patches` regenerates `patches_uuid` on every call. Two consecutive jobs applying the identical
  LoRA at identical strength still mismatch, so every LoRA-bearing job pays the full model
  down-and-up under ComfyUI's own mechanism. There is no content hash.
- `LoadedModel.model_memory_required` reports zero additional bytes when the module's recorded device
  is already the load device, and the **full model size** when it is not. `load_models_gpu` sizes its
  `free_memory` call from this, so anything that flips a resident module's device to CPU makes the
  next load run eviction with a full-model target.

## Why acquire-side restoration was retracted

The eager restore-on-acquire design called `unpatch_model(offload_device)` on every patcher-shaped
object in a marked entry's payload before serving it. Against the semantics above, that is:

- **A wash on the slot the LoRA actually patched.** ComfyUI's next load was going to do the identical
  full round trip anyway, because the patch-set comparison always mismatches after a LoRA job.
- **Pure added cost on every other slot.** The VAE is never LoRA-patched and an untouched CLIP holds
  its own patch set, so ComfyUI's clean path would have moved nothing; the eager restore ejected them
  from the device and forced a re-upload regardless, because the mark was per entry, not per slot.
- **Added eviction pressure on everything else.** The restore flips the module's device to CPU, so
  the next load's `free_memory` runs with a full-model target instead of zero and evicts other
  resident components. It also unregisters and re-registers pinned host memory around every cycle.

These are per-job costs on LoRA-bearing traffic that the lazy path never pays, and correctness does
not need them: every production consumer reaches weights through a load, and pristine-at-next-load
covers that. Acquire-side normalisation was therefore retracted. `ComponentCache.get` hands entries
out untouched; `will_mutate` remains as a declaration that feeds the restore statistics.

## Consumers

**In-process node graph.** `node_model_loader` is the only production reader of the payloads. Every
path to weights goes through a ComfyUI load, which is what makes pristine-at-next-load sufficient.

**The worker.** `horde_process` uses `hordelib.api` for residency operations: `held_components()`
for the memory report it forwards to the parent, `evict_components()` when the parent answers RAM
pressure, and `restore_components()` as the explicit device-memory lever. The worker names
identities and never touches payloads.

The parent cannot choose a rung it cannot see, so `HeldComponentSnapshot` carries whether an entry
holds patch residue. The indicator is computed live from the components at report time (the module's
applied patch set differing from the cached patcher's own), not from a stored flag, so it answers
what the entry holds now rather than what an earlier acquisition intended. A stored flag has no
correct clearing point: ComfyUI clears residue lazily inside its own load, where the cache cannot
see it.

## Residue at rest is a serving-mode phenomenon

With smart memory enabled, a finished job leaves its models comfy-loaded, so a LoRA job's baked
patches persist on the shared module between jobs; that is the state the residue indicator reports
and the restore lever clears. With smart memory disabled, ComfyUI's executor unloads every model at
the end of each prompt, which also unpatches them, so residue never survives a run.

The worker's default is the disabled regime: `comfy_smart_memory` defaults off and the worker then
passes `--disable-smart-memory` to every ComfyUI child, because cross-process residency is not yet
reconciled at dispatch time. Smart memory on is the operator opt-in and the intended residency
endstate. Everything in this contract that concerns residue at rest, and every per-job cost the
retracted eager restore added, is a property of that opt-in regime; under the default regime the
module returns to pristine at each prompt end and the eager restore was a near-free walk over an
already-clean component. The test harness initialises with smart memory disabled, matching the
default regime, which is why the GPU gate for the poisoning contract establishes the
patched-resident state explicitly (patch the cached base through `comfy.sd.load_lora_for_models`
and load the clone) rather than relying on a pipeline run to leave it behind.

## Design as it stands

- **Restore registry** (`hordelib/execution/component_restore.py`): one place that knows how to
  return each payload shape to its loaded state, dispatching on the object. `ModelPatcher`-shaped
  payloads go through ComfyUI's `unpatch_model` to the patcher's own offload device, so the weights
  end where the accounting it resets says they are. Passing `device_to=None` instead would leave
  weights on the card while the bookkeeping reports nothing resident, which the worker's reclaim
  decisions cannot tolerate.
- **Residue probe** (`has_patch_residue`): the live predicate behind the snapshot indicator.
- **Explicit lever**: `restore_components` on `hordelib.api`, handled in the worker's lanes by the
  `RESTORE_COMPONENTS` control message, paired with an allocator-cache release because restoring
  alone returns blocks to torch's allocator where the card cannot see the reclaim. The paired
  behaviour is pinned by a GPU test.
- **Padding capture/reset**: `capture_pristine_state` records each convolution's as-constructed
  `padding_mode` at load; `make_regular` resets to the recorded mode rather than a blanket
  `"zeros"`, which is only correct for architectures built that way (the pinned tree constructs the
  Genmo/Mochi and ACE VAEs with `"replicate"`).
- **Statistics**: `ComponentRestoreStats` counts declared-mutator acquisitions, explicit restores,
  and the device bytes those restores released, reported through the worker's memory messages.

## Rulings

1. **A restore call is never gated behind a VRAM preference.** Standing constraint on any future
   release-side work.
2. **Acquire-side restoration is retracted.** The correctness contract is ComfyUI's
   pristine-at-next-load. The LoRA poisoning gate asserts the actual contract: residue is reported
   on the resident entry, the restore lever returns the weights to their pristine values, and an
   identical job reproduces its earlier output after an interleaved LoRA job.
3. **`pristine_lora_serving_enabled` and `entry_reusable` are retired.** Sharing is safe under the
   next-load guarantee; the knob had nothing left to protect.
4. **Tiling restores to captured state.** The as-constructed padding is recorded at load and
   restored exactly, closing the latent hazard for replicate-padded architectures.
5. **Scope is all consumers, including the worker**, through `hordelib.api` rather than the cache's
   internals.

## Open: where the restore rung belongs

The candidate placement is the VRAM reclaim ladder, between `RELEASE_ALLOCATOR_CACHE` (which frees
no model-held memory) and `UNLOAD_MODELS_FROM_VRAM` (which gives up residency entirely): a targeted
restore frees a named component's device memory while keeping pristine weights warm in RAM, so the
next job re-uploads instead of re-reading from disk. Restoring does not relieve host RAM and
slightly increases it, so it does not belong on the host-RAM pressure rung.

That is a scheduling change: churn accounting, `last_control_flag` bookkeeping that idle-and-teardown
predicates read, and under "two ladders, one actuator surface" a new reclaim rung is inherited by the
recovery ladder. There is no measurement yet showing a targeted restore beats the existing rungs on
workloads that reach them. The mechanism is in place and inert until that placement is ruled on.

## The LoRA reload tax, and the chosen lever

Because `add_patches` mints a fresh `patches_uuid` per call with no content hash, ComfyUI cannot
recognise that a job's patch set is the one already baked into the resident weights, so its lazy
mechanism pays a full model down-and-up for **every** LoRA-bearing job, even back-to-back jobs with
the identical LoRA at identical strength. The tax only bites in the smart-memory residency regime;
under the default disabled regime every job re-uploads everything regardless.

The tax and its mechanism are pinned by executable tests rather than by source reading:
`tests/test_comfy_contract_drift.py::TestModelPatcherResidencyPins` pins the uuid regeneration, the
missing content hash, and the unconditional accounting retraction on CPU;
`tests/test_component_cache_gpu.py::TestLoraServingCostPins` demonstrates on a real load that an
identical repeat LoRA still mismatches the baked patch set and pays the full unpatch.

**Chosen lever: content-derived patch identity in the loader node.** The patch set
`load_lora_for_models` produces is fully determined by the base component, the LoRA file, and the
two strengths, so `HordeLoraLoader` can assign the returned clones a `patches_uuid` derived
deterministically from that identity (folding in the incoming patcher's uuid so chained loaders
compose). Consecutive jobs with the same LoRA stack then match the module's recorded
`current_weight_patches_uuid` and take `partially_load`'s verified zero-byte fast path: the resident
baked weights are reused as-is. This obtains the repeat-case transfer win inside the bake path whose
correctness ComfyUI already guarantees, with no change to the compute path and no output-parity
question. Different-stack transitions still pay the bake round trip, which is inherent to baking.

### Bypass-LoRA: evaluated and rejected as-is

The pinned tree ships `comfy.sd.load_bypass_lora_for_models` (upstream, from the weight-adapter
bypass forward mode work), which applies adapter-backed LoRA content as forward-time injections
without touching base weights. A working mechanism demonstration is pinned
(`TestLoraServingCostPins`: real LoRA applied, base checksum byte-identical, no residue). Adoption
was evaluated and rejected on the pinned tree for cause:

- Chained loaders collapse: every load stores its injections under the same `"bypass_lora"` key on
  the cloned patcher, so a chain of loader nodes silently serves only the last LoRA.
- The eject path restores instance `forward` attributes in injection order rather than reverse, so
  stacked injections leave the first hook welded onto the shared module and the next inject captures
  its own hook as the original, recursing. Partial loads eject and reinject routinely, so this fires
  under ordinary VRAM pressure.
- A dropped clone leaks: `LoadedModel` holds the patcher by weakref, nothing else ejects, and the
  shared module keeps the previous job's monkeypatched forward plus GPU-resident adapter tensors,
  invisible to `patches_uuid`, `backup`, and the memory accounting alike.
- Coverage and equivalence gaps: bias diffs, norm-layer patches, full diffs, and `set_weight`
  content still bake; DoRA silently degrades to plain LoRA; OFT/BOFT bypass output is mathematically
  different from baking (rotated bias, rescale asymmetry); LoHa and OFT pay pathological
  per-forward costs; adapter tensors are excluded from every memory figure the load hijack reads.
- Upstream ships no tests for the path and labels its only inference consumer a debugging node.

Revisit only if the injection lifecycle is fixed (unique keys, reverse-order eject, explicit
end-of-job eject on model and CLIP patchers, adapter memory accounted), most plausibly as upstream
contributions; the drift pins will show if a future pin improves any of this.

## Open: the lowvram latch

`ModelPatcher.partially_unload` sets `model_lowvram = True` on the shared module even when its
guarded loop freed zero bytes (modules lacking `comfy_patched_weights` are skipped but the flag is
set regardless). A module latched this way permanently loses `partially_load`'s zero-cost fast
path. This is a candidate mechanism for the module-by-module lowvram loading regime observed only
under whole-suite runs, and for a sampler hitting a CPU-resident weight mid-run; neither link is
established. Worth a targeted reproduction before any fix is proposed upstream or patched around.

Both halves of the latch are pinned by
`tests/test_comfy_contract_drift.py::TestModelPatcherResidencyPins`: a partial unload against a
module whose `comfy_patched_weights` marks were removed (which is what another patcher's unpatch
does to the shared module) frees zero bytes and still sets `model_lowvram`, and a latched patcher
pays a `load()` walk on a `partially_load` that previously returned free. What remains open is only
the link from this mechanism to the two observed failures.

## Verification

- CPU tests cover the registry, the residue probe, the cache's serve-untouched behaviour, the
  statistics, and the lever (`tests/execution/`).
- GPU tests pin the poisoning gate (residue reported, lever restores pristine, interleaved LoRA job
  does not change an identical job's output) and the reclaim pairing (restore frees weights into the
  allocator; only the allocator release reaches the card; a restored component stays warm and the
  next job does not touch the disk).
- The worker-side residency reporting, eviction, restore handling, and statistics forwarding are
  covered by the worker repository's own suites.
