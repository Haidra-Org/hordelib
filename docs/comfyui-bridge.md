# The ComfyUI bridge

How hordelib embeds ComfyUI: which native ComfyUI surfaces it couples to, which behaviors it
deliberately overrides, and how each coupling is protected against drift when the pinned
ComfyUI version (see `hordelib/installation/manifest.json`) is bumped. The code lives in
`hordelib/comfy_horde.py` (the `Comfy_Horde` orchestrator) and `hordelib/execution/`
(the typed seams). Everything above this layer talks to the `ExecutionBackend` protocol
(`hordelib/execution/interface.py`) and never touches ComfyUI.

## The embedding shape

ComfyUI is designed to run behind its aiohttp `PromptServer`, but its executor only needs a
server-shaped object. hordelib runs the real `PromptExecutor` against
`HeadlessComfyServer` (`hordelib/execution/server_shim.py`), a named class exposing exactly
the surface the executor touches headless: `client_id` (assigned from `extra_data` each
run), `last_node_id` (written per node), `sockets_metadata` (read only when previews are
enabled), and `send_sync` (every execution event). That surface is pinned by a strict fake
in `tests/test_comfy_contract_drift.py` that raises on any attribute access outside it, so
growth in ComfyUI's server expectations is discovered as a named test failure.

A fresh `PromptExecutor` is built per run, on purpose: a persistent executor's node caches
would pin tensors in RAM/VRAM directly against the worker's aggressive unload policy, and
executor construction is cheap. A consequence worth knowing: cross-run node caching never
happens, so the `cache_type`/`cache_args` plumbing in `_get_executor` only configures
within-run caching.

## Event flow

During a run, ComfyUI pushes events (`execution_start`, `executing`, `executed`,
`progress_state`, `execution_error`, ...) into the shim's `send_sync`, which forwards them
to `Comfy_Horde.send_sync`. There they are parsed into typed models by
`hordelib.execution.comfy_events.parse_event` and dispatched on type: errors and interrupts
are logged with their full exception context, the raw event is forwarded to the embedder's
`comfyui_callback`. Every string key of a ComfyUI event payload lives only in
`comfy_events.py`; unknown labels or drifted payload shapes parse to `UnknownEvent` rather
than raising, and the drift tests assert every label the pinned ComfyUI actually emits
parses into a typed model.

## Output retrieval

Artifacts are not scraped from events. After `execute()` returns, the executor exposes
`success`, `status_messages`, and `history_result` (`{"outputs": {node_title: ui_dict},
"meta": ...}`); `Comfy_Horde._collect_run_result` turns those into a typed
`PipelineRunResult` (`hordelib/execution/results.py`). Collection walks every list-valued
ui key of each output node's ui dict, keeping entries that carry an in-memory `BytesIO`
(the contract hordelib output nodes implement, e.g. `node_image_output.py`), and tags each
with its source node title. Graphs are title-keyed (`ComfyGraph`), so the `history_result`
key is the node title declared in the pipeline's `OutputSpec`s; `run_pipeline` fails
loudly, naming the node and the typed error summary, when a declared output produced
nothing.

Two ComfyUI behaviors this relies on, both pinned by the drift tests: `history_result` is
assigned on the success and handled-error paths alike (only an exception escaping the
executor leaves it unset), and cached output nodes are delivered into it only when the
server has a `client_id`, which is why `extra_data` always carries one.

## Progress channels

Three channels exist, in priority order:

1. **The native global hook** (`comfy.utils.set_progress_bar_global_hook`, installed by
   `hordelib/execution/progress_hook.py`): the primary sampling-progress stream feeding the
   embedder's progress callback.
2. **`progress_state` events**: per-node pending/running/finished state for the whole
   graph, delivered for free because ComfyUI force-registers its `WebUIProgressHandler`
   against our server shim every run. Parsed by the typed event layer; useful for
   multi-stage graphs.
3. **The tqdm fallback**: the `OutputCollector` stdout parser, for processes without the
   native hook.

ComfyUI's newer `ProgressRegistry`/`ProgressHandler` API was evaluated and deliberately not
adopted: `reset_progress_state` discards all registered handlers inside every
`execute_async`, so persisting a handler would require a new monkeypatch. The global hook
is a public, stable seam. The drift tests pin both facts so the decision is revisited if
ComfyUI changes either.

## Per-job measured VRAM footprint

Every image job publishes what it actually occupied on the device, so a consumer can price a model
from measurement instead of from a registry constant. `hordelib.api.record_job_vram_footprint`
brackets both execution paths: the monolithic run in `HordeLib._run_materialized`, and the sampler
stage of a disaggregated job in `HordeLib.sample_stage`. It wraps `record_job_vram_profile` (which
wraps `load_models_gpu` for the component union and samples on-device weights and device-wide use on
a background thread) and hands the result to the metrics collector as
`JobPhaseMetrics.vram_footprint`, alongside the rest of the job's phase metrics. The sampling period
is `JOB_VRAM_FOOTPRINT_POLL_INTERVAL_S` (50 ms): every job pays it, so it is looser than the interval
a dedicated calibration run uses.

The record (`hordelib.metrics.JobVramFootprint`) carries four measurements:

- `peak_resident_weights_mb`: the largest the on-device weight set ever got at one instant. This is
  the co-residency figure, and it sits below the component sum whenever the backend evicted or
  block-swapped something mid-run.
- `peak_device_used_mb`: device-wide used VRAM at its high-water, weights plus transient activation.
- `sum_component_weights_mb`: the union of every component loaded during the run, the conservative
  upper bound.
- `resident_weights_after_job_mb`: what the next job inherits on the card (0 when the run returned it).

It also carries the keying context: model name, baseline, width, height, batch size, and `stage`
(`whole_job` or `sample_stage`). Occupancy is only comparable within that bucket; a sampler-only
stage loads the UNet alone and carries the whole sampling activation, so its numbers describe a
different thing from a whole job on the same model. Every field defaults to `None`, so a producer
that cannot measure (no ComfyUI, no device) and an older producer both still emit a valid record.

**A footprint is one observation, not a budget.** It carries no safety margin, no allowance for a
neighbouring process, and no smoothing across runs: it is what one job did on one card under one
driver state. A consumer turning footprints into a forecast has to bucket them, aggregate across
runs, and add its own headroom. hordelib publishes the measurement and changes no registry constant
in `hordelib/feature_impact.py` on the strength of it.

Separately, the native progress hook samples two VRAM high-waters per step:
`JobPhaseMetrics.vram_used_high_water_mb` (the process-local view, which counts the torch allocator's
reclaimable cache as free and therefore under-states device occupancy) and
`vram_device_used_high_water_mb` (the device-wide reading from `get_torch_device_free_vram_mb`). The
first is kept unchanged for consumers already calibrated against it; the second is the one to compare
against hardware capacity.

## Model directories

Model paths are registered through `folder_paths.add_model_folder_path` via the
`MODEL_CATEGORY_DIRS` table in `hordelib/execution/model_dirs.py`, which is the single
declaration of the horde-directory-to-ComfyUI-category mapping. Registration appends, so
ComfyUI's own default directories (empty in horde deployments) keep precedence. The one
direct registry touch left is setting the extension filter for categories hordelib itself
introduces (`facerestore_models`), because the setter API cannot; it is confined to
`model_dirs.py`. `invalidate_filename_cache` covers the mid-process rescan needed when
files (textual inversions) appear between jobs.

## Checkpoint RAM cache

`SharedModelManager._models_in_ram` is a content-addressed, MB-budgeted LRU
(`hordelib/execution/component_cache.py`, `ComponentCache`) of loaded component tuples. Each entry is keyed
by a `ComponentCacheKey(kind, identity)`:

- **Full or subset checkpoint** (`file_type is None`): `kind=CHECKPOINT`, `identity=<model name>`. Every
  full-or-subset load of a model shares one entry, and the identity is free to derive so a warm hit resolves
  no record or disk path. A cached tuple is only reused when it covers every component the current request
  asks for: a component-subset load (a text-encode stage loads only the CLIP, an image lane only the VAE)
  caches `None` in the omitted slots, and a later, broader request treats such an entry as a miss, reloading
  from disk and replacing the entry with the fuller tuple. Seamless-tiling state is re-applied on reuse to
  whichever of the UNet and VAE are present.
- **Bare component** (`file_type` in {unet, vae, text_encoder}, loaded via `comfy.sd.load_diffusion_model`):
  `kind` is the component's slot (UNET/CLIP/VAE), `identity=<model name>:<file_type>`.
- **Standalone VAE** (see below): `kind=VAE`, `identity=vae@<content-hash>`.

**Budget and eviction.** The budget is `HORDE_COMPONENT_CACHE_MB` megabytes, read once per process. A load
inserts its entry and the cache evicts the least-recently-used entries until the summed approximate RAM cost
fits the budget (the just-loaded entry is never evicted, so a single component larger than the budget still
loads). Each entry's cost is estimated from the checkpoint's component-identity sidecar (the UNet-ish
residual for the model slot, the text-encoder bytes for the CLIP slot, the VAE bytes for the VAE slot), or a
conservative per-kind constant when no sidecar is available; the estimate bounds intent, it is not a measured
resident-set delta. Evictions, hits, misses, and resident megabytes are recorded per job on the metrics
collector (`component_cache_*` fields on `JobPhaseMetrics`).

**Default (`0`) is the rollback lever.** With the budget unset or `0`, the cache holds exactly one component:
each insert evicts every other entry, reproducing the historical single-slot behaviour, so residency changes
only when a deployment opts in with a positive budget.

**LoRA serving.** Every entry is shared with later jobs, including LoRA-bearing ones. The graph's LoRA
loader (`comfy.sd.load_lora_for_models`) clones the base ModelPatcher/CLIP before patching, but a clone
shares the underlying module and patch backup with the base, so a LoRA job's patches do reach the cached
weights while its clone is loaded. Sharing is safe anyway because ComfyUI restores lazily at the
component's next load: a patcher whose patch set differs from the one the module's weights currently hold
triggers a full unpatch before the load proceeds, so a component always reaches a job carrying that job's
own patches. Residency reporting exposes whether a resident entry currently carries another patcher's
weights (`HeldComponentSnapshot.mutated`), and `hordelib.api.restore_components` returns named entries to
their loaded state on demand (see `docs/plans/component-restore-contract.md`).

### Standalone-VAE path (content-addressed VAE sharing)

A VAE-only decode of a monolithic checkpoint (`output_vae` set, `output_model`/`output_clip` clear,
`file_type is None`) otherwise subset-loads the whole multi-gigabyte checkpoint and caches the VAE under
the *model* name, so two models that embed byte-identical VAE weights never share a cached VAE. When
horde_model_reference has written a component-identity sidecar beside the checkpoint (see below), the loader
instead loads the small pre-extracted standalone VAE and caches it under `vae@<content-hash>`, so those two
models resolve to one cache entry (a cross-model cache hit). The result tuple mirrors the subset path's
`(model, clip, vae, clipvision)` shape with only the VAE populated, and seamless tiling is applied exactly as
on the subset path. Any absence (no fresh sidecar, no extracted file) falls back to the unchanged subset
load; non-VAE-only requests are untouched.

The sidecar (and the extracted `vae/<...>.safetensors`) is written by `CompVisModelManager` after a
successful checkpoint download, and can be (re)built for every on-disk image checkpoint via
`CompVisModelManager.ensure_component_identity_sweep()` (idempotent; never run automatically at init, so a
child process pays no boot-time hashing cost). Extraction targets the same `vae/` folder ComfyUI searches
(`hordelib/execution/model_dirs.py`), so the loader finds the file with no extra wiring.

Set `HORDE_DISABLE_STANDALONE_VAE_PATH` truthy (`1`/`true`/`yes`/`on`) to disable the path entirely and
restore the prior subset-load behaviour for every VAE-only request.

### Weight page-in and CPU-side weight retention

The zero-copy loader leaves a checkpoint's weights as views over the memory-mapped file, so their bytes are
read from disk only when first touched, which is ComfyUI's `.to(device)` inside the model load: a page-fault
driven read at a fraction of the disk's rate, sitting inside the sampling window. `hordelib/execution/
weight_prefetch.py` moves that read off the critical path: after a cold checkpoint load
(`node_model_loader`) and again before the sampling lease is acquired (`sampling_lease`), a daemon thread
asks the OS to prefetch the module's CPU-resident weight ranges (`PrefetchVirtualMemory` on Windows,
`madvise(MADV_WILLNEED)` elsewhere) and then touches one byte per page so the pages join the process's
working set (a prefetch alone leaves them on the standby list, and the copy still soft-faults them in at a
fraction of memory speed). Kill switch: `HORDELIB_DISABLE_WEIGHT_PREFETCH=1`.

The matching unload path (`hordelib/execution/cpu_weight_retention.py`, hooked into `ModelPatcher.load` and
`ModelPatcher.unpatch_model`) records each CPU tensor before the load replaces it and points the parameter
back at that tensor at unload, so ComfyUI's device-to-host copy-back (over a second per UNet on every
eviction, and a private copy of the weights per process) becomes a no-op; the retained tensor is byte-identical
to what the copy would have produced. LoRA-patched keys (the patcher's own `backup`) are left to ComfyUI.
Kill switch: `HORDELIB_DISABLE_CPU_WEIGHT_RETENTION=1`.

## The monkeypatches

Seven ComfyUI internals are patched at import time (`hordelib/execution/comfy_patches.py`),
all policy injections with no native hook:

- `load_models_gpu` and `ModelPatcher.load`: force full GPU loads (with VRAM-overflow and
  model-class guards) so sibling worker processes sharing a GPU behave predictably. When the
  overflow guard declines the full load, ComfyUI sizes a partial (low-VRAM) load instead, and the
  `ModelPatcher.load` hijack lets that stand for those patchers (`_partial_load_allowed`) rather
  than forcing the whole weight set into free VRAM the guard just found too small. Small
  support-model loads (VAEs) additionally have their caller-supplied working-memory estimate
  clamped *and* the free-memory target capped near their own weights for the duration of the
  load: ComfyUI otherwise frees the worst-case decode estimate (or, failing that, its ~1GB
  inference-reserve floor) up front, evicting a co-resident diffusion model (a multi-second
  PCIe round-trip each way, every job) to host a few hundred MB of autoencoder, when a genuine
  shortfall would only mean a tiled decode. Eviction remains possible when free VRAM cannot
  host even the support weights themselves.
- `ModelPatcher.unpatch_model`: restore the recorded CPU tensors instead of copying weights back from the
  device (see the section above).
- `text_encoder_initial_device`: load text encoders on CPU first.
- `comfy.lora.calculate_weight`: repair malformed "diff" patch tuples.
- `IsChangedCache.get`: prompt-change logging.
- `comfy.samplers.ksampler`: bound the adaptive sampler and apply per-run solver options (below).

ComfyUI's native `force_full_load=` parameter cannot replace the first two: comfy's
internal call sites would never pass horde's policy. Each patch stores its original,
supports temporary-state swaps, and is guarded by `assert_force_load_class_names_exist`
plus the signature pins in `tests/test_comfy_contract_drift.py`.

### Per-run free-VRAM clamp

ComfyUI sizes every free by the shortfall it computes from `get_free_memory`, which reads the CUDA
driver's process-local view. Under WDDM that view reports memory sibling processes hold as free, so the
shortfall comes out too small and the load overcommits the card. A caller that measures free VRAM at the
device level (the worker parent, which holds the NVML figure) passes it as `device_free_truth_mb` on
`run_pipeline`/`basic_inference`/`generate`, and on the disaggregated `sample_stage` and `decode_stage`
(ComfyUI sizes each VAE decode pass by the free VRAM it reads, so an unclamped batch decode on the image
lane allocates its whole activation set at once); for the duration of that run
`comfy_patches.free_memory_view_clamped` interposes `get_free_memory` so the reported free total is the
lower of ComfyUI's own answer and a ceiling of what the process can obtain without evicting anything: the
supplied figure, less this process's allocator growth since the run started, plus the memory free inside
torch's own reserved pool. Own growth and the reclaimable pool are both honest under WDDM; sibling growth
is bounded by the caller's dispatch admission. The reclaimable term is not optional: ComfyUI's own total
is defined the same way (device free plus `reserved - active`), and after sampling that cache is often
gigabytes, so a ceiling without it would invent a shortfall at the decode-time VAE load and evict the
resident diffusion model mid-job.
The swap is a plain local capture/restore (like the support-model free cap above), so it is invisible to
the monkeypatch registry and cannot double-restore against it. With no figure supplied, or on a non-CUDA
device, nothing is interposed.

### Retention grants, and reporting when the device did not honour one

A host that knows the next job reuses the model passes `defer_vram_unload=True`, which suppresses the
bridge's own end-of-run eviction so the weights survive the job boundary and the next job skips the
host-to-device upload. The grant is a *prediction*, not a guarantee: ComfyUI satisfies an allocation by
freeing memory on the device, and the `load_models_gpu` patch above falls back to an unbounded
requirement, so a single load inside the run can unload every other model on the card, including the
checkpoint the grant covers. Nothing outside the process can see that happen, and a host that keeps
believing its own prediction charges VRAM to weights the card does not hold and routes later jobs to a
process that has to reload them anyway.

So the bridge reports it. At the end of a run that was granted the deferral,
`comfy_horde.device_holds_no_loaded_model()` reads `current_loaded_models` for the inference device; an
empty device means the grant kept nothing. The verdict rides `OutputArtifact.metadata` under
`RETAINED_WEIGHTS_EVICTED_METADATA_KEY` (the same channel the sampler-truncation record uses, and for the
same reason: it is a property of the whole run) and lands on `ResultingImageReturn.retained_weights_evicted`
and `SampleStageResult.retained_weights_evicted`, so the monolithic and disaggregated paths disclose it
identically. It is always `False` for a run that asked for no deferral, whose own evictor empties the
device by design. An entry whose device cannot be read counts as being on this one, so an unrecognised
ComfyUI build reports no eviction rather than a false one.

### A full unload reports what it actually freed

`unload_all_models_vram()` (and `ExecutionBackend.free_vram()` through it) returns a `VramUnloadResult`
rather than `None`. The free is a request: ComfyUI walks `current_loaded_models` and drops what it can,
skipping any entry a live reference still pins, and it tells the caller nothing about what it skipped. A
host that reports the weights moved to host RAM on the strength of having asked then carries gigabytes the
card is holding but its ledger is not, and every later admission decision is made against room that does not
exist.

The result carries `freed_mb` (the gain between the device-free readings taken either side of the free, a
floor rather than an exact figure since the reading is process-local), `remaining_loaded_models`,
`remaining_loaded_weights_mb` (`None` when the figure cannot be read, which a caller must not treat as zero),
and `dead_model_refs_dropped`. Its `complete` property is the verdict: nothing still listed, or nothing above
`_UNLOAD_REMAINDER_FLOOR_MB` still resident.

Dead references are dropped here. A `LoadedModel` whose patcher weakref has been collected raises on every
accessor (`'NoneType' object has no attribute 'model_size'`), pins nothing, and can never be unloaded, so
leaving it in the list would keep the loaded count non-zero forever and make every later unload read as
incomplete. Removing it is the same thing ComfyUI's own `cleanup_models` does for its dead references. The
existing log lines (`comfy.models_after_unload`, the free-RAM traces) are unchanged; the result is additive.

## The bounded adaptive sampler

`dpm_adaptive` is the one sampler whose iteration count is chosen by the solver rather than by
the schedule: `DPMSolver.dpm_solver_adaptive` is a `while` loop under a PID step-size
controller, and the nominal steps only set the sigma range it integrates over. On some model
and payload combinations the controller never reaches its tolerance and the loop does not
terminate, while the sample is already essentially converged.

`hordelib/execution/adaptive_sampler_bound.py` bounds it at
`ADAPTIVE_ITERATION_BUDGET_MULTIPLIER` (1.25) times the nominal step count and delivers the
best-effort sample. Iterations past the schedule are tolerance polish with approximately no
marginal quality, so a looser bound pays multiples of the advertised GPU cost for nothing.
The multiplier is derived from the SDK policy, and `hordelib.api` exports
`SAMPLER_EXECUTION_CONTRACT_VERSION` for bridges to place directly in their worker check-in. The
contract version is fixed deliberately rather than following the SDK's latest version automatically:
a dependency update must never claim backend behavior that hordelib has not implemented and tested.

The seam is `comfy.samplers.ksampler` rather than `sample_dpm_adaptive`, because the factory is
where the sampler function is built as a closure over the full `sigmas` tensor; the replacement
reads `len(sigmas) - 1` from its own arguments, so no state has to be threaded through globals
or context. Below the bound nothing is reimplemented: the replacement constructs the stock
`DPMSolver`, runs the stock `dpm_solver_adaptive`, and only chains a check onto the solver's
info callback, which unwinds the loop with the working `x` at the bound.

A truncated run is recorded as a `SamplerTruncation` (sampler, nominal steps, iterations, the
multiplier, `capped`). The record is scoped to one backend `run_pipeline` call, rides
`OutputArtifact.metadata`, and lands on `ResultingImageReturn.sampler_truncation` for the
consumer to disclose. `faults` is a fixed enum schema and cannot express the counts, which is
why the record has its own typed field. The disaggregated `sample_stage` carries it across the
lane split the same way: it returns a `SampleStageResult` (the LATENT plus the record read off
the stage run's artifact), because the sample and the decode that produces the image run in
different processes and a bare `bytes` return would drop the record at the split.

## Solver options

The stock `KSampler` node exposes the sampler name and the schedule, and nothing else. The
tuning arguments the sampler functions themselves take (`eta`, `s_noise`, `s_churn` with its
`s_tmin`/`s_tmax` sigma window, `solver_type`, and the multistep `order`/`max_order`) have no
graph inputs, so they can only be reached where the sampler object is built. That is the same
`comfy.samplers.ksampler(name, extra_options)` call upstream's dedicated per-sampler nodes make;
`SamplerCustom` only runs a `SAMPLER` it is handed.

`hordelib/pipeline/payload.py` maps the payload's `sampler_*` fields onto the upstream argument
names, `hordelib/execution/sampler_options.py` holds them for the duration of one run, and the
`ksampler` hijack merges them into the options the factory receives. A sampler is only handed
arguments its own function signature declares, because an unaccepted keyword raises `TypeError`
from inside graph execution rather than at the call site. `option_bounds` is the one place
per-sampler ranges are consulted; the values a sampler cannot use are held inside its range
instead of dropped, so the request keeps the control it asked for. Every field defaults to
unset, and an unset run builds samplers exactly as before.

## Sigma schedules with no name

`comfy.samplers.calculate_sigmas` resolves a schedule name through `SCHEDULER_HANDLERS`, which holds
the schedules computable from the model's sigma range. Align Your Steps and GITS are not in it: both
are measured tables rather than functions, and upstream exposes each as a node emitting SIGMAS, which
only the custom-sampler graph shape consumes. The graphs here name their schedule on a `KSampler` or
`BasicScheduler` input instead.

They are therefore carried beside the graph. `hordelib/execution/sigma_schedules.py` holds the
requested schedule for one run, `comfy.samplers.calculate_sigmas` is patched to supply it, and the
graph's own scheduler input carries `SIGMA_GENERATOR_GRAPH_SCHEDULE` because an input outside the
node's declared list fails prompt validation for the whole graph. The patched function is the seam
because both graph shapes reach it: `KSampler` calls it as a module global, `BasicScheduler` as a
module attribute. The tables and the interpolation are imported from the pinned `comfy_extras`
modules rather than copied, so a schedule computed here is the one its node would have produced.

Align Your Steps noise levels were measured per model family, so the request carries the family
resolved from the model's baseline (`ALIGN_YOUR_STEPS_MODEL_TYPES`). A baseline with no published
levels is refused rather than served another family's schedule. GITS tables are keyed by coefficient
and hold for any model, so it needs no family.

## Flow-model shift

Flow-matching models take a timestep shift that moves where the sampler spends its steps. The qwen
graph already carries `ModelSamplingAuraFlow`, so the payload's `flow_shift` sets that node's input;
flux graphs get a `ModelSamplingFlux` inserted between the model (or the last LoRA) and the sampling
nodes, with `base_shift` and `max_shift` set equal so the node's area interpolation is constant and
the requested value is the shift the model runs with. Unset leaves every graph exactly as it was.

## Custom nodes: classic and V3

The existing horde nodes (`hordelib/nodes/`) are classic-API (`NODE_CLASS_MAPPINGS` +
`INPUT_TYPES`), which remains a first-class path in the pinned ComfyUI; they are not being
migrated. **New nodes are written against the typed V3 API** (`comfy_api.latest`
`ComfyExtension`/`comfy_entrypoint`, typed sockets including `Audio`, `Video`, `SVG`,
`Voxel`), which is the intended path for new modalities. `node_v3_canary.py` is the living
proof: it registers headless, executes, and returns BytesIO ui entries through the same
collection path, verified by the drift tests. See `docs/modality-readiness.md` for the
end-to-end modality recipe.

## Drift protection summary

`tests/test_comfy_contract_drift.py` runs a real CPU-only execution round trip (no GPU, no
models) against the pinned ComfyUI and pins: the `validate_prompt` 4-tuple, the executor
result attributes, the event label set (and that every emitted event parses typed), the
error payload shape, the headless server surface, the progress-hook and registry
lifecycles, the monkeypatched signatures, the folder_paths surface, and the V3 canary.
`tests/test_node_schema_freshness.py` separately pins the node input schemas pipelines bind
against. A ComfyUI version bump that breaks any bridge assumption fails there first, by
name, instead of deep inside a GPU run.
