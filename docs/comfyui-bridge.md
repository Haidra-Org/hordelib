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

## Model directories

Model paths are registered through `folder_paths.add_model_folder_path` via the
`MODEL_CATEGORY_DIRS` table in `hordelib/execution/model_dirs.py`, which is the single
declaration of the horde-directory-to-ComfyUI-category mapping. Registration appends, so
ComfyUI's own default directories (empty in horde deployments) keep precedence. The one
direct registry touch left is setting the extension filter for categories hordelib itself
introduces (`facerestore_models`), because the setter API cannot; it is confined to
`model_dirs.py`. `invalidate_filename_cache` covers the mid-process rescan needed when
files (textual inversions) appear between jobs.

## The monkeypatches

Five ComfyUI internals are patched at import time (`hordelib/execution/comfy_patches.py`),
all policy injections with no native hook:

- `load_models_gpu` and `ModelPatcher.load`: force full GPU loads (with VRAM-overflow and
  model-class guards) so sibling worker processes sharing a GPU behave predictably. Small
  support-model loads (VAEs) additionally have their caller-supplied working-memory estimate
  clamped: ComfyUI otherwise frees the full worst-case decode estimate up front, evicting a
  co-resident diffusion model (a multi-second PCIe round-trip each way, every job) to host a
  few hundred MB of autoencoder, when a genuine shortfall would only mean a tiled decode.
- `text_encoder_initial_device`: load text encoders on CPU first.
- `comfy.lora.calculate_weight`: repair malformed "diff" patch tuples.
- `IsChangedCache.get`: prompt-change logging.

ComfyUI's native `force_full_load=` parameter cannot replace the first two: comfy's
internal call sites would never pass horde's policy. Each patch stores its original,
supports temporary-state swaps, and is guarded by `assert_force_load_class_names_exist`
plus the signature pins in `tests/test_comfy_contract_drift.py`.

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
