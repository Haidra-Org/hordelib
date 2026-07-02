# Adding a pipeline

How to add a new ComfyUI pipeline to hordelib. You do not need to understand hordelib's
internals: a pipeline is one graph JSON plus one Python definition object, and the test suite
tells you immediately when either is wrong.

A runnable reference implementation lives in `tests/pipeline/examples/`
(`example_pipeline.json` + `example_definition.py`); `tests/pipeline/test_example_pipeline.py`
exercises it end to end. Edit it freely to experiment.

## The mental model

- **Graph**: a pure ComfyUI API export in `hordelib/pipelines/pipeline_<name>.json`. Design
  and test it in the ComfyUI GUI with standard nodes; hordelib swaps in its own node
  implementations at load time (`HORDE_NODE_REPLACEMENTS` in `hordelib/pipeline/graph.py`).
- **Node titles are the parameter namespace**: a KSampler titled `sampler` is addressed as
  `sampler.steps`. Titles must be unique and stable; duplicate titles are rejected at
  registration.
- **`PipelineDefinition`** (`hordelib/pipeline/definition.py`): the single object a pipeline
  author writes. It colocates the graph reference, the exposed parameters (bindings), the
  selection declaration, the patch steps, and the outputs.
- **Bindings are the user-facing surface**: every way a payload (prompt, steps, cfg, ...) can
  influence the graph is declared node-first through a `node()` handle:
  `node("sampler", "KSampler").bind(steps="ddim_steps", cfg="cfg_scale")`. The handle names
  the graph node and its expected class; `bind` keywords are that node's real input names,
  and values are payload field names, transforms, or `scaled(field, multiplier)`.
- **Everything is audited at registration**: the node title must exist in the graph, the
  node must have the declared class, the input name must be one the class actually declares
  (checked against the committed ComfyUI schema snapshot,
  `hordelib/pipeline/comfy_node_inputs.json`), and the payload field must exist on the
  family's payload model. This matters because ComfyUI itself *silently drops* unknown input
  names: without the audit, a typo like `denosie=` would ship and quietly produce
  default-denoise images forever.
- **Patch steps** handle what bindings cannot: resolved facts (model files on disk,
  downloaded LoRAs) and structural surgery (chain insertion, rewires).

## Walkthrough: the reference definition

From `tests/pipeline/examples/example_definition.py` (a real image pipeline looks the same,
with `graph_file=pipeline_graph("<name>")` pointing at `hordelib/pipelines/`):

```python
EXAMPLE_PIPELINE = PipelineDefinition(
    name="example_txt2img",
    graph_file=EXAMPLE_GRAPH,
    selector=ImageSelector(tier=SelectionTier.FALLBACK),
    bindings=bindings.compose(
        node("sampler", "KSampler").bind(     # this node, with these real inputs:
            sampler_name=bindings.comfy_sampler,  # a transform computes the value
            cfg="cfg_scale",                      # a payload field is copied
            denoise="denoising_strength",
            seed="seed",
            scheduler="scheduler",
            steps="ddim_steps",
        ),
        bindings.EMPTY_LATENT,   # reusable groups: empty_latent_image.{width,height,batch_size}
        bindings.PROMPTS,        # prompt.text, negative_prompt.text
    ),
    outputs=(OutputSpec(node="output_image"),),
    patch_steps=(steps.apply_main_model,),
)
```

Field by field:

- `selector` — when this pipeline is chosen. `SelectionTier` names the precedence level
  (workflow override > baseline family > feature > painting > generic variant > fallback);
  `order` breaks ties within a tier. Criteria are typed fields (`baselines=`, `workflow=`)
  and named payload features (`features=(features.CONTROLNET,)`) from
  `hordelib/pipeline/families/image_gen/features.py`. Add a new `PayloadFeature` there if
  your condition doesn't exist yet; avoid `extra_predicate` (the contract tests flag it).
- `bindings` — authored with `node()` handles and/or composed from the named groups in
  `hordelib/pipeline/families/image_gen/bindings.py` (which are themselves `node()`-authored).
  Compose exactly what your graph has: a group targeting a node your graph lacks, a wrong
  node class, an input name the class does not declare, or a payload field that does not
  exist all fail registration with the offender named. Use `scaled("ddim_steps", 0.33)` to
  bind a field with a constant multiplier.
- `outputs` — the node(s) results are collected from. The node must exist and be an output
  class for its kind (`HordeImageOutput` for images); execution fails loudly, naming the
  node, if a declared output produces nothing. Title the node `output_image` in new exports.
- `patch_steps` — opt in to what you need from
  `hordelib/pipeline/families/image_gen/steps.py` (each step is payload/context-gated, so it
  is a no-op when it does not apply). At minimum you almost always want
  `steps.apply_main_model`.

## Checklist for a real image pipeline

1. Export the graph to `hordelib/pipelines/pipeline_<name>.json` with unique, descriptive
   node titles (`model_loader`, `sampler`, `prompt`, `output_image`, ...). If the graph uses
   a node class no packaged graph used before, regenerate the node schema snapshot once:
   `uv run --no-sync python -m hordelib.pipeline.node_schemas` (needs an initialisable
   ComfyUI; the audit will tell you when this is required).
2. Add `hordelib/pipeline/families/image_gen/<name>.py` with the `PipelineDefinition`
   (pattern-match `creative_upscale.py` for a workflow with node()-authored bindings and
   opt-in patch steps, `qwen.py` for a simple baseline, `controlnet.py` for a feature
   variant group).
3. List the definition in `IMAGE_PIPELINES` in
   `hordelib/pipeline/families/image_gen/__init__.py` (there is deliberately no discovery
   magic).
4. Add a matching member to the public `ImagePipeline` enum in
   `hordelib/pipeline/identifiers.py` (the vocabulary callers use to select a pipeline
   explicitly via `HordeLib.generate`). An import-time audit in the family package, mirrored
   by `tests/pipeline/test_pipeline_identifiers.py`, fails loudly if the enum and
   `IMAGE_PIPELINES` drift.
5. Run `pytest tests/pipeline` — the contract tests (`test_pipeline_contracts.py`)
   automatically cover the new definition: graph loads, audit is clean, strict
   materialization works. Fix whatever they name.
6. If your pipeline's materialized graph should be pinned (it should), add canned cases to
   `tests/pipeline/test_materialize_snapshots.py` and generate with `--snapshot-update`.
7. Add a GPU oracle test (`tests/test_horde_inference_<name>.py`) with reference outputs in
   `images_expected/`.

If the pipeline serves a **new model baseline**, also follow
[adding-a-baseline.md](adding-a-baseline.md) (model reference, `BaselineProfile`, coverage
tripwire).

## New payload parameters

If users need a knob that does not exist yet (a new field like `prompt` or `steps`), add it
to `ImageGenPayload` in `hordelib/pipeline/payload.py` with a clamping validator (Horde
semantics: bad input never raises; it is coerced, clamped, or defaulted), then bind it.

## New families/modalities

A family = its own payload model + selection context + registry. The machinery is generic:
see `hordelib/pipeline/families/post_processing.py` for a complete second family in ~90
lines, and `tests/pipeline/test_registry_generics.py` for a toy family showing a custom
`Selector` subclass. New output modalities add an `OutputKind` member, a Horde output node
class registered in `HORDE_OUTPUT_CLASS_TYPES` (`hordelib/pipeline/graph.py`), and flow
through the execution layer unchanged.
