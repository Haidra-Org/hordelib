# Adding a new model baseline

The checklist for supporting a new image-generation architecture ("baseline") end to end.
The mechanics of authoring the pipeline itself are in
[adding-a-pipeline.md](adding-a-pipeline.md); this page covers the baseline-specific extras.

The CI tripwire `tests/pipeline/test_baseline_coverage.py` fails the moment a new
`KNOWN_IMAGE_GENERATION_BASELINE` enum value appears in horde_model_reference, and stays
red until the baseline is consciously classified — either by completing this checklist or
by adding it to the test's `UNSUPPORTED_BASELINES` set.

## Where baseline knowledge is allowed to live

Baseline-conditional logic must use `KNOWN_IMAGE_GENERATION_BASELINE` enum members — never
raw strings — and is confined to:

- `hordelib/pipeline/families/image_gen/baselines.py` — the `BaselineProfile` table (loader
  kind, force-load policy, CLIP type, flow shift), the per-model `IMAGE_MODEL_OVERRIDES`, and
  the named baseline groupings (`FLUX_BASELINES`, ...)
- `hordelib/pipeline/families/image_gen/` — per-pipeline selectors and patch steps
- `hordelib/pipeline/context.py` — the `ModelContext` data structure (its `baseline` field is
  the typed input every selector reads; no conditional logic lives here)

`ModelContext` carries no per-baseline boolean properties. Every pipeline selects by
declaring a baseline set on its `ImageSelector` (`baselines=frozenset({...})`); a family that
spans several baselines (as flux does) declares a named frozenset grouping in `baselines.py`
and reuses it.

## One graph per family

A family's graph is shared by every model of that baseline, so anything that varies between
its members is a variable rather than a second graph:

- **Component files.** A split-files graph's `clip_loader.clip_name` and `vae_loader.vae_name`
  are patched from the resolved record's own component files (`apply_component_loaders` in
  `steps.py`, reading `ModelContext.extra_files` by `file_purpose`). The filenames in the
  committed graph JSON are defaults that document the family; they are not what runs.
- **CLIP type.** `clip_loader.type` comes from `resolve_clip_type`: the baseline's
  `BaselineProfile.clip_type`, overridden per model by `IMAGE_MODEL_OVERRIDES`. The component
  lane loads a standalone text encoder through the same resolver, so a shared encoder module is
  wrapped exactly as the pipeline would have wrapped it.
- **Flow shift.** The shift node is inserted on demand (`apply_flow_shift`), never carried in
  the graph: `resolve_flow_shift` names the node the baseline uses (`ModelSamplingAuraFlow` for
  qwen, `ModelSamplingFlux` for flux) and the shift to apply, which is the payload's
  `flow_shift` or the baseline's default. A model that takes no shift (Krea 2 Turbo) declares
  `applies_flow_shift=False` in its override, and a requested shift is warned about and ignored.

A model needs its own pipeline only when its graph *shape* differs; differing components,
CLIP type or shift do not qualify.

## Checklist

1. **Model reference**: the baseline exists as a `KNOWN_IMAGE_GENERATION_BASELINE` member
   and the models are present in `stable_diffusion.json` (horde_model_reference).
2. **Pipeline**: follow [adding-a-pipeline.md](adding-a-pipeline.md) — graph export, a
   definition module in `families/image_gen/` selecting on the baseline at
   `SelectionTier.BASELINE_FAMILY`, and a spot in `IMAGE_PIPELINES`.
3. **Baseline profile**: if the baseline loads split files (bare diffusion model + separate
   CLIP/VAE), needs a flow-matching shift, or its models must not be force-loaded into VRAM on
   weaker cards, add a `BaselineProfile` row in
   `hordelib/pipeline/families/image_gen/baselines.py` (`loader=LoaderKind.UNET`, `clip_type=`,
   `flow_shift_node=`/`default_flow_shift=`, and/or
   `force_load_skip_classes=("<comfy model_base class>",)`)
   and mirror any new comfy class names in `FORCE_LOAD_SKIP_CLASS_NAMES`
   (`hordelib/execution/comfy_patches.py` — kept horde_model_reference-free on purpose; the
   startup tripwire fails if the two disagree). Consumers then pass the enum member to
   `initialise(models_not_to_force_load=...)`.
4. **Classify in the tripwire**: move the baseline from `UNSUPPORTED_BASELINES` to
   `BASELINE_EXPECTED_TEMPLATE` in `tests/pipeline/test_baseline_coverage.py`, mapping it
   to the new pipeline's name.
5. **GPU oracle**: add an inference test (`tests/test_horde_inference_<name>.py`) with
   reference outputs in `images_expected/`.
6. **Worker**: zero changes required. The worker resolves models via the reference and
   passes baseline enums through `hordelib.api`; new baselines flow through automatically
   once the worker's model list includes them.
