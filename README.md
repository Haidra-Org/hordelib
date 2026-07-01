# horde-engine

[![PyPI Version][pypi-image]][pypi-url]
[![Downloads][downloads-image]][downloads-url]
![GitHub license][license-url]

[![Build][build-image]][build-url]
[![Test Images][main-test-image]][main-test-url]
[![Test Images][pr-test-image]][pr-test-url]
[![All Models][all-model-images]][all-model-url]
[![Release Changelog][changelog-image]][changelog-url]

`horde-engine` wraps a pinned commit of [ComfyUI](https://github.com/comfyanonymous/ComfyUI) so the [AI Horde](https://aihorde.net/) can run ComfyUI-designed inference pipelines programmatically. It is the default inference backend for AI Horde since v1.0.0.

Python `>=3.12`. GPU/CUDA required. The package name on PyPI is `horde-engine`; the Python module is `hordelib`.

Community: [AI Horde Discord](https://discord.gg/3DxrhksKzn).

---

## Quick start

1. Install `horde-engine` together with a PyTorch build for your hardware (see
   [Choosing a PyTorch build](#choosing-a-pytorch-build)). For NVIDIA:
   ```
   # CUDA 13+ driver -> torch 2.12.0 (the default line):
   pip install --extra-index-url https://download.pytorch.org/whl/cu130 horde-engine
   # CUDA 12.x driver -> the newest CUDA 12.8 build is torch 2.11.0:
   pip install --extra-index-url https://download.pytorch.org/whl/cu128 horde-engine "torch==2.11.0"
   ```

2. Set `AIWORKER_CACHE_HOME` to a model directory (see [Model directory layout](#model-directory-layout)).

3. Call `hordelib.initialise()` once per process, then use the public API:
   ```python
   import hordelib
   hordelib.initialise()

   from hordelib.api import HordeLib, ImageGenPayload

   horde = HordeLib()
   payload = ImageGenPayload(
       prompt="an ancient llamia monster",
       model="Deliberate",
       sampler_name="k_dpmpp_2m",
       cfg_scale=7.5,
       ddim_steps=25,
       width=512,
       height=512,
       seed=123456789,
   )
   result = horde.basic_inference_single_image(payload)
   result.image.save("test.png")
   ```

`hordelib.initialise()` wipes `sys.argv` -- parse command-line arguments first. It also forbids dev-mode paths containing spaces (do not clone the repo to a path with spaces) and raises if you try to import any ComfyUI-touching code before calling it.

For more usage patterns, see the [public API docs](docs/public_api.md) and the example scripts in `examples/`. For complete API surface reference, consult the test suite under `tests/`.

---

## Installation

Requires an NVIDIA GPU with CUDA. On Linux, install the [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) first. Systems with 16 GB RAM or less need swap space and may need an enlarged `/tmp`:
```bash
sudo mount -o remount,size=16G /tmp
```

### From PyPI

```
# CUDA 13+ driver -> torch 2.12.0 (the default line):
pip install --extra-index-url https://download.pytorch.org/whl/cu130 horde-engine
# CUDA 12.x driver -> the newest CUDA 12.8 build is torch 2.11.0:
pip install --extra-index-url https://download.pytorch.org/whl/cu128 horde-engine "torch==2.11.0"
```

### From source (development)

```bash
git clone https://github.com/Haidra-Org/hordelib.git
cd hordelib
uv sync   # installs the latest torch (2.12.0, the CUDA 12.6 build) -- see "Choosing a PyTorch build"
```

### Choosing a PyTorch build

`horde-engine` does not pin `torch`: it declares `torchvision` as a version *range*, and torchvision
pins the exact matching torch in its metadata, so installing the library pulls the latest supported
torch (**2.12.0** -> torchvision 0.27.0). You only pick the *build* (which CUDA/ROCm wheels) for your
hardware. `torch` and `torchvision` are both routed per-build in `[tool.uv.sources]` so they always
share one CUDA build (a torchvision left on generic PyPI would pull a different CUDA build and break the
install).

`torchaudio` is **not** installed by default: it is the only package in the trio with no `+cu132` wheel,
and audio generation is currently unsupported. ComfyUI's eager `import torchaudio` is satisfied by a
lazy stub (`hordelib.utils.torch_build`) when it is absent; audio operations raise a clear error only if
actually used. Install a matching torchaudio ad-hoc if you need audio (cu126/cu130/cpu only).

For development, `uv sync --extra cu126` installs the **CUDA 12.6** build of torch 2.12.0. It runs on any
CUDA 12.6+ driver and, via NVIDIA driver backward-compatibility, on CUDA 13 drivers too -- the broadest
single build. Pick a different build with the matching extra:

```bash
uv sync --extra cu132   # exact CUDA 13.2 build
uv sync --extra cu130   # CUDA 13.0/13.1 build (also runs on a 13.2 driver)
uv sync --extra cu126   # CUDA 12.6+ build (also runs on CUDA 13 drivers)
uv sync --extra cpu     # no GPU; ~100x slower, testing only
```

With `pip`, select the index with `--extra-index-url` (install `torch` **and** `torchvision` from the
same index):

| Hardware | index | Notes |
|---|---|---|
| NVIDIA, CUDA 13.2 (exact) | `cu132` | torch 2.12.0; no torchaudio wheel (audio unavailable on this build). |
| NVIDIA, CUDA 13.0/13.1 driver | `cu130` | torch 2.12.0; also runs on a 13.2 driver. |
| NVIDIA, CUDA 12.6+ driver | `cu126` | torch 2.12.0; the only CUDA-12 build of 2.12.0. |
| NVIDIA, CUDA 12.x, older torch | `cu128` | No 2.12.0 wheel; caps at torch **2.11.0** (add `"torch==2.11.0"`). |
| AMD (Linux only) | `rocm6.4` | torch **2.9.1 only** (see note below). |
| No GPU | `cpu` | ~100x slower; testing only. |

> **ROCm:** only torch 2.9.1 (ROCm 6.4) is currently installable. PyTorch publishes torch 2.10-2.12 on
> rocm7.x but not the matching `pytorch-triton-rocm` (3.6.0/3.7.0) those wheels hard-depend on, so they
> cannot be resolved yet. Install it ad-hoc: `uv pip install torch==2.9.1 torchvision
> pytorch-triton-rocm --extra-index-url https://download.pytorch.org/whl/rocm6.4`.

---

## Architecture (v3)

The v3 refactor replaced the legacy dict-heavy pipeline machinery with a typed, testable, registry-driven layer. The key pieces:

| Layer | Location | Role |
|---|---|---|
| **Public API** | `hordelib/api.py` | Declared consumer surface. Workers import from here only. |
| **Bootstrap** | `hordelib/initialisation.py` | `initialise()` -- manifest installer, path patching, ComfyUI import, monkeypatches. |
| **Typed pipeline** | `hordelib/pipeline/` | `ImageGenPayload` (pydantic, clamp-dont-reject), `PipelineDefinition`/`ParamBinding`/`Selector`, tier-ordered `PipelineRegistry` with registration audits, pure graph patch functions, IO/model resolution. All GPU-free and unit-tested. |
| **Execution bridge** | `hordelib/execution/` | `ExecutionBackend` protocol, in-process ComfyUI backend, VRAM monkeypatches, pure graph utilities. |
| **Environment manifest** | `hordelib/installation/` | `manifest.json` pins ComfyUI commit + external custom nodes. `EnvironmentInstaller` checks out/clones idempotently. |
| **ComfyUI bridge** | `hordelib/comfy_horde.py` | `Comfy_Horde` -- the actual ComfyUI process. `do_comfy_import()` applies monkeypatches. |
| **Model managers** | `hordelib/model_manager/` | Per-category model records (pydantic), download/validate/lookup, LoRA/TI adhoc cache. |
| **Pipeline JSON** | `hordelib/pipelines/` | Backend-format pipeline files executed at runtime. |
| **Pipeline designs** | `hordelib/pipeline_designs/` | ComfyUI-app-editable source workflows (convert to `pipelines/`). |
| **Custom nodes** | `hordelib/nodes/` | First-party Horde nodes (`HordeCheckpointLoader`, `HordeImageOutput`) and vendored compat nodes. External nodes (controlnet aux, QR) are git-pinned in the manifest. |

### Pipeline flow

```
dict payload --> normalize (horde_compat) --> ImageGenPayload
  --> resolve_context (model files, loras, TIs)
  --> registry.select (Selector tier + criteria --> PipelineDefinition)
  --> materialize (bindings + patch steps --> ComfyGraph)
  --> backend.run_pipeline --> ResultingImageReturn
```

### Key invariants

- **`initialise()` before everything.** Importing `HordeLib` or any ComfyUI-touching code before `initialise()` raises `RuntimeError`. The test suite enforces this.
- **Public API only.** Consumers import from `hordelib.api`. Anything not re-exported there is internal and may change without notice. Enforced by `tests/meta/test_public_api_contract.py`.
- **Clamp, don't reject.** Payload validation coerces bad values to defaults rather than raising -- the Horde contract requires tolerance.
- **Node titles are the parameter namespace.** A KSampler titled `sampler` produces dotted parameters `sampler.cfg`, `sampler.seed`, etc. Renaming a title silently breaks parameter mapping.
- **Pipeline selection is registry-driven.** Declarative `Selector`s (named tiers + typed criteria) replace the old if/elif tree, and every definition is audited against its graph at registration. Adding a pipeline is one module in `hordelib/pipeline/families/image_gen/` plus one entry in `IMAGE_PIPELINES`. See [Adding a pipeline](docs/adding-a-pipeline.md) and [Adding a baseline](docs/adding-a-baseline.md).

---

## Model directory layout

`AIWORKER_CACHE_HOME` must follow the AI Horde directory structure:

```
<AIWORKER_CACHE_HOME>/
    clip/
    codeformer/
    compvis/
        Deliberate.ckpt
        ...
    controlnet/
    embeds/
    esrgan/
    gfpgan/
    safety_checker/
```

Models are managed through `SharedModelManager` (download, validate, list available). See `examples/download_all_sd_models.py` for bulk downloading.

---

## Development

Requirements: `git`, [`uv`](https://docs.astral.sh/uv/), `AIWORKER_CACHE_HOME` set to a model directory, and a CUDA GPU.

### Build, lint, test

The whole toolchain lives in one `uv`-managed `.venv`. Sync it once (installs the CUDA 12.6 build of
torch 2.12.0 via `[tool.uv.sources]`), then drive tools with `uv run --no-sync`:

```bash
uv sync                          # one-time, and after dependency changes

# Full test suite. The import guard runs first in a clean process (it asserts no ComfyUI
# import happens before initialise()), then the rest of the suite runs:
uv run --no-sync pytest -x tests/meta -k test_no_initialise_comfy_horde
uv run --no-sync pytest -x tests --cov --ignore=tests/meta --durations=20

uv run --no-sync pytest -k <pattern>           # run a subset
uv run --no-sync pre-commit run --all-files    # lint + format gate (black + ruff + mypy)
uv build                                       # build sdist + wheel
```

(Equivalently, activate `.venv` and call `pytest` / `pre-commit` directly.)

The first test run downloads many multi-GB models and is very slow. Most CI/agent sandboxes
cannot run the full inference suite; prefer `uv run --no-sync pre-commit run --all-files` for
quick feedback.

Tests require a CUDA GPU, `AIWORKER_CACHE_HOME`, `CIVIT_API_TOKEN`, and the committed `images_expected/` reference images. Image-output tests compare generated images against references via cosine/histogram similarity. Set `HORDELIB_SKIP_SIMILARITY_FAIL=1` to downgrade similarity failures to skips on different hardware.

### Key test areas

| Path | What it covers |
|---|---|
| `tests/pipeline/` | Payload models, graph operations, template integrity, registry selection -- all GPU-free |
| `tests/execution/` | Graph utilities, modality seams -- GPU-free |
| `tests/meta/` | Public API contract, `initialise()`-before-use guard |
| `tests/test_horde_inference_*.py` | Full GPU inference with image-similarity oracles |
| `tests/installation/` | Manifest round-trip, installer idempotency |

### Designing and converting pipelines

Design pipelines in the ComfyUI web app (`cd ComfyUI && python main.py`, then http://127.0.0.1:8188/). Assign stable `title` attributes to nodes -- these become the dotted parameter names. Save the design in `hordelib/pipeline_designs/` as `pipeline_<name>.json`.

Convert to backend format by running the pipeline through the embedded ComfyUI (it auto-saves `comfy-prompt.json`), then save the result in `hordelib/pipelines/` with the same filename.

Do not hand-edit `pipelines/*.json` if a `pipeline_designs/` source exists.

### Updating ComfyUI or external custom nodes

Change the relevant commit SHA in `hordelib/installation/manifest.json`:
- `comfyui_ref` for ComfyUI itself
- `ref` for each external custom node

Full 40-character SHAs only. The next `hordelib.initialise()` (or `python -m hordelib.installation`) applies the update. Run the full test suite afterwards.

### Local wheel build

```bash
uv build        # writes dist/horde_engine-<version>.tar.gz and -<version>-py3-none-any.whl
```

The version is derived from git tags by `hatch-vcs`: a clean checkout sitting exactly on a
`vX.Y.Z` tag builds `X.Y.Z`; anything else builds a dev version (e.g. `1.2.3.devN`). ComfyUI is
not bundled into the wheel -- it is fetched per `hordelib/installation/manifest.json` on first
`initialise()`. To smoke-test the wheel away from the source tree (so the repo's `hordelib/`
cannot shadow the installed package), install it into a throwaway venv created outside the repo.

### Release process

Releases are cut by pushing a version tag. The tag **is** the version (`hatch-vcs`), and the
[release workflow](.github/workflows/release.yml) builds with `uv build` and publishes to PyPI
via trusted publishing (then creates a GitHub Release for the tag).

```bash
git checkout main && git pull
git tag vX.Y.Z          # must be vX.Y.Z and sit on the commit you want released
git push origin vX.Y.Z
```

---

## Documentation

| Document | Content |
|---|---|
| [docs/public_api.md](docs/public_api.md) | Declared consumer surface and key patterns |
| [docs/adding-a-baseline.md](docs/adding-a-baseline.md) | Checklist for supporting a new model architecture |
| [docs/modality-readiness.md](docs/modality-readiness.md) | Audit of audio/video modality seams |
| [docs/plans/major-overhaul-1.md](docs/plans/major-overhaul-1.md) | v3 phased refactor plan and status |
| [AGENTS.md](AGENTS.md) | Comprehensive codebase map, conventions, and sensitive areas |

---

## License

[GNU Affero General Public License v3.0](LICENSE)

## Acknowledgments

`horde-engine` builds on a large number of open-source projects:

- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) (GPLv3) -- the graph/nodes inference engine this project wraps.
- [comfyui_controlnet_aux](https://github.com/Fannovel16/comfy_controlnet_preprocessors) (Apache 2.0) -- ControlNet preprocessors, installed via the manifest.
- [ComfyUI-layerdiffuse](https://github.com/huchenlei/ComfyUI-layerdiffuse) -- layer diffusion nodes (vendored with compat fixes).
- [facerestore_cf](https://civitai.com/models/24690/comfyui-facerestore-node) -- face restoration nodes (vendored with compat fixes).
- [ComfyQR](https://gitlab.com/sofuego-comfy-nodes/ComfyQR) -- QR code generation, installed via the manifest.
- [horde-sdk](https://github.com/Haidra-Org/horde-sdk) and [horde-model-reference](https://github.com/Haidra-Org/horde-model-reference) -- payload types and model metadata consumed by this library.



<!-- Badges: -->

[pypi-image]: https://badge.fury.io/py/hordelib.svg?branch=main&kill_cache=1
[pypi-url]: https://badge.fury.io/py/horde-engine
[downloads-image]: https://pepy.tech/badge/horde-engine
[downloads-url]: https://pepy.tech/project/horde-engine
[license-url]: https://img.shields.io/github/license/Haidra-Org/hordelib
[build-image]: https://github.com/Haidra-Org/hordelib/actions/workflows/maintests.yml/badge.svg?branch=main
[all-model-images]: https://badgen.net/badge/all-models/images/blue?icon=awesome
[build-url]: https://tests.hordelib.org/
[main-test-image]: https://badgen.net/badge/main/latest-images/blue?icon=awesome
[main-test-url]: https://tests.hordelib.org/
[pr-test-image]: https://badgen.net/badge/develop/latest-images/blue?icon=awesome
[pr-test-url]: https://tests.hordelib.org/unstable/index.html
[all-model-url]: https://tests.hordelib.org/all_models/
[changelog-image]: https://img.shields.io/badge/Release-Changelog-yellow
[changelog-url]: https://github.com/Haidra-Org/hordelib/releases
