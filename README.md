# hordelib

`hordelib` is a thin wrapper around [ComfyUI](https://github.com/comfyanonymous/ComfyUI) primarily to enable the [AI Horde](https://aihorde.net/) to run inference pipelines designed in ComfyUI. `hordelib` and ComfyUI is are licensed under the terms of the GNU General Public License.

The developers of this project can be found in the AI Horde Discord server: https://discord.gg/3DxrhksKznb

## Purpose

The goal here is to be able to design inference pipelines in the excellent ComfyUI, and then call those inference pipelines programmatically, in a manner ultimately suitable for use in stable horde.

## Installation

If being installed from pypi, use a requirements file of the form:
```
--extra-index-url https://download.pytorch.org/whl/cu117
hordelib

...your other dependancies...
```

## Usage

Horde payloads can be processed simply with (for example):

```python
import os
from hordelib.horde import HordeLib

# Wherever your models are
os.environ["AIWORKER_CACHE_HOME"] = "f:/ai/models"

generate = HordeLib()

data = {
    "sampler_name": "k_dpmpp_2m",
    "cfg_scale": 7.5,
    "denoising_strength": 1.0,
    "seed": 123456789,
    "height": 512,
    "width": 512,
    "karras": True,
    "tiling": False,
    "hires_fix": False,
    "clip_skip": 1,
    "control_type": "canny",
    "image_is_control": False,
    "return_control_map": False,
    "prompt": "an ancient llamia monster",
    "ddim_steps": 25,
    "n_iter": 1,
    "model": "Deliberate.ckpt",
}
pil_image = generate.text_to_image(data)
pil_image.save("test.png")
```

## Development

Requirements:
- `git` (install git)
- `tox` (`pip install tox`)
- Copy _any_ checkpoint model into the project root named `model.ckpt`

### Running the Tests

Simply execute: `tox` (or `tox -q` for less noisy output)

This will take a while the first time as it installs all the dependencies.

If the tests run successfully images will be produced in the project with the name of each pipeline, for example, `pipeline_stable_diffusion.png` and likely some other .png files too.

#### Running a specific test file

`tox -- -k <filename>` for example `tox -- -k test_initialisation`

### Directory Structure

`hordelib/pipeline_designs/`
Contains ComfyUI pipelines in a format that can be opened by the ComfyUI web app. These saved directly from the web app.

`hordelib/pipelines/`
Contains the above pipeline JSON files converted to the format required by the backend pipeline processor. These are converted from the web app, see _Converting ComfyUI pipelines_ below.

`hordelib/nodes/` These are the custom ComfyUI nodes we use for `hordelib` specific processing.

### Running ComfyUI Web Application

`tox -e comfyui`

Then open a browser at: http://127.0.0.1:8188/

### Designing ComfyUI Pipelines

Use the standard ComfyUI web app. Use the "title" attribute to name the nodes, these names become parameter names in the `hordelib`. For example, a KSampler with the "title" of "sampler2" would become a parameter `sampler2.seed`, `sampler2.cfg`, etc. Load the pipeline `hordelib/pipeline_designs/pipeline_stable_diffusion.json` in the ComfyUI web app for an example.

Save any new pipeline in `hordelib/pipeline_designs` using the naming convention "pipeline_\<name\>.json".

Convert the JSON for the model (see _Converting ComfyUI pipelines_ below) and save the resulting JSON in `hordelib/pipelines` using the same filename as the previous JSON file.

That is all. This can then be called from `hordelib` using the `run_image_pipeline()` method in `hordelib.comfy.Comfy()`

### Converting ComfyUI pipelines

In addition to the design file saved from the UI, we need to save the pipeline file in the backend format. This file is created in the `hordelib` project root named `comfy-prompt.json` automatically if you run a pipeline through the ComfyUI version embedded in `hordelib`. Running ComfyUI with `tox -e comfyui` automatically patches ComfyUI so this JSON file is saved.

### Build Configuration

The main config files for the project are: `pyproject.toml`, `tox.ini` and `requirements.txt`

### PyPi Publishing

Three steps:
1. Bump the version in `hordelib/__init__.py`
1. `tox` _make sure everything works_
1. `python publish.py` _builds the dist files_
1. `twine upload -r pypi dist/*`

### Updating the embedded version of ComfyUI

We use a ComfyUI version pinned to a specific commit, see `hordelib/__init__.py:COMFYUI_VERSION`

To test if the latest version works and upgrade to it, from the project root simply:

1. `cd hordelib/Comfy` _Change CWD to the embedded comfy_
1. `git checkout master` _Switch to master branch_
1. `git pull` _Get the latest comfyui code_
1. `git rev-parse HEAD` _Print the commit hash we'll need this_
1. `cd ../../` _Get back to the hordelib project root_
1. `tox` _See if everything still works_

If everything still works. Take the commit hash printed above and put it in `hordelib/__init__.py:COMFYUI_VERSION`

Now ComfyUI is pinned to a new version.
