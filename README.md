# hordelib

[![PyPI version](https://badge.fury.io/py/hordelib.svg?branch=main&nocache=1)](https://badge.fury.io/py/hordelib)
[![Downloads](https://pepy.tech/badge/hordelib)](https://pepy.tech/project/hordelib)
![GitHub license](https://img.shields.io/github/license/jug-dev/hordelib)

[![Build](https://github.com/jug-dev/hordelib/actions/workflows/maintests.yml/badge.svg?branch=main)](http://hordelib.s3-website-eu-west-1.amazonaws.com/)
[![Test Images](https://badgen.net/badge/main/latest-images/blue?icon=awesome)](http://hordelib.s3-website-eu-west-1.amazonaws.com/)
[![Test Images](https://badgen.net/badge/develop/latest-images/blue?icon=awesome)](http://hordelib.s3-website-eu-west-1.amazonaws.com/unstable)

`hordelib` is a thin wrapper around [ComfyUI](https://github.com/comfyanonymous/ComfyUI) primarily to enable the [AI Horde](https://aihorde.net/) to run inference pipelines designed visually in the ComfyUI GUI.

The developers of this project can be found in the AI Horde Discord server: [https://discord.gg/3DxrhksKzn](https://discord.gg/3DxrhksKzn)

NOTE: This project is in early development and is not yet in use by Stable Horde.

## Purpose

The goal here is to be able to design inference pipelines in the excellent ComfyUI, and then call those inference pipelines programmatically. Whilst providing features that maintain compatibility with the existing horde implementation.

## Installation

If being installed from pypi, use a requirements file of the form:
```
--extra-index-url https://download.pytorch.org/whl/cu117
hordelib

...your other dependencies...
```

## Usage

Horde payloads can be processed simply with (for example):

```python
import os
import hordelib
hordelib.initialise()

from hordelib.horde import HordeLib
from hordelib.shared_model_manager import SharedModelManager

# Wherever your models are
os.environ["AIWORKER_CACHE_HOME"] = "f:/ai/models"

generate = HordeLib()
SharedModelManager.loadModelManagers(compvis=True)
SharedModelManager.manager.load("Deliberate")

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
    "control_type": None,
    "image_is_control": False,
    "return_control_map": False,
    "prompt": "an ancient llamia monster",
    "ddim_steps": 25,
    "n_iter": 1,
    "model": "Deliberate",
}
pil_image = generate.basic_inference(data)
pil_image.save("test.png")
```

Note that `hordelib.initialise()` will erase all command line arguments from argv. So make sure you parse them before you call that.

See `tests/run_*.py` for more standalone examples.

## Development

Requirements:
- `git` (install git)
- `tox` (`pip install tox`)
- Set the environmental variable `AIWORKER_CACHE_HOME` to point to a model directory.

Note the model directory must currently be in the original AI Horde directory structure:
```
<AIWORKER_CACHE_HOME>\
   nataili\
        clip\
        codeformer\
        compvis\
            Deliberate.ckpt
            ...etc...
        controlnet\
        embeds\
        esrgan\
        gfpgan\
        safety_checker\
```

### Running the Tests

Simply execute: `tox` (or `tox -q` for less noisy output)

This will take a while the first time as it installs all the dependencies.

If the tests run successfully images will be produced in the `images/` folder.

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

_Will be automatic from the `releases` branch_

But currently:

1. checkout the `releases` branch
1. `tox` _make sure everything works_
1. `python build_helper.py` _builds the dist files_
1. `twine upload -r pypi dist/*`

### Standalone "clean" environment test from Pypi

Here's an example:

Start in a new empty directory. Create requirements.txt:
```
--extra-index-url https://download.pytorch.org/whl/cu117
hordelib
```

Create the directory `images/` and copy the `test_db0.jpg` into it.

Copy `run_controlnet.py` from the `hordelib/tests/` directory.

Build a venv:
```
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

Run the test we copied:
```
python run_controlnet.py

The `images/` directory should have our test images.
```

### Updating the embedded version of ComfyUI

We use a ComfyUI version pinned to a specific commit, see `hordelib/consts.py:COMFYUI_VERSION`

To test if the latest version works and upgrade to it, from the project root simply:

1. `cd ComfyUI` _Change CWD to the embedded comfy_
1. `git checkout master` _Switch to master branch_
1. `git pull` _Get the latest comfyui code_
1. `git rev-parse HEAD` _Update the hash in `hordelib.consts:COMFYUI_VERSION`_
1. `cd ..` _Get back to the hordelib project root_
1. `tox` _See if everything still works_

Now ComfyUI is pinned to a new version.

### ComfyUI Patching

We need to patch the ComfyUI source code. It's only a small patch to:

1. Allow ComfyUI to find our custom nodes without copying files and folder around.
2. Allow make ComfyUI output some handy JSON we need for development purposes.

To create a patch file:
- Make the required changes to a clean install of ComfyUI and then run `git diff > yourfile.patch` then move the patch file to wherever you want to save it.

Note that the patch file _really_ needs to be in UTF-8 format and some common terminals, e.g. Powershell, won't do this by default. In Powershell to create a patch file use: `git diff | Set-Content -Encoding utf8 -Path yourfile.patch`

Patches can be applied with the `hordelib.install_comfyui.Installer` classes `apply_patch()` method.
