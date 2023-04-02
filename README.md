# hordelib

`hordelib` is a thin wrapper around [ComfyUI](https://github.com/comfyanonymous/ComfyUI) primarily to enable the [AI Horde](https://aihorde.net/) to run inference pipelines designed in ComfyUI. `hordelib` and ComfyUI is are licensed under the terms of the GNU General Public License.

## Purpose

The goal here is to be able to design inference pipelines in the excellent ComfyUI, and then call those inference pipelines programmatically, in a manner ultimately suitable for use in stable horde.

For example, if we designed a pipeline named "stable_diffusion" in the ComfyUI and saved the JSON within hordelib, it could be executed simply with:
```python
params = {
    "sampler.seed": 12345,
    "sampler.cfg": 7.5,
    "sampler.scheduler": "karras",
    "sampler.sampler_name": "dpmpp_2m",
    "sampler.steps": 25,
    "prompt.text": "a closeup photo of a confused dog",
    "negative_prompt.text": "cat, black and white, deformed",
    "model_loader.ckpt_name": "model.ckpt",
    "empty_latent_image.width": 768,
    "empty_latent_image.height": 768,
}
images = self.comfy.run_image_pipeline("stable_diffusion", params)
```

## Development

Requirements:
- `git` (install git)
- `tox` (`pip install tox`)
- Copy _any_ checkpoint model into the project root named `model.ckpt`

### Running the Tests

Simply execute: `tox`

This will take a while the first time as it installs all the dependencies.

If the tests run successfully images will be produced in the project with the name of each pipeline, for example, `pipeline_stable_diffusion.png`.

### Directory Structure

`hordelib/pipeline_designs/`
Contains ComfyUI pipelines in a format that can be opened by the ComfyUI web app. These saved directly from the web app.

`hordelib/pipelines/`
Contains the above pipeline JSON files converted to the format required by the backend pipeline processor. These are converted from the web app, see _Converting ComfyUI pipelines_ below.

`hordelib/nodes/` These are the custom ComfyUI nodes we use for `hordelib` specific processing.

### Designing ComfyUI Pipelines

Use the standard ComfyUI web app. Use the "title" attribute to name the nodes, these names become parameter names in the `hordelib`. For example, a KSampler with the "title" of "sampler2" would become a parameter `sampler2.seed`, `sampler2.cfg`, etc. Load the pipeline `hordelib/pipeline_designs/pipeline_stable_diffusion.json` in the ComfyUI web app for an example.

Save any new pipeline in `hordelib/pipeline_designs` using the naming convention "pipeline_\<name\>.json".

Convert the JSON for the model (see _Converting ComfyUI pipelines_ below) and save the resulting JSON in `hordelib/pipelines` using the same filename as the previous JSON file.

That is all. This can then be called from `hordelib` using the `run_image_pipeline()` method in `hordelib.comfy.Comfy()`

### Converting ComfyUI pipelines

The quickest way to get from a pipeline diagram in the ComfyUI web app to a usable JSON file is to simply patch the ComfyUI backend to save the JSON we require when the web app submits the inference pipeline for rendering.

Patch `ComfyUI/execution.py:validate_prompt()` to include the following just before the final `return` statement:
```python
    with open("prompt.json", "wt", encoding="utf-8") as f:
        f.write(json.dumps(prompt, indent=4))
```

This will create the file `prompt.json` in the root of the ComfyUI project for the submitted pipeline job.

### Build Configuration

The config files for the project are: `pyproject.toml`, `tox.ini` and `requirements.txt`

### PyPi Publishing

Three steps:
1. Bump the version in `hordelib/__init__.py`
2. `python -m build`
3. `twine upload -r pypi dist/*`