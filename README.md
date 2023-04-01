# hordelib

`hordelib` is a thin wrapper around [ComfyUI](https://github.com/comfyanonymous/ComfyUI) primarily to enable the [AI Horde](https://aihorde.net/) to run inference pipelines designed in ComfyUI. `hordelib` and ComfyUI is are licensed under the terms of the GNU General Public License.

## Development

Requirements:
- `git` (install git)
- `tox` (`pip install tox`)
- Copy _any_ checkpoint model into the project root named `model.ckpt`

To build the development environment and run the unit tests simply run `tox` from the project root. This will take a while the first time as it installs all the dependencies.

### Tests

If the tests run successfully an image will be produced named `image-test.png`

### Build Configuration

The config files for the project are: `pyproject.toml`, `tox.ini` and `requirements.txt`
