# hordelib

`hordelib` is a thin wrapper around [ComfyUI](https://github.com/comfyanonymous/ComfyUI) primarily to enable the [AI Horde](https://aihorde.net/) to run inference pipelines designed in ComfyUI. `hordelib` and ComfyUI is are licensed under the terms of the GNU General Public License.

## Development

To build the development environment and run the unit tests simply run `tox` from the project root. This will assume you have `git` installed and on the command line search path.

If you don't have tox, install with `pip install tox`.

### Tests

If the tests run successfully an image will be produced named `image-test.png`