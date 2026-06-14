# consts.py

from enum import Enum, auto
from pathlib import Path

from strenum import StrEnum

from hordelib.config_path import is_release_install
from hordelib.installation.manifest import load_packaged_manifest

COMFYUI_VERSION = load_packaged_manifest().comfyui_ref
"""The exact ComfyUI commit to load, as pinned by hordelib/installation/manifest.json."""

RELEASE_VERSION = is_release_install()
"""A flag for if this is a pypi release or a git dev mode"""

CIVITAI_API_PATH = "civitai.com/api"
"""A domain and path to CivitAI API"""


class HordeSupportedBackends(Enum):
    ComfyUI = auto()


class MODEL_CATEGORY_NAMES(StrEnum):
    """Look up str enum for the categories of models (compvis, controlnet, clip, etc...)."""

    default_models = auto()
    """Unspecified model category."""
    codeformer = auto()
    compvis = auto()
    """Stable Diffusion models."""
    controlnet = auto()
    # diffusers = "auto()
    esrgan = auto()
    gfpgan = auto()
    safety_checker = auto()
    lora = auto()
    ti = auto()
    blip = auto()
    clip = auto()
    miscellaneous = auto()


# Default model managers to load
DEFAULT_MODEL_MANAGERS = {
    MODEL_CATEGORY_NAMES.codeformer: True,
    MODEL_CATEGORY_NAMES.compvis: True,
    MODEL_CATEGORY_NAMES.controlnet: True,
    # MODEL_CATEGORY_NAMES.diffusers: True,
    MODEL_CATEGORY_NAMES.esrgan: True,
    MODEL_CATEGORY_NAMES.gfpgan: True,
    # MODEL_CATEGORY_NAMES.safety_checker: True,
    MODEL_CATEGORY_NAMES.lora: True,
    MODEL_CATEGORY_NAMES.ti: True,
    MODEL_CATEGORY_NAMES.miscellaneous: True,
}
"""The default model managers to load."""  # XXX Clarify

MODEL_DB_NAMES = {
    MODEL_CATEGORY_NAMES.codeformer: MODEL_CATEGORY_NAMES.codeformer,
    MODEL_CATEGORY_NAMES.compvis: "stable_diffusion",
    MODEL_CATEGORY_NAMES.controlnet: MODEL_CATEGORY_NAMES.controlnet,
    # MODEL_CATEGORY_NAMES.diffusers: MODEL_CATEGORY_NAMES.diffusers,
    MODEL_CATEGORY_NAMES.esrgan: MODEL_CATEGORY_NAMES.esrgan,
    MODEL_CATEGORY_NAMES.gfpgan: MODEL_CATEGORY_NAMES.gfpgan,
    MODEL_CATEGORY_NAMES.safety_checker: MODEL_CATEGORY_NAMES.safety_checker,
    MODEL_CATEGORY_NAMES.lora: MODEL_CATEGORY_NAMES.lora,
    MODEL_CATEGORY_NAMES.ti: MODEL_CATEGORY_NAMES.ti,
    MODEL_CATEGORY_NAMES.miscellaneous: MODEL_CATEGORY_NAMES.miscellaneous,
}
"""The name of the json file (without the extension) of the corresponding model database."""

MODEL_FOLDER_NAMES = {
    MODEL_CATEGORY_NAMES.codeformer: MODEL_CATEGORY_NAMES.codeformer,
    MODEL_CATEGORY_NAMES.compvis: "compvis",
    MODEL_CATEGORY_NAMES.controlnet: MODEL_CATEGORY_NAMES.controlnet,
    # MODEL_CATEGORY_NAMES.diffusers: MODEL_CATEGORY_NAMES.diffusers,
    MODEL_CATEGORY_NAMES.esrgan: MODEL_CATEGORY_NAMES.esrgan,
    MODEL_CATEGORY_NAMES.gfpgan: MODEL_CATEGORY_NAMES.gfpgan,
    MODEL_CATEGORY_NAMES.safety_checker: MODEL_CATEGORY_NAMES.safety_checker,
    MODEL_CATEGORY_NAMES.lora: MODEL_CATEGORY_NAMES.lora,
    MODEL_CATEGORY_NAMES.ti: MODEL_CATEGORY_NAMES.ti,
    MODEL_CATEGORY_NAMES.miscellaneous: MODEL_CATEGORY_NAMES.miscellaneous,
}
"""The folder name on disk where the models are stored in AIWORKER_CACHE_HOME."""


COMPONENT_PURPOSE_FOLDERS: dict[str, str] = {
    "vae": "vae",
    "text_encoders": "text_encoders",
    "text_encoder": "text_encoders",
}
"""Multi-file model components whose ``file_purpose`` routes them to a sibling ComfyUI folder.

A model such as Qwen-Image ships its unet, VAE and text-encoder as separate files. ComfyUI's
component loaders look for the VAE in ``<models>/vae`` and the text-encoder in
``<models>/text_encoders`` (see the ``folder_names_and_paths`` setup in ``hordelib.comfy_horde``),
not in the owning manager's own folder (e.g. ``<models>/compvis``). Keys are
``DownloadRecord.file_purpose`` values; values are the destination folder names. Anything not
listed here (e.g. ``unet``/checkpoints) stays in the manager's folder.

This mirrors the pre-refactor legacy layout (the old records used an explicit ``"../vae"``
directory redirect), so component files fetched by older hordelib versions are found in place and
are never needlessly re-downloaded.
"""


def component_relative_path(file_name: str, file_purpose: str | None) -> Path:
    """Return a download/validation path for *file_name*, relative to the manager's model folder.

    Components with a recognised ``file_purpose`` (see :data:`COMPONENT_PURPOSE_FOLDERS`) are
    redirected to the matching sibling folder via a ``../<folder>`` prefix so ComfyUI's component
    loaders find them; every other file stays in the manager's own folder. The ``..``-relative
    form resolves correctly for any manager folder that lives directly under the models directory.
    """
    if file_purpose:
        folder = COMPONENT_PURPOSE_FOLDERS.get(file_purpose)
        if folder:
            return Path("..") / folder / file_name
    return Path(file_name)
