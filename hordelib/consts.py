# consts.py

from enum import Enum, auto

from strenum import StrEnum

from hordelib.config_path import get_hordelib_path

COMFYUI_VERSION = "2a813c3b09292c9aeab622ddf65d77e5d8171d0d"
"""The exact version of ComfyUI version to load."""

REMOTE_PROXY = ""

REMOTE_MODEL_DB = "https://raw.githubusercontent.com/db0/AI-Horde-image-model-reference/main/"
"""The default base endpoint where to find model databases. See MODEL_DB_NAMES for valid database names."""

RELEASE_VERSION = (get_hordelib_path() / "_version.py").exists()
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


# Default model managers to load
DEFAULT_MODEL_MANAGERS = {
    MODEL_CATEGORY_NAMES.codeformer: True,
    MODEL_CATEGORY_NAMES.compvis: True,
    MODEL_CATEGORY_NAMES.controlnet: True,
    # MODEL_CATEGORY_NAMES.diffusers: True,
    MODEL_CATEGORY_NAMES.esrgan: True,
    MODEL_CATEGORY_NAMES.gfpgan: True,
    MODEL_CATEGORY_NAMES.safety_checker: True,
    MODEL_CATEGORY_NAMES.lora: True,
    MODEL_CATEGORY_NAMES.ti: True,
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
}
"""The folder name on disk where the models are stored in AIWORKER_CACHE_HOME."""
