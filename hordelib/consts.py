# consts.py
import os
from enum import Enum, auto

from hordelib.config_path import get_hordelib_path

COMFYUI_VERSION = "6908f9c94992b32fbb96be0f6cd8c5b362d72a77"
"""The exact version of ComfyUI version to load."""

REMOTE_MODEL_DB = "https://raw.githubusercontent.com/db0/AI-Horde-image-model-reference/main/"
"""The default base endpoint where to find model databases. See MODEL_DB_NAMES for valid database names."""

RELEASE_VERSION = os.path.exists(os.path.join(get_hordelib_path(), "_version.py"))
"""A flag for if this is a pypi release or a git dev mode"""


class HordeSupportedBackends(Enum):
    ComfyUI = auto()


class MODEL_CATEGORY_NAMES(str, Enum):
    """Look up str enum for the categories of models (compvis, controlnet, clip, etc...)."""

    blip = "blip"
    clip = "clip"
    codeformer = "codeformer"
    compvis = "compvis"
    controlnet = "controlnet"
    diffusers = "diffusers"
    esrgan = "esrgan"
    gfpgan = "gfpgan"
    safety_checker = "safety_checker"


# Default model managers to load
DEFAULT_MODEL_MANAGERS = {
    MODEL_CATEGORY_NAMES.blip: True,
    MODEL_CATEGORY_NAMES.clip: True,
    MODEL_CATEGORY_NAMES.codeformer: True,
    MODEL_CATEGORY_NAMES.compvis: True,
    MODEL_CATEGORY_NAMES.controlnet: True,
    MODEL_CATEGORY_NAMES.diffusers: True,
    MODEL_CATEGORY_NAMES.esrgan: True,
    MODEL_CATEGORY_NAMES.gfpgan: True,
    MODEL_CATEGORY_NAMES.safety_checker: True,
}
"""The default model managers to load."""  # XXX Clarify

MODEL_DB_NAMES = {
    MODEL_CATEGORY_NAMES.blip: MODEL_CATEGORY_NAMES.blip,
    MODEL_CATEGORY_NAMES.clip: MODEL_CATEGORY_NAMES.clip,
    MODEL_CATEGORY_NAMES.codeformer: MODEL_CATEGORY_NAMES.codeformer,
    MODEL_CATEGORY_NAMES.compvis: "stable_diffusion",
    MODEL_CATEGORY_NAMES.controlnet: MODEL_CATEGORY_NAMES.controlnet,
    MODEL_CATEGORY_NAMES.diffusers: MODEL_CATEGORY_NAMES.diffusers,
    MODEL_CATEGORY_NAMES.esrgan: MODEL_CATEGORY_NAMES.esrgan,
    MODEL_CATEGORY_NAMES.gfpgan: MODEL_CATEGORY_NAMES.gfpgan,
    MODEL_CATEGORY_NAMES.safety_checker: MODEL_CATEGORY_NAMES.safety_checker,
}
"""The name of the json file (without the extension) of the corresponding model database."""

MODEL_FOLDER_NAMES = {
    MODEL_CATEGORY_NAMES.blip: MODEL_CATEGORY_NAMES.blip,
    MODEL_CATEGORY_NAMES.clip: MODEL_CATEGORY_NAMES.clip,
    MODEL_CATEGORY_NAMES.codeformer: MODEL_CATEGORY_NAMES.codeformer,
    MODEL_CATEGORY_NAMES.compvis: "compvis",
    MODEL_CATEGORY_NAMES.controlnet: MODEL_CATEGORY_NAMES.controlnet,
    MODEL_CATEGORY_NAMES.diffusers: MODEL_CATEGORY_NAMES.diffusers,
    MODEL_CATEGORY_NAMES.esrgan: MODEL_CATEGORY_NAMES.esrgan,
    MODEL_CATEGORY_NAMES.gfpgan: MODEL_CATEGORY_NAMES.gfpgan,
    MODEL_CATEGORY_NAMES.safety_checker: MODEL_CATEGORY_NAMES.safety_checker,
}
"""The folder name on disk where the models are stored in AIWORKER_CACHE_HOME."""
