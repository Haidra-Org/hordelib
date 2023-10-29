# consts.py

from enum import Enum, auto

from strenum import StrEnum

from hordelib.config_path import get_hordelib_path

COMFYUI_VERSION = "0e763e880f5e838e7a1e3914444cae6790c48627"
"""The exact version of ComfyUI version to load."""

REMOTE_PROXY = ""

REMOTE_MODEL_DB = "https://raw.githubusercontent.com/db0/AI-Horde-image-model-reference/main/"
"""The default base endpoint where to find model databases. See MODEL_DB_NAMES for valid database names."""

RELEASE_VERSION = (get_hordelib_path() / "_version.py").exists()
"""A flag for if this is a pypi release or a git dev mode"""


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

CIVITAI_COOKIES = {
    "__Secure-civitai-token": "eyJhbGciOiJkaXIiLCJlbmMiOiJBMjU2R0NNIn0..JkkSLR2cGyYialBY.Rh2n4ru0hG5ASXVmRHJCU82RQ1ErgximkNls0KpYhlMG4gBywHuqbFNg20IBxCSWpN5uEHRd-uwR967Vv1wR7g1W7Iuqut0KaHS6I7PEgZEvhCZLG08gFNiPhJ9o7Ep3vfOkNKKdB_kUcK_tbL67dHprmMLR6Ww4ZcsX0vShQ_VNXxewrwBbjmxSXgyyAMYGWNmchWDxzP08GU09IonGQJt-e0PJok_gRCfzPd7Y4Q06aPus7TRWQGKiYpZCIB7xGbRwde62JpjowpxjWOLZa0bKJanYkiFUw3vCZgvMxzO-H9sb62HXck7GOgOxS6NOVYUvvN1nF4r0c0pVInbPrn406oaeav5mvO2KJa9KZIuoSaS5bLyqxgV32A2GISL1fQ-K2FL84E0gFZZxMGVXd0qW7Z_U7ft4laV-vKUzELtRzwM_omD8toNvRL5c96HqoPQ4FM5c6sXmsOiufVNNPB4-LlFmRivTdEQ0Q2Wz3b3kypkYjqb3FnNLAFxgGaftTqIf3TlcJkaF1YZFSes9TbSTTXAX3dK6aZltauCgoAx0yQggUk1qtOuup4bZhkc2fIyorbMKjU-z_A8Jvea4R-RWHQmiM1YVfHQ1pPc98tbK2bkJpUBEF5_QGTHPvqsaHISfEx3uKSmh4aan3smC-cUnmfK-xc6S_c3Aokmxe4Ia9IvOYu1T-UhoZ6PcHwqD8tcRLgURAH4zgZ1aOhGavlIi7mT0RiOgBGkiVcbWFBrCb9V_a6K0D5BaHqDAALKIdaffsCzmKGTJpwFqxSA7cmFwxDJ3ROjj2uVQ7LRr0Nyv8qL8yiMsonJNvGg-4jejMc2bq4zvx5LgpNGGamHljVGLDIFZuZrBfR7xNzv3V_-aoIPES8z8YOeOYPIORmW0.T_1IZDaN-HHPWLJyA6Dv5w"
}