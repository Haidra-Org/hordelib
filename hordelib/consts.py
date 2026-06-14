# consts.py

from enum import Enum, auto

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
