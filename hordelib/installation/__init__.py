"""Declarative installation of the ComfyUI environment (ComfyUI itself + custom nodes).

The single source of truth is ``manifest.json`` (packaged alongside this module), which pins
the ComfyUI commit and every external custom-node repository. :class:`EnvironmentInstaller`
makes the on-disk environment match the manifest, idempotently.
"""

from hordelib.installation.installer import EnvironmentInstaller
from hordelib.installation.manifest import ComfyEnvironmentManifest, CustomNodeSpec, load_packaged_manifest

__all__ = [
    "ComfyEnvironmentManifest",
    "CustomNodeSpec",
    "EnvironmentInstaller",
    "load_packaged_manifest",
]
