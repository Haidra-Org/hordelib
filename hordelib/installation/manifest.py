"""The ComfyUI environment manifest: pinned ComfyUI commit + pinned custom nodes."""

import json
from pathlib import Path
from typing import Self

from pydantic import BaseModel, field_validator

_FULL_SHA_LENGTH = 40


class CustomNodeSpec(BaseModel):
    """A single external custom-node package, pinned to an exact commit."""

    name: str
    """The directory name under ``custom_nodes/`` (Comfy Registry convention)."""
    repo_url: str
    """The git repository URL to clone from."""
    ref: str
    """The full commit SHA to check out. Always a full SHA so installs are reproducible."""
    registry_id: str | None = None
    """The Comfy Registry identifier, if published there. Informational only."""

    @field_validator("ref")
    @classmethod
    def _ref_must_be_full_sha(cls, v: str) -> str:
        if len(v) != _FULL_SHA_LENGTH or not all(c in "0123456789abcdef" for c in v.lower()):
            raise ValueError(f"Custom node ref must be a full 40-character commit SHA, got: {v!r}")
        return v


class ComfyEnvironmentManifest(BaseModel):
    """The full declaration of the ComfyUI environment hordelib requires."""

    schema_version: int = 1
    comfyui_repo: str = "https://github.com/comfyanonymous/ComfyUI.git"
    comfyui_ref: str
    """The exact ComfyUI commit SHA."""
    custom_nodes: list[CustomNodeSpec] = []

    @field_validator("comfyui_ref")
    @classmethod
    def _comfyui_ref_must_be_full_sha(cls, v: str) -> str:
        if len(v) != _FULL_SHA_LENGTH or not all(c in "0123456789abcdef" for c in v.lower()):
            raise ValueError(f"comfyui_ref must be a full 40-character commit SHA, got: {v!r}")
        return v

    @classmethod
    def from_file(cls, path: Path) -> Self:
        """Load a manifest from a JSON file."""
        return cls.model_validate_json(path.read_text(encoding="utf-8"))

    def to_file(self, path: Path) -> None:
        """Write this manifest to a JSON file."""
        path.write_text(self.model_dump_json(indent=4) + "\n", encoding="utf-8")

    def to_manager_snapshot(self) -> dict:
        """Export in ComfyUI-Manager snapshot format (``snapshots/*.json``).

        This allows a hordelib environment to be reproduced (or inspected) with
        ComfyUI-Manager / comfy-cli tooling.
        """
        return {
            "comfyui": self.comfyui_ref,
            "git_custom_nodes": {
                node.repo_url: {
                    "hash": node.ref,
                    "disabled": False,
                }
                for node in self.custom_nodes
            },
            "file_custom_nodes": [],
        }


def load_packaged_manifest() -> ComfyEnvironmentManifest:
    """Load the manifest shipped with this hordelib version."""
    manifest_path = Path(__file__).parent / "manifest.json"
    manifest = ComfyEnvironmentManifest.from_file(manifest_path)

    # json.loads round-trip guard: a malformed packaged manifest should fail loudly at import
    # of the installer, not at install time on a worker.
    json.loads(manifest.model_dump_json())
    return manifest
