"""Unit tests for the environment manifest. No network or GPU required."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from hordelib.installation.manifest import (
    ComfyEnvironmentManifest,
    CustomNodeSpec,
    load_packaged_manifest,
)

A_SHA = "fb991e2c1e7476809d566a4620c2132e05a466dd"
B_SHA = "0123456789abcdef0123456789abcdef01234567"


class TestManifestValidation:
    def test_packaged_manifest_loads(self):
        manifest = load_packaged_manifest()
        assert len(manifest.comfyui_ref) == 40

    def test_short_sha_rejected(self):
        with pytest.raises(ValidationError):
            ComfyEnvironmentManifest(comfyui_ref="fb991e2c")

    def test_branch_name_rejected(self):
        with pytest.raises(ValidationError):
            ComfyEnvironmentManifest(comfyui_ref="master" + "0" * 34)

    def test_node_short_sha_rejected(self):
        with pytest.raises(ValidationError):
            CustomNodeSpec(name="x", repo_url="https://example.com/x.git", ref="abc123")


class TestManifestRoundTrip:
    def test_file_round_trip(self, tmp_path: Path):
        manifest = ComfyEnvironmentManifest(
            comfyui_ref=A_SHA,
            custom_nodes=[
                CustomNodeSpec(
                    name="comfyui_controlnet_aux",
                    repo_url="https://github.com/Fannovel16/comfyui_controlnet_aux",
                    ref=B_SHA,
                    registry_id="comfyui_controlnet_aux",
                ),
            ],
        )
        path = tmp_path / "manifest.json"
        manifest.to_file(path)
        loaded = ComfyEnvironmentManifest.from_file(path)
        assert loaded == manifest


class TestManagerSnapshotExport:
    def test_snapshot_shape(self):
        manifest = ComfyEnvironmentManifest(
            comfyui_ref=A_SHA,
            custom_nodes=[
                CustomNodeSpec(name="node_a", repo_url="https://example.com/a.git", ref=B_SHA),
            ],
        )
        snapshot = manifest.to_manager_snapshot()
        assert snapshot["comfyui"] == A_SHA
        assert snapshot["git_custom_nodes"]["https://example.com/a.git"]["hash"] == B_SHA
        assert snapshot["git_custom_nodes"]["https://example.com/a.git"]["disabled"] is False
        assert snapshot["file_custom_nodes"] == []
        # Must be JSON-serializable as ComfyUI-Manager expects
        json.dumps(snapshot)
