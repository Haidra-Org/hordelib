"""Installer tests against local temp git repositories. No network or GPU required."""

import subprocess
from pathlib import Path

import pytest

from hordelib.installation.installer import EnvironmentInstaller, GitCommandError, _run_git
from hordelib.installation.manifest import ComfyEnvironmentManifest, CustomNodeSpec


def _git(args: list[str], cwd: Path) -> str:
    result = subprocess.run(["git", *args], cwd=str(cwd), text=True, capture_output=True, encoding="utf-8")
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


@pytest.fixture
def fake_upstream(tmp_path: Path) -> tuple[Path, str, str]:
    """A local 'upstream' repo with two commits; returns (path, first_sha, second_sha)."""
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    _git(["init", "--initial-branch=master"], upstream)
    _git(["config", "user.email", "test@example.com"], upstream)
    _git(["config", "user.name", "Test"], upstream)
    # Throwaway fixture repo: don't inherit GPG signing from the developer's global config
    _git(["config", "commit.gpgsign", "false"], upstream)
    _git(["config", "tag.gpgsign", "false"], upstream)
    (upstream / "file.txt").write_text("one\n")
    _git(["add", "."], upstream)
    _git(["commit", "-m", "first"], upstream)
    first_sha = _git(["rev-parse", "HEAD"], upstream)
    (upstream / "file.txt").write_text("two\n")
    _git(["add", "."], upstream)
    _git(["commit", "-m", "second"], upstream)
    second_sha = _git(["rev-parse", "HEAD"], upstream)
    return upstream, first_sha, second_sha


class TestEnsureComfyui:
    def test_fresh_clone_lands_on_pinned_commit(self, tmp_path: Path, fake_upstream):
        upstream, first_sha, _second_sha = fake_upstream
        manifest = ComfyEnvironmentManifest(comfyui_repo=str(upstream), comfyui_ref=first_sha)
        root = tmp_path / "ComfyUI"

        EnvironmentInstaller(manifest).ensure_comfyui(root)

        assert _git(["rev-parse", "HEAD"], root) == first_sha

    def test_idempotent_when_already_pinned(self, tmp_path: Path, fake_upstream):
        upstream, first_sha, _second_sha = fake_upstream
        manifest = ComfyEnvironmentManifest(comfyui_repo=str(upstream), comfyui_ref=first_sha)
        root = tmp_path / "ComfyUI"
        installer = EnvironmentInstaller(manifest)

        installer.ensure_comfyui(root)
        installer.ensure_comfyui(root)  # must be a no-op, not an error

        assert _git(["rev-parse", "HEAD"], root) == first_sha

    def test_updates_to_new_pin_and_discards_local_changes(self, tmp_path: Path, fake_upstream):
        upstream, first_sha, second_sha = fake_upstream
        root = tmp_path / "ComfyUI"
        EnvironmentInstaller(
            ComfyEnvironmentManifest(comfyui_repo=str(upstream), comfyui_ref=first_sha),
        ).ensure_comfyui(root)

        # Simulate local tampering (e.g. the legacy patch flow or a stray edit)
        (root / "file.txt").write_text("tampered\n")

        EnvironmentInstaller(
            ComfyEnvironmentManifest(comfyui_repo=str(upstream), comfyui_ref=second_sha),
        ).ensure_comfyui(root)

        assert _git(["rev-parse", "HEAD"], root) == second_sha
        assert (root / "file.txt").read_text() == "two\n"


class TestEnsureCustomNode:
    def test_node_cloned_into_custom_nodes(self, tmp_path: Path, fake_upstream):
        upstream, first_sha, _second_sha = fake_upstream
        comfy_root = tmp_path / "ComfyUI"
        comfy_root.mkdir()
        node = CustomNodeSpec(name="some_node", repo_url=str(upstream), ref=first_sha)
        manifest = ComfyEnvironmentManifest(comfyui_repo=str(upstream), comfyui_ref=first_sha)

        EnvironmentInstaller(manifest).ensure_custom_node(comfy_root, node)

        node_path = comfy_root / "custom_nodes" / "some_node"
        assert node_path.exists()
        assert _git(["rev-parse", "HEAD"], node_path) == first_sha


class TestExtraModelPaths:
    def test_yaml_written_with_nodes_path(self, tmp_path: Path):
        comfy_root = tmp_path / "ComfyUI"
        comfy_root.mkdir()
        nodes_path = tmp_path / "hordelib" / "nodes"
        nodes_path.mkdir(parents=True)
        manifest = ComfyEnvironmentManifest(comfyui_ref="f" * 40)

        EnvironmentInstaller(manifest).write_extra_model_paths(comfy_root, nodes_path)

        content = (comfy_root / "extra_model_paths.yaml").read_text(encoding="utf-8")
        assert "custom_nodes:" in content
        assert "hordelib/nodes" in content


class TestGitHelpers:
    def test_run_git_raises_on_failure(self, tmp_path: Path):
        with pytest.raises(GitCommandError):
            _run_git(["rev-parse", "HEAD"], tmp_path)  # not a repo

    def test_run_git_missing_git_is_actionable(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """A git that is not on PATH surfaces a clear, actionable GitCommandError, not a raw OSError."""

        def _no_git(*_args: object, **_kwargs: object) -> None:
            raise FileNotFoundError(2, "No such file or directory: 'git'")

        monkeypatch.setattr(subprocess, "run", _no_git)
        with pytest.raises(GitCommandError, match="git was not found on PATH"):
            _run_git(["--version"], tmp_path)
