"""Installer tests against local temp git repositories. No network or GPU required."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

import hordelib.installation.installer as installer_module
from hordelib.installation.installer import EnvironmentInstaller, GitCommandError, _run_git
from hordelib.installation.manifest import ComfyEnvironmentManifest, CustomNodeSpec

IMPORT_ROOT = Path(installer_module.__file__).parents[2]
"""The directory the ``hordelib`` package lives in, so a child process can import it."""


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


def _run_children(script_source: str, tmp_path: Path, arguments: list[str]) -> None:
    """Start two children on ``script_source``, release them together, and require clean exits."""
    child_script = tmp_path / "child.py"
    child_script.write_text(script_source, encoding="utf-8")
    go_file = tmp_path / "go"
    env = {**os.environ, "PYTHONPATH": str(IMPORT_ROOT)}

    children = [
        subprocess.Popen(
            [sys.executable, str(child_script), str(tmp_path / f"ready.{index}"), str(go_file), *arguments],
            cwd=str(tmp_path),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        for index in range(2)
    ]
    try:
        deadline = time.time() + 180
        while not all((tmp_path / f"ready.{index}").exists() for index in range(2)):
            assert time.time() < deadline, "child processes never became ready"
            assert all(child.poll() is None for child in children), "a child exited before it was ready"
            time.sleep(0.05)
        go_file.write_text("go", encoding="utf-8")

        outputs = [child.communicate(timeout=180)[0] for child in children]
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()

    for child, output in zip(children, outputs, strict=True):
        assert child.returncode == 0, output


CONCURRENT_ENSURE_CHILD = '''
"""Runs one installer against a shared target, announcing when it starts cloning."""

import pathlib
import sys
import time

ready_file, go_file, clone_log, upstream, ref, root = sys.argv[1:]

import hordelib.installation.installer as installer_module
from hordelib.installation.manifest import ComfyEnvironmentManifest

_real_clone = installer_module._clone_at_ref


def _announcing_clone(repo_url, clone_ref, target):
    with open(clone_log, "a", encoding="utf-8") as handle:
        handle.write("clone\\n")
    # Hold the critical section open long enough that an unserialised sibling would enter it too.
    time.sleep(0.5)
    _real_clone(repo_url, clone_ref, target)


installer_module._clone_at_ref = _announcing_clone

pathlib.Path(ready_file).write_text("ready", encoding="utf-8")
deadline = time.time() + 120
while not pathlib.Path(go_file).exists():
    if time.time() > deadline:
        raise SystemExit("timed out waiting for the start signal")
    time.sleep(0.01)

manifest = ComfyEnvironmentManifest(comfyui_repo=upstream, comfyui_ref=ref)
installer_module.EnvironmentInstaller(manifest).ensure_comfyui(pathlib.Path(root))
'''


class TestConcurrentEnsure:
    def test_two_processes_sharing_a_target_produce_one_checkout(self, tmp_path: Path, fake_upstream):
        """Worker children cold-starting together must not clone over each other."""
        upstream, first_sha, _second_sha = fake_upstream
        root = tmp_path / "ComfyUI"
        clone_log = tmp_path / "clones.log"

        _run_children(
            CONCURRENT_ENSURE_CHILD,
            tmp_path,
            [str(clone_log), str(upstream), first_sha, str(root)],
        )

        assert clone_log.read_text(encoding="utf-8").count("clone") == 1
        assert _git(["rev-parse", "HEAD"], root) == first_sha


CONCURRENT_FULL_ENSURE_CHILD = '''
"""Runs a full environment install, recording every git command this process issues."""

import os
import pathlib
import sys
import time

ready_file, go_file, git_log, upstream, ref, root = sys.argv[1:]

import hordelib.installation.installer as installer_module
from hordelib.installation.manifest import ComfyEnvironmentManifest, CustomNodeSpec

_real_run_git = installer_module._run_git
_real_clone = installer_module._clone_at_ref


def _recording_run_git(args, cwd):
    with open(git_log, "a", encoding="utf-8") as handle:
        handle.write(f"{os.getpid()} {args[0]}\\n")
    return _real_run_git(args, cwd)


def _slow_clone(repo_url, clone_ref, target):
    # Hold the critical section open long enough that an unserialised sibling would enter it too.
    time.sleep(0.5)
    _real_clone(repo_url, clone_ref, target)


installer_module._run_git = _recording_run_git
installer_module._clone_at_ref = _slow_clone

pathlib.Path(ready_file).write_text("ready", encoding="utf-8")
deadline = time.time() + 120
while not pathlib.Path(go_file).exists():
    if time.time() > deadline:
        raise SystemExit("timed out waiting for the start signal")
    time.sleep(0.01)

manifest = ComfyEnvironmentManifest(
    comfyui_repo=upstream,
    comfyui_ref=ref,
    custom_nodes=[CustomNodeSpec(name="some_node", repo_url=upstream, ref=ref)],
)
installer_module.EnvironmentInstaller(manifest).ensure(pathlib.Path(root))
'''


MUTATING_GIT_COMMANDS = frozenset({"clone", "checkout", "reset", "fetch", "pull", "merge"})
"""Git commands that change a checkout, as opposed to reading its state."""


class TestEnsureCompletionMarker:
    def test_concurrent_cold_start_leaves_the_loser_no_mutating_git_work(self, tmp_path: Path, fake_upstream):
        """Only the process that wins the environment lock installs; the other sees it done."""
        upstream, first_sha, _second_sha = fake_upstream
        root = tmp_path / "ComfyUI"
        git_log = tmp_path / "git.log"

        _run_children(
            CONCURRENT_FULL_ENSURE_CHILD,
            tmp_path,
            [str(git_log), str(upstream), first_sha, str(root)],
        )

        issued = [line.split() for line in git_log.read_text(encoding="utf-8").splitlines()]
        mutating_pids = {pid for pid, command in issued if command in MUTATING_GIT_COMMANDS}
        assert len(mutating_pids) == 1, "both processes installed into the same environment"
        assert _git(["rev-parse", "HEAD"], root) == first_sha
        assert _git(["rev-parse", "HEAD"], root / "custom_nodes" / "some_node") == first_sha
        assert (tmp_path / "ComfyUI.ensure.json").exists()

    def test_warm_start_with_valid_marker_does_no_mutating_git_work(
        self,
        tmp_path: Path,
        fake_upstream,
        monkeypatch: pytest.MonkeyPatch,
    ):
        upstream, first_sha, _second_sha = fake_upstream
        root = tmp_path / "ComfyUI"
        nodes_path = tmp_path / "hordelib" / "nodes"
        nodes_path.mkdir(parents=True)
        manifest = ComfyEnvironmentManifest(
            comfyui_repo=str(upstream),
            comfyui_ref=first_sha,
            custom_nodes=[CustomNodeSpec(name="some_node", repo_url=str(upstream), ref=first_sha)],
        )
        EnvironmentInstaller(manifest).ensure(root, hordelib_nodes_path=nodes_path)

        real_run_git = installer_module._run_git

        def _reject_mutating_git(args: list[str], cwd: Path) -> str:
            assert args[0] not in MUTATING_GIT_COMMANDS, f"a warm start must not run git {args[0]}"
            return real_run_git(args, cwd)

        monkeypatch.setattr(installer_module, "_run_git", _reject_mutating_git)

        EnvironmentInstaller(manifest).ensure(root, hordelib_nodes_path=nodes_path)

    def test_target_moved_off_its_pin_resyncs_despite_a_fresh_marker(self, tmp_path: Path, fake_upstream):
        """Every start converges the checkouts onto the manifest, marker or not."""
        upstream, first_sha, second_sha = fake_upstream
        root = tmp_path / "ComfyUI"
        nodes_path = tmp_path / "hordelib" / "nodes"
        nodes_path.mkdir(parents=True)
        manifest = ComfyEnvironmentManifest(comfyui_repo=str(upstream), comfyui_ref=second_sha)
        EnvironmentInstaller(manifest).ensure(root, hordelib_nodes_path=nodes_path)
        marker_before = (tmp_path / "ComfyUI.ensure.json").read_text(encoding="utf-8")

        _git(["checkout", "--force", first_sha], root)
        assert _git(["rev-parse", "HEAD"], root) == first_sha

        EnvironmentInstaller(manifest).ensure(root, hordelib_nodes_path=nodes_path)

        assert _git(["rev-parse", "HEAD"], root) == second_sha
        assert (tmp_path / "ComfyUI.ensure.json").read_text(encoding="utf-8") == marker_before

    def test_changed_pin_invalidates_the_marker(self, tmp_path: Path, fake_upstream):
        upstream, first_sha, second_sha = fake_upstream
        root = tmp_path / "ComfyUI"
        nodes_path = tmp_path / "hordelib" / "nodes"
        nodes_path.mkdir(parents=True)
        EnvironmentInstaller(
            ComfyEnvironmentManifest(comfyui_repo=str(upstream), comfyui_ref=first_sha),
        ).ensure(root, hordelib_nodes_path=nodes_path)

        EnvironmentInstaller(
            ComfyEnvironmentManifest(comfyui_repo=str(upstream), comfyui_ref=second_sha),
        ).ensure(root, hordelib_nodes_path=nodes_path)

        assert _git(["rev-parse", "HEAD"], root) == second_sha
        marker = json.loads((tmp_path / "ComfyUI.ensure.json").read_text(encoding="utf-8"))
        assert marker["installed_commits"]["ComfyUI"] == second_sha

    def test_corrupt_marker_is_ignored_and_rewritten(self, tmp_path: Path, fake_upstream):
        upstream, first_sha, _second_sha = fake_upstream
        root = tmp_path / "ComfyUI"
        nodes_path = tmp_path / "hordelib" / "nodes"
        nodes_path.mkdir(parents=True)
        manifest = ComfyEnvironmentManifest(comfyui_repo=str(upstream), comfyui_ref=first_sha)
        EnvironmentInstaller(manifest).ensure(root, hordelib_nodes_path=nodes_path)
        (tmp_path / "ComfyUI.ensure.json").write_text("{not json", encoding="utf-8")

        EnvironmentInstaller(manifest).ensure(root, hordelib_nodes_path=nodes_path)

        marker = json.loads((tmp_path / "ComfyUI.ensure.json").read_text(encoding="utf-8"))
        assert marker["installed_commits"]["ComfyUI"] == first_sha

    def test_marker_is_dropped_when_a_target_is_repaired(self, tmp_path: Path, fake_upstream):
        upstream, first_sha, _second_sha = fake_upstream
        root = tmp_path / "ComfyUI"
        nodes_path = tmp_path / "hordelib" / "nodes"
        nodes_path.mkdir(parents=True)
        node = CustomNodeSpec(name="some_node", repo_url=str(upstream), ref=first_sha)
        manifest = ComfyEnvironmentManifest(
            comfyui_repo=str(upstream),
            comfyui_ref=first_sha,
            custom_nodes=[node],
        )
        installer = EnvironmentInstaller(manifest)
        installer.ensure(root, hordelib_nodes_path=nodes_path)
        marker_path = tmp_path / "ComfyUI.ensure.json"
        assert marker_path.exists()

        node_path = root / "custom_nodes" / "some_node"
        (node_path / ".git").rename(node_path / "not-a-git-dir")

        installer.ensure_custom_node(root, node)

        assert not marker_path.exists(), "a repaired environment must not keep vouching for itself"


class TestExtraModelPathsAtomicity:
    def test_failed_write_leaves_the_previous_file_intact(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        comfy_root = tmp_path / "ComfyUI"
        comfy_root.mkdir()
        nodes_path = tmp_path / "hordelib" / "nodes"
        nodes_path.mkdir(parents=True)
        config_file = comfy_root / "extra_model_paths.yaml"
        config_file.write_text("previous contents\n", encoding="utf-8")

        def _failing_replace(*_args: object, **_kwargs: object) -> None:
            raise OSError("replace refused")

        monkeypatch.setattr(installer_module.os, "replace", _failing_replace)

        with pytest.raises(OSError, match="replace refused"):
            EnvironmentInstaller(
                ComfyEnvironmentManifest(comfyui_ref="f" * 40),
            ).write_extra_model_paths(comfy_root, nodes_path)

        assert config_file.read_text(encoding="utf-8") == "previous contents\n"
        assert list(comfy_root.glob(".extra_model_paths.yaml.*.tmp")) == []


class TestUnusableTargetRepair:
    def test_non_git_content_is_moved_aside_and_recloned(self, tmp_path: Path, fake_upstream):
        upstream, first_sha, _second_sha = fake_upstream
        root = tmp_path / "ComfyUI"
        root.mkdir()
        (root / "stray.txt").write_text("left behind by a failed install\n")
        manifest = ComfyEnvironmentManifest(comfyui_repo=str(upstream), comfyui_ref=first_sha)

        EnvironmentInstaller(manifest).ensure_comfyui(root)

        assert _git(["rev-parse", "HEAD"], root) == first_sha
        moved = list(tmp_path.glob("ComfyUI.unusable-*"))
        assert len(moved) == 1
        assert (moved[0] / "stray.txt").exists(), "the displaced content must be kept, not deleted"

    def test_clone_succeeded_checkout_failed_state_is_repaired(self, tmp_path: Path, fake_upstream):
        """A git directory whose worktree was never checked out cannot be cloned into or reset."""
        upstream, first_sha, _second_sha = fake_upstream
        root = tmp_path / "ComfyUI"
        (root / ".git").mkdir(parents=True)
        (root / "file.txt").write_text("untracked conflict\n")
        manifest = ComfyEnvironmentManifest(comfyui_repo=str(upstream), comfyui_ref=first_sha)

        EnvironmentInstaller(manifest).ensure_comfyui(root)

        assert _git(["rev-parse", "HEAD"], root) == first_sha
        assert len(list(tmp_path.glob("ComfyUI.unusable-*"))) == 1

    def test_checkout_of_a_different_remote_is_moved_aside(self, tmp_path: Path, fake_upstream):
        upstream, first_sha, _second_sha = fake_upstream
        other = tmp_path / "other-upstream"
        _git(["clone", str(upstream), str(other)], tmp_path)
        root = tmp_path / "ComfyUI"
        EnvironmentInstaller(
            ComfyEnvironmentManifest(comfyui_repo=str(other), comfyui_ref=first_sha),
        ).ensure_comfyui(root)

        EnvironmentInstaller(
            ComfyEnvironmentManifest(comfyui_repo=str(upstream), comfyui_ref=first_sha),
        ).ensure_comfyui(root)

        assert _git(["remote", "get-url", "origin"], root) == str(upstream)
        assert len(list(tmp_path.glob("ComfyUI.unusable-*"))) == 1

    def test_healthy_checkout_is_neither_recloned_nor_moved(
        self,
        tmp_path: Path,
        fake_upstream,
        monkeypatch: pytest.MonkeyPatch,
    ):
        upstream, first_sha, second_sha = fake_upstream
        root = tmp_path / "ComfyUI"
        EnvironmentInstaller(
            ComfyEnvironmentManifest(comfyui_repo=str(upstream), comfyui_ref=first_sha),
        ).ensure_comfyui(root)

        def _fail_if_cloned(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("an existing healthy checkout must not be re-cloned")

        monkeypatch.setattr(installer_module, "_clone_at_ref", _fail_if_cloned)

        EnvironmentInstaller(
            ComfyEnvironmentManifest(comfyui_repo=str(upstream), comfyui_ref=first_sha),
        ).ensure_comfyui(root)
        EnvironmentInstaller(
            ComfyEnvironmentManifest(comfyui_repo=str(upstream), comfyui_ref=second_sha),
        ).ensure_comfyui(root)

        assert _git(["rev-parse", "HEAD"], root) == second_sha
        assert list(tmp_path.glob("ComfyUI.unusable-*")) == []

    def test_unusable_custom_node_is_repaired(self, tmp_path: Path, fake_upstream):
        upstream, first_sha, _second_sha = fake_upstream
        comfy_root = tmp_path / "ComfyUI"
        node_path = comfy_root / "custom_nodes" / "some_node"
        node_path.mkdir(parents=True)
        (node_path / "leftover.py").write_text("# partial clone\n")
        node = CustomNodeSpec(name="some_node", repo_url=str(upstream), ref=first_sha)
        manifest = ComfyEnvironmentManifest(comfyui_repo=str(upstream), comfyui_ref=first_sha)

        EnvironmentInstaller(manifest).ensure_custom_node(comfy_root, node)

        assert _git(["rev-parse", "HEAD"], node_path) == first_sha
        # ComfyUI imports every directory under custom_nodes, so quarantined content cannot stay there.
        assert list((comfy_root / "custom_nodes").glob("some_node.unusable-*")) == []
        moved = list(tmp_path.glob("some_node.unusable-*"))
        assert len(moved) == 1
        assert (moved[0] / "leftover.py").exists()


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
