"""Idempotent installer that makes the on-disk ComfyUI environment match the manifest."""

import os
import subprocess
from pathlib import Path

from loguru import logger

from hordelib.installation.manifest import ComfyEnvironmentManifest, CustomNodeSpec

EXTRA_MODEL_PATHS_YAML = """
hordelib:
    base_path: {base_path}
    custom_nodes: {custom_nodes_path}
"""
"""Registers hordelib's first-party nodes directory with ComfyUI."""


class GitCommandError(RuntimeError):
    """A git command run by the installer failed."""


def _run_git(args: list[str], cwd: Path) -> str:
    """Run a git command, returning stripped stdout.

    Raises:
        GitCommandError: If git is not installed/on PATH, or exits non-zero.
    """
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            text=True,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
        )
    except FileNotFoundError as exc:
        # A bare-name "git" that the OS cannot find raises FileNotFoundError, not a non-zero exit. Turn it
        # into the same actionable error all callers already handle, naming why git is needed.
        raise GitCommandError(
            "git was not found on PATH. hordelib needs git to fetch and pin ComfyUI and its custom nodes; "
            "install git (https://git-scm.com/downloads) and make sure it is on PATH, then retry.",
        ) from exc
    if result.returncode != 0:
        raise GitCommandError(f"git {' '.join(args)} failed in {cwd}: {result.stderr.strip()}")
    return result.stdout.strip()


def _head_commit(repo_path: Path) -> str | None:
    """Return the HEAD commit SHA of a repo, or None if it isn't a usable git repo."""
    try:
        return _run_git(["rev-parse", "HEAD"], repo_path)
    except (GitCommandError, OSError):
        return None


def _clone_at_ref(repo_url: str, ref: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Cloning repository: url={}, target={}", repo_url, target)
    _run_git(["clone", repo_url, target.name], target.parent)
    _run_git(["checkout", "--force", ref], target)


def _checkout_ref(repo_path: Path, ref: str) -> None:
    """Fetch (if needed) and check out an exact commit, discarding local changes."""
    try:
        # Cheap path: the commit may already be present locally
        _run_git(["cat-file", "-e", f"{ref}^{{commit}}"], repo_path)
    except GitCommandError:
        logger.info("Fetching to obtain pinned commit: repo={}, ref={}", repo_path.name, ref[:8])
        _run_git(["fetch", "--all", "--tags"], repo_path)

    _run_git(["reset", "--hard"], repo_path)
    _run_git(["checkout", "--force", ref], repo_path)


class EnvironmentInstaller:
    """Installs and pins ComfyUI and external custom nodes per the manifest."""

    def __init__(self, manifest: ComfyEnvironmentManifest):
        self.manifest = manifest

    def ensure(self, comfyui_root: Path, *, hordelib_nodes_path: Path | None = None) -> None:
        """Make the environment at ``comfyui_root`` match the manifest. Idempotent.

        Args:
            comfyui_root: The directory that contains (or will contain) the ComfyUI checkout.
            hordelib_nodes_path: The directory of hordelib's first-party nodes, registered with
                ComfyUI via ``extra_model_paths.yaml``. Defaults to ``hordelib/nodes``.
        """
        self.ensure_comfyui(comfyui_root)
        for node in self.manifest.custom_nodes:
            self.ensure_custom_node(comfyui_root, node)
        self.write_extra_model_paths(comfyui_root, hordelib_nodes_path)

    def ensure_comfyui(self, comfyui_root: Path) -> None:
        """Clone or update the ComfyUI checkout to the pinned commit."""
        target_ref = self.manifest.comfyui_ref

        if not comfyui_root.exists():
            _clone_at_ref(self.manifest.comfyui_repo, target_ref, comfyui_root)
            return

        current = _head_commit(comfyui_root)
        if current == target_ref:
            logger.debug("ComfyUI already at pinned commit: ref={}", target_ref[:8])
            return

        logger.info(
            "ComfyUI commit {} does not match pinned {}; updating",
            (current or "unknown")[:8],
            target_ref[:8],
        )
        _checkout_ref(comfyui_root, target_ref)

        verified = _head_commit(comfyui_root)
        if verified != target_ref:
            raise RuntimeError(
                f"ComfyUI checkout verification failed: HEAD is {verified}, expected {target_ref}",
            )

    def ensure_custom_node(self, comfyui_root: Path, node: CustomNodeSpec) -> None:
        """Clone or update a single external custom node to its pinned commit."""
        node_path = comfyui_root / "custom_nodes" / node.name

        if not node_path.exists():
            _clone_at_ref(node.repo_url, node.ref, node_path)
            return

        current = _head_commit(node_path)
        if current == node.ref:
            logger.debug("Custom node already at pinned commit: node={}, ref={}", node.name, node.ref[:8])
            return

        logger.info(
            "Custom node {} commit {} does not match pinned {}; updating",
            node.name,
            (current or "unknown")[:8],
            node.ref[:8],
        )
        _checkout_ref(node_path, node.ref)

        verified = _head_commit(node_path)
        if verified != node.ref:
            raise RuntimeError(
                f"Custom node {node.name} checkout verification failed: HEAD is {verified}, expected {node.ref}",
            )

    def write_extra_model_paths(self, comfyui_root: Path, hordelib_nodes_path: Path | None = None) -> None:
        """Write ComfyUI's ``extra_model_paths.yaml`` registering hordelib's first-party nodes."""
        if hordelib_nodes_path is None:
            from hordelib.config_path import get_hordelib_path

            hordelib_nodes_path = get_hordelib_path() / "nodes"

        config_file = comfyui_root / "extra_model_paths.yaml"
        content = EXTRA_MODEL_PATHS_YAML.format(
            base_path=hordelib_nodes_path.parent.parent.as_posix(),
            custom_nodes_path=Path(
                os.path.relpath(hordelib_nodes_path, hordelib_nodes_path.parent.parent),
            ).as_posix(),
        )
        config_file.write_text(content, encoding="utf-8")
        logger.debug("Wrote extra_model_paths.yaml: path={}", config_file)

    def installed_comfyui_commit(self, comfyui_root: Path) -> str | None:
        """Return the currently installed ComfyUI commit, or None if not installed."""
        return _head_commit(comfyui_root)
