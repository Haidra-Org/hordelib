"""Idempotent installer that makes the on-disk ComfyUI environment match the manifest."""

import enum
import hashlib
import json
import os
import subprocess
import time
import uuid
from pathlib import Path

from filelock import FileLock
from loguru import logger

from hordelib.installation.manifest import ComfyEnvironmentManifest, CustomNodeSpec

EXTRA_MODEL_PATHS_YAML = """
hordelib:
    base_path: {base_path}
    custom_nodes: {custom_nodes_path}
"""
"""Registers hordelib's first-party nodes directory with ComfyUI."""

LOCK_TIMEOUT_SECONDS = 900
"""How long to wait for another process to finish installing the same directory before giving up.

Generous enough to cover a cold clone of ComfyUI over a slow link, bounded so a lock that somehow
never releases surfaces as an error instead of hanging the process forever.
"""

ENV_MARKER_VERSION = 1
"""Bump when the completion marker's fields change so older markers are treated as invalid."""


def _write_atomic(path: Path, content: str) -> None:
    """Write text so any concurrent reader sees either the previous file or the whole new one.

    The temporary file is created in the destination directory so the rename stays within one
    filesystem, which is what makes it atomic.
    """
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex[:8]}.tmp"
    try:
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _default_nodes_path() -> Path:
    """The directory holding hordelib's first-party ComfyUI nodes."""
    from hordelib.config_path import get_hordelib_path

    return get_hordelib_path() / "nodes"


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


def _target_lock(target: Path) -> FileLock:
    """Return an OS-level, cross-process lock guarding installation of ``target``.

    Several processes of a worker pool can call the installer against one shared environment
    directory at the same time. Without serialisation they clone and check out over each other,
    and the loser leaves a half-written tree that no later clone can recover from. The lock file
    sits beside the target so each target (ComfyUI, each custom node) serialises independently.

    The lock is held by the operating system on an open file handle, so it is released when the
    holding process dies for any reason; a lock file left behind by a killed process is not stale.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    return FileLock(str(target.parent / f"{target.name}.lock"), timeout=LOCK_TIMEOUT_SECONDS)


def _env_lock(comfyui_root: Path) -> FileLock:
    """Return the lock serialising a whole-environment install, beside the per-target locks."""
    comfyui_root.parent.mkdir(parents=True, exist_ok=True)
    return FileLock(
        str(comfyui_root.parent / f"{comfyui_root.name}.ensure.lock"),
        timeout=LOCK_TIMEOUT_SECONDS,
    )


def _env_marker_path(comfyui_root: Path) -> Path:
    """Where the completion marker for an environment lives.

    Beside the environment rather than inside it, because ComfyUI loads every directory under
    ``custom_nodes`` and the marker must not be mistaken for content.
    """
    return comfyui_root.parent / f"{comfyui_root.name}.ensure.json"


def _env_targets(comfyui_root: Path, manifest: ComfyEnvironmentManifest) -> list[tuple[str, Path, str]]:
    """Every checkout the manifest declares, as ``(marker key, path, pinned ref)``."""
    targets = [("ComfyUI", comfyui_root, manifest.comfyui_ref)]
    targets += [
        (f"custom_nodes/{node.name}", comfyui_root / "custom_nodes" / node.name, node.ref)
        for node in manifest.custom_nodes
    ]
    return targets


def _env_state_digest(manifest: ComfyEnvironmentManifest, nodes_path: Path) -> str:
    """Digest the environment the manifest asks for: every repository, its pin, and the nodes path.

    The nodes path is included because it is baked into ``extra_model_paths.yaml``; a hordelib that
    moved must re-stamp that file even though no repository pin changed.
    """
    payload = json.dumps(
        {
            "comfyui": [manifest.comfyui_repo, manifest.comfyui_ref],
            "custom_nodes": sorted([node.name, node.repo_url, node.ref] for node in manifest.custom_nodes),
            "nodes_path": nodes_path.as_posix(),
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _env_is_current(comfyui_root: Path, manifest: ComfyEnvironmentManifest, nodes_path: Path) -> bool:
    """Whether a completion marker vouches for the environment the manifest currently asks for.

    The marker is rejected if it is missing, unreadable, written by a different marker version,
    digests a different manifest or nodes path, names a checkout that is no longer on disk, or
    names one whose HEAD has since moved off the pin.

    That last check is why this reads each target's HEAD. A marker records what an install
    established, which is not the same as what is there now: a checkout can be moved off its pin
    afterwards by anything with access to the directory. Every start converging the checkouts back
    onto the manifest's pins is a guarantee the installer holds regardless of the marker, so the
    marker may only shorten work that would have been a no-op. Reading HEAD costs one read-only git
    call per target and never mutates anything, so a process with nothing to do still does no
    fetching, cloning or checking out.
    """
    try:
        recorded = json.loads(_env_marker_path(comfyui_root).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False

    if not isinstance(recorded, dict):
        return False
    if recorded.get("marker_version") != ENV_MARKER_VERSION:
        return False
    if recorded.get("state_digest") != _env_state_digest(manifest, nodes_path):
        return False

    installed = recorded.get("installed_commits")
    if not isinstance(installed, dict):
        return False

    for key, target, ref in _env_targets(comfyui_root, manifest):
        if installed.get(key) != ref:
            return False
        if not (target / ".git").exists():
            return False
        if _head_commit(target) != ref:
            return False

    return (comfyui_root / "extra_model_paths.yaml").exists()


def _write_env_marker(comfyui_root: Path, manifest: ComfyEnvironmentManifest, nodes_path: Path) -> None:
    """Record that this environment matches the manifest, so later starts can skip the work."""
    marker = {
        "marker_version": ENV_MARKER_VERSION,
        "state_digest": _env_state_digest(manifest, nodes_path),
        "installed_commits": {key: _head_commit(target) for key, target, _ref in _env_targets(comfyui_root, manifest)},
    }
    _write_atomic(_env_marker_path(comfyui_root), json.dumps(marker, indent=4, sort_keys=True) + "\n")


class _TargetState(enum.Enum):
    """How a clone target on disk relates to the repository it is supposed to hold."""

    ABSENT = "absent"
    """Nothing there, or an empty directory: a clone can proceed."""

    USABLE = "usable"
    """A git checkout rooted at the target, with a resolvable HEAD and the expected remote."""

    UNUSABLE = "unusable"
    """Occupied by something a clone cannot land on and a checkout cannot recover."""


def _normalise_repo_url(url: str) -> str:
    """Reduce a repository URL or local path to a form two spellings of the same remote share."""
    cleaned = url.strip().replace("\\", "/").rstrip("/")
    if cleaned.endswith(".git"):
        cleaned = cleaned[: -len(".git")]
    return cleaned.casefold()


def _classify_target(target: Path, repo_url: str) -> tuple[_TargetState, str]:
    """Classify a clone target, returning its state and a reason when it is unusable.

    ``git clone`` refuses a non-empty destination, so a directory holding anything other than a
    healthy checkout of the expected remote is a permanent failure unless it is moved out of the
    way. Everything that is not recognisably the right repository is therefore reported unusable.
    """
    if not target.exists():
        return _TargetState.ABSENT, ""
    if not target.is_dir():
        return _TargetState.UNUSABLE, "path exists but is not a directory"
    if not any(target.iterdir()):
        return _TargetState.ABSENT, ""

    try:
        toplevel = _run_git(["rev-parse", "--show-toplevel"], target)
    except (GitCommandError, OSError):
        return _TargetState.UNUSABLE, "directory is not empty and is not a git checkout"

    try:
        is_own_repo = Path(toplevel).resolve() == target.resolve()
    except OSError:
        is_own_repo = False
    if not is_own_repo:
        # A directory nested inside some other repository resolves to that repository's root; it is
        # loose content as far as this target is concerned.
        return _TargetState.UNUSABLE, "directory is not the root of its own git checkout"

    if _head_commit(target) is None:
        return _TargetState.UNUSABLE, "git checkout has no resolvable HEAD"

    try:
        origin = _run_git(["remote", "get-url", "origin"], target)
    except (GitCommandError, OSError):
        return _TargetState.UNUSABLE, "git checkout has no origin remote"

    if _normalise_repo_url(origin) != _normalise_repo_url(repo_url):
        return _TargetState.UNUSABLE, f"git checkout tracks {origin}, expected {repo_url}"

    return _TargetState.USABLE, ""


def _move_aside(target: Path, reason: str, quarantine_parent: Path) -> Path:
    """Move an unusable target under ``quarantine_parent`` so a fresh clone can take its place.

    The directory is moved rather than deleted: it may hold content the operator put there, and a
    silent recursive delete of a path the caller chose is not the installer's to make. It does not
    go beside the target, because ComfyUI loads every directory under ``custom_nodes`` as a node
    and would try to import whatever was moved.
    """
    quarantine_parent.mkdir(parents=True, exist_ok=True)
    quarantined = quarantine_parent / f"{target.name}.unusable-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    logger.warning(
        "Moving unusable install directory aside so it can be re-cloned: target={}, moved_to={}, reason={}",
        target,
        quarantined,
        reason,
    )
    try:
        target.rename(quarantined)
    except OSError as exc:
        raise RuntimeError(
            f"Could not move the unusable directory {target} aside to {quarantined}: {exc}. "
            f"It is unusable because: {reason}. Move or delete it manually, then retry.",
        ) from exc
    return quarantined


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

        Several processes of a worker pool call this against one shared environment as they start.
        They are serialised on an environment-wide lock, and the one that completes the install
        leaves a marker describing what it established, so the processes behind it acquire the
        lock, recognise the environment as already matching the manifest, and return without
        cloning, fetching or checking anything out. The lock is per environment rather than held
        across each process's git work in turn, so a warm start does not queue every process
        behind the others.
        """
        if hordelib_nodes_path is None:
            hordelib_nodes_path = _default_nodes_path()

        with _env_lock(comfyui_root):
            if _env_is_current(comfyui_root, self.manifest, hordelib_nodes_path):
                logger.debug("ComfyUI environment already matches the manifest: root={}", comfyui_root)
                return

            self.ensure_comfyui(comfyui_root)
            for node in self.manifest.custom_nodes:
                self.ensure_custom_node(comfyui_root, node)
            self.write_extra_model_paths(comfyui_root, hordelib_nodes_path)
            _write_env_marker(comfyui_root, self.manifest, hordelib_nodes_path)

    def _sync_repo(self, target: Path, repo_url: str, ref: str, label: str, comfyui_root: Path) -> None:
        """Bring one checkout to ``ref``, serialised against other processes sharing ``target``.

        Whichever process wins the lock does the work; the others wait, then re-classify the target
        and find the work already done.
        """
        with _target_lock(target):
            state, reason = _classify_target(target, repo_url)

            if state is _TargetState.UNUSABLE:
                _move_aside(target, reason, comfyui_root.parent)
                # The environment no longer matches whatever a marker claims about it.
                _env_marker_path(comfyui_root).unlink(missing_ok=True)
                state = _TargetState.ABSENT

            if state is _TargetState.ABSENT:
                _clone_at_ref(repo_url, ref, target)
            else:
                current = _head_commit(target)
                if current == ref:
                    logger.debug("{} already at pinned commit: ref={}", label, ref[:8])
                    return

                logger.info(
                    "{} commit {} does not match pinned {}; updating",
                    label,
                    (current or "unknown")[:8],
                    ref[:8],
                )
                _checkout_ref(target, ref)

            verified = _head_commit(target)
            if verified != ref:
                raise RuntimeError(
                    f"{label} checkout verification failed: HEAD is {verified}, expected {ref}",
                )

    def ensure_comfyui(self, comfyui_root: Path) -> None:
        """Clone or update the ComfyUI checkout to the pinned commit."""
        self._sync_repo(
            comfyui_root,
            self.manifest.comfyui_repo,
            self.manifest.comfyui_ref,
            "ComfyUI",
            comfyui_root,
        )

    def ensure_custom_node(self, comfyui_root: Path, node: CustomNodeSpec) -> None:
        """Clone or update a single external custom node to its pinned commit."""
        node_path = comfyui_root / "custom_nodes" / node.name
        self._sync_repo(
            node_path,
            node.repo_url,
            node.ref,
            f"Custom node {node.name}",
            comfyui_root,
        )

    def write_extra_model_paths(self, comfyui_root: Path, hordelib_nodes_path: Path | None = None) -> None:
        """Write ComfyUI's ``extra_model_paths.yaml`` registering hordelib's first-party nodes."""
        if hordelib_nodes_path is None:
            hordelib_nodes_path = _default_nodes_path()

        config_file = comfyui_root / "extra_model_paths.yaml"
        content = EXTRA_MODEL_PATHS_YAML.format(
            base_path=hordelib_nodes_path.parent.parent.as_posix(),
            custom_nodes_path=Path(
                os.path.relpath(hordelib_nodes_path, hordelib_nodes_path.parent.parent),
            ).as_posix(),
        )
        # Written by replacement, not in place: sibling processes install concurrently, and ComfyUI
        # reads this file, so a reader must never catch it half-written.
        _write_atomic(config_file, content)
        logger.debug("Wrote extra_model_paths.yaml: path={}", config_file)

    def installed_comfyui_commit(self, comfyui_root: Path) -> str | None:
        """Return the currently installed ComfyUI commit, or None if not installed."""
        return _head_commit(comfyui_root)
