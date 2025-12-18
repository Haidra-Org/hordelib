# install.py
# Fetch a specific version of the upstream ComfyUI project
import os
import subprocess
from pathlib import Path

from loguru import logger

from hordelib.config_path import get_comfyui_path, get_hordelib_path
from hordelib.consts import RELEASE_VERSION

CUSTOM_NODE_YAML = """
hordelib:
    base_path: ../
    custom_nodes: hordelib/nodes
"""


class Installer:
    """Handles the installation of ComfyUI."""

    @classmethod
    def get_commit_hash(cls) -> str:
        if RELEASE_VERSION:
            return ""
        head_file = get_comfyui_path() / ".git" / "HEAD"
        if not head_file.exists():
            return "NOT FOUND"
        try:
            with open(head_file) as f:
                head_contents = f.read().strip()

            if not head_contents.startswith("ref:"):
                return head_contents

            ref_path = os.path.join(".git", *head_contents[5:].split("/"))

            with open(ref_path) as f:
                return f.read().strip()

        except Exception:
            return ""

    @classmethod
    def _run_get_result(cls, command, directory=None):
        if directory is None:
            directory = get_hordelib_path()
        # Don't if we're a release version
        if RELEASE_VERSION:
            return None
        return subprocess.run(
            command,
            shell=True,
            text=True,
            capture_output=True,
            cwd=str(directory),
        )

    @classmethod
    def _run(cls, command, directory=None) -> tuple[bool, str] | None:
        if directory is None:
            directory = get_hordelib_path()
        # Don't if we're a release version
        if RELEASE_VERSION:
            return None
        try:
            result = cls._run_get_result(command, str(directory))
        except Exception as Ex:
            logger.error(Ex)
            return None
        if result.returncode:
            logger.error(result.stderr)
            return None
        return (True, result.stdout)

    @classmethod
    def install(cls, comfy_version: str) -> None:
        # Don't if we're a release version
        if RELEASE_VERSION:
            return
        # Install if ComfyUI is missing completely
        if not get_comfyui_path().exists():
            install_dir = get_comfyui_path().parent
            cls._run(
                "git clone https://github.com/comfyanonymous/ComfyUI.git",
                install_dir,
            )
            cls._run(f"git checkout {comfy_version}", str(get_comfyui_path()))
            # Apply our patches to comfyui
            cls.apply_patch(get_hordelib_path() / "install_comfy.patch")
            return

        # If it's installed, is it up to date?
        version = cls.get_commit_hash()
        if version == comfy_version:
            # Yes, all done
            return

        # Update comfyui
        logger.info(
            f"Current ComfyUI version {version[:8]} requires {comfy_version[:8]}",
        )
        cls.reset_comfyui_to_version(comfy_version)
        # Apply our patches to comfyui
        cls.apply_patch(get_hordelib_path() / "install_comfy.patch")

    @classmethod
    def remove_local_comfyui_changes(cls):
        # Don't if we're a release version
        if RELEASE_VERSION:
            return
        cls._run("git reset --hard", get_comfyui_path())
        cls._run("git clean -fd", get_comfyui_path())

    @classmethod
    def reset_comfyui_to_version(cls, comfy_version):
        # Don't if we're a release version
        if RELEASE_VERSION:
            return
        # Try hard to ensure we reset everything even if we have been
        # hacking on ComfyUI or are in a weird repo state
        cls.remove_local_comfyui_changes()
        cls._run("git checkout master", get_comfyui_path())
        cls._run("git pull", get_comfyui_path())
        cls._run(f"git checkout {comfy_version}", get_comfyui_path())

    @classmethod
    def apply_patch(cls, patch_file: Path, skip_dot_patch: bool = True):
        # Don't if we're a release version
        if RELEASE_VERSION:
            return
        if not skip_dot_patch:  # FIXME
            # Check if the patch has already been applied
            result = cls._run_get_result(
                f"git apply --check {patch_file}",
                get_comfyui_path(),
            )
            could_apply = not result.returncode
            result = cls._run_get_result(
                f"git apply --reverse --check {patch_file}",
                get_comfyui_path(),
            )
            could_reverse = not result.returncode
            if could_apply:
                # Apply the patch
                logger.info(f"Applying patch {patch_file}")
                result = cls._run_get_result(f"git apply {patch_file}", get_comfyui_path())
                logger.debug(f"{result}")
            elif could_reverse:
                # Patch is already applied, all is well
                logger.info(f"Already applied patch {patch_file}")
            else:
                # Couldn't apply or reverse? That's not so good, but maybe we are partially applied?
                # Reset local changes
                cls.remove_local_comfyui_changes()
                # Try to apply the patch
                logger.info(f"Applying patch {patch_file}")
                result = cls._run_get_result(f"git apply {patch_file}", get_comfyui_path())
                logger.debug(f"{result}")
                if result.returncode:
                    logger.error(f"Could not apply patch {patch_file}")

        # Drop in custom node config
        config_file = os.path.join(get_comfyui_path(), "extra_model_paths.yaml")
        with open(config_file, "w") as outfile:
            outfile.write(CUSTOM_NODE_YAML)
