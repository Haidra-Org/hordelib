# install.py
# Fetch a specific version of the upstream ComfyUI project
import os
import subprocess

from loguru import logger

from hordelib.config_path import get_comfyui_path, get_hordelib_path
from hordelib.consts import RELEASE_VERSION


class Installer:
    """Handles the installation of ComfyUI."""

    @classmethod
    def get_commit_hash(cls) -> str:
        if RELEASE_VERSION:
            return ""
        head_file = os.path.join(get_comfyui_path(), ".git", "HEAD")
        if not os.path.exists(head_file):
            return "NOT FOUND"
        try:
            with open(head_file) as f:
                head_contents = f.read().strip()

            if not head_contents.startswith("ref:"):
                return head_contents

            ref_path = os.path.join(".git", *head_contents[5:].split("/"))

            with open(ref_path) as f:
                commit_hash = f.read().strip()

            return commit_hash
        except Exception:
            return ""

    @classmethod
    def _run_get_result(cls, command, directory=get_hordelib_path()):
        # Don't if we're a release version
        if RELEASE_VERSION:
            return
        return subprocess.run(
            command,
            shell=True,
            text=True,
            capture_output=True,
            cwd=directory,
        )

    @classmethod
    def _run(cls, command, directory=get_hordelib_path()) -> tuple[bool, str] | None:
        # Don't if we're a release version
        if RELEASE_VERSION:
            return None
        try:
            result = cls._run_get_result(command, directory)
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
        if not os.path.exists(get_comfyui_path()):
            installdir = os.path.dirname(get_comfyui_path())
            cls._run(
                "git clone https://github.com/comfyanonymous/ComfyUI.git",
                installdir,
            )
            cls._run(f"git checkout {comfy_version}", get_comfyui_path())
            # Apply our patches to comfyui
            cls.apply_patch(os.path.join(get_hordelib_path(), "install_comfy.patch"))
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
        cls.apply_patch(os.path.join(get_hordelib_path(), "install_comfy.patch"))

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
    def apply_patch(cls, patchfile):
        # Don't if we're a release version
        if RELEASE_VERSION:
            return
        # Check if the patch has already been applied
        result = cls._run_get_result(
            f"git apply --check {patchfile}",
            get_comfyui_path(),
        )
        could_apply = not result.returncode
        result = cls._run_get_result(
            f"git apply --reverse --check {patchfile}",
            get_comfyui_path(),
        )
        could_reverse = not result.returncode
        if could_apply:
            # Apply the patch
            logger.info(f"Applying patch {patchfile}")
            result = cls._run_get_result(f"git apply {patchfile}", get_comfyui_path())
            logger.debug(f"{result}")
        elif could_reverse:
            # Patch is already applied, all is well
            logger.info(f"Already applied patch {patchfile}")
        else:
            # Couldn't apply or reverse? That's not so good, but maybe we are partially applied?
            # Reset local changes
            cls.remove_local_comfyui_changes()
            # Try to apply the patch
            logger.info(f"Applying patch {patchfile}")
            result = cls._run_get_result(f"git apply {patchfile}", get_comfyui_path())
            logger.debug(f"{result}")
            if result.returncode:
                logger.error(f"Could not apply patch {patchfile}")
