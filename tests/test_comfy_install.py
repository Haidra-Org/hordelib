# test_comfy_install.py
import os

from hordelib import COMFYUI_VERSION
from hordelib.install import Installer


class TestSetup:
    def test_get_hash(self):
        install = Installer()
        comfyhash = install.get_commit_hash()
        assert comfyhash == COMFYUI_VERSION

    def test_head_not_found(self):
        install = Installer()
        install.ourdir = "does-not-exist"
        comfyhash = install.get_commit_hash()
        assert comfyhash == "NOT FOUND"

    def test_run_without_subdir(self):
        install = Installer()
        result = install._run(["echo", "1"])
        assert result
        assert result[0] is True

    def test_run_with_subdir(self):
        install = Installer()
        result = install._run(["echo", "1"], "nodes")
        assert result
        assert result[0] is True

    def test_run_with_missing_subdir(self):
        install = Installer()
        result = install._run(["echo", "1"], "does-not-exist")
        assert result is None

    def test_bad_exitcode(self):
        # XXX This test doesn't work on github for some reason
        if os.getenv("HORDELIB_TESTING", "") == "no-cuda":
            return
        install = Installer()
        result = install._run(["exit", "1"])
        assert result is None

    def test_comfy_install(self):
        # Should cause an install
        install = Installer()
        comfydir = os.path.join(install.ourdir, "ComfyUI")

        # Should now exist
        assert os.path.exists(comfydir)

        # Force downgrade to old version
        target_version = "9a8f58638c74c11e515fc10f760ba84a7ce7b2a4"
        install.install(target_version)

        # Should now be downgraded
        installed_version = install.get_commit_hash()
        assert installed_version == target_version

        # Upgrade to standard version
        install.install(COMFYUI_VERSION)

        # Should now be upgraded
        installed_version = install.get_commit_hash()
        assert installed_version == COMFYUI_VERSION
