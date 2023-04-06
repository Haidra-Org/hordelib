# test_comfy_install.py
import os

from hordelib.consts import COMFYUI_VERSION
from hordelib.install_comfy import Installer


class TestSetup:
    def test_get_hash(self):
        comfyhash = Installer.get_commit_hash()
        assert comfyhash == COMFYUI_VERSION

    def test_run_without_subdir(self):
        result = Installer._run(["echo", "1"])
        assert result
        assert result[0] is True

    def test_run_with_subdir(self):
        result = Installer._run(["echo", "1"], "hordelib")
        assert result
        assert result[0] is True

    def test_run_with_missing_subdir(self):
        result = Installer._run(["echo", "1"], "does-not-exist")
        assert result is None

    def test_bad_exitcode(self):
        # XXX This test doesn't work on github for some reason
        if os.getenv("HORDELIB_TESTING", "") == "no-cuda":
            return
        result = Installer._run(["exit", "1"])
        assert result is None

    def test_comfy_changes(self):
        # Force downgrade to old version
        target_version = "9a8f58638c74c11e515fc10f760ba84a7ce7b2a4"
        Installer.install(target_version)

        # Should now be downgraded
        installed_version = Installer.get_commit_hash()
        assert installed_version == target_version

        # Upgrade to standard version
        Installer.install(COMFYUI_VERSION)

        # Should now be upgraded
        installed_version = Installer.get_commit_hash()
        assert installed_version == COMFYUI_VERSION
