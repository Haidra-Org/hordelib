# run_comfyui.py
# This is a helper to run the embedded comfyui during development.
# This is not required for runtime use of hordelib.
# Run this only with: tox -e comfyui
import os
import subprocess
import webbrowser

from loguru import logger

from hordelib.config_path import get_comfyui_path, get_hordelib_path
from hordelib.install_comfy import Installer


class ComfyWebAppLauncher:
    @classmethod
    def run_comfyui(cls):
        # Launch a browser
        webbrowser.open("http://127.0.0.1:8188/")
        logger.warning(
            "Wait a moment and then refresh your browser. It takes a while to load the backend.",
        )

        # Tell comfyui where are custom nodes are
        os.environ["AIWORKER_CUSTOM_NODES"] = os.path.join(get_hordelib_path(), "nodes")

        # Now launch the comfyui process and replace our current process
        os.chdir(get_comfyui_path())
        subprocess.run(
            ["python", "main.py"],
            shell=True,
            text=True,
            cwd=get_comfyui_path(),
        )


if __name__ == "__main__":
    ComfyWebAppLauncher.run_comfyui()
