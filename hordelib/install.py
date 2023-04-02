# install.py
# Fetch a specific version of the upstream ComfyUI project
import os
import sys
import subprocess
from loguru import logger


class Installer:

    def __init__(self):
        self.ourdir = os.path.dirname(os.path.realpath(__file__))

    def get_commit_hash(self):
        head_file = os.path.join(self.ourdir, "ComfyUI", ".git", "HEAD")
        if not os.path.exists(head_file):
            return "NOT FOUND"
        try:
            with open(head_file, "r") as f:
                head_contents = f.read().strip()

            if not head_contents.startswith("ref:"):
                return head_contents

            ref_path = os.path.join(".git", *head_contents[5:].split("/"))

            with open(ref_path, "r") as f:
                commit_hash = f.read().strip()

            return commit_hash
        except Exception:
            return ""

    def install(self, comfy_version):

        commands = [
            f"git clone https://github.com/comfyanonymous/ComfyUI.git {self.ourdir}/ComfyUI",
            # f"cd {self.ourdir}/ComfyUI && git checkout {comfy_version}",
        ]

        return  # FIXME call it good

        if os.path.exists(f"{self.ourdir}/ComfyUI"):
            # Check ComfyUI is up to date
            version = self.get_commit_hash()
            if version == comfy_version:
                return
            commands = [
                f"cd {self.ourdir}/ComfyUI && git checkout {comfy_version}",
            ]
            logger.info(
                f"Current ComfyUI version {version[:8]} requires {comfy_version[:8]}"
            )

        logger.info("Updating ComfyUI")

        for command in commands:
            logger.warning(command)
            result = subprocess.run(command, shell=True, text=True, capture_output=True)
            if result.returncode:
                print(result.stderr, file=sys.stderr)
                raise Exception(result.returncode)
