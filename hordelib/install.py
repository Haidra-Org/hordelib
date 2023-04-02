# install.py
# Fetch a specific version of the upstream ComfyUI project
import os
import sys
import subprocess
from hordelib import COMFYUI_VERSION
from loguru import logger


def get_commit_hash():
    target_dir = os.path.dirname(os.path.realpath(__file__))
    head_file = os.path.join(target_dir, "ComfyUI", ".git", "HEAD")
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


commands = [
    "git clone https://github.com/comfyanonymous/ComfyUI.git hordelib/ComfyUI",
    f"cd hordelib/ComfyUI && git checkout {COMFYUI_VERSION}",
]

if os.path.exists("hordelib/ComfyUI"):
    # Check ComfyUI is up to date
    version = get_commit_hash()
    if version == COMFYUI_VERSION:
        exit(0)
    commands = [
        f"cd hordelib/ComfyUI && git checkout {COMFYUI_VERSION}",
    ]
    logger.info(f"Current ComfyUI version {version[:8]} requires {COMFYUI_VERSION[:8]}")

logger.info("Updating ComfyUI")

for command in commands:
    result = subprocess.run(command, shell=True, text=True, capture_output=True)
    if result.returncode:
        print(result.stderr, file=sys.stderr)
        exit(result.returncode)
