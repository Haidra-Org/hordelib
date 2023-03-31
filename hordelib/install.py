# install.py
# Fetch a specific version of the upstream ComfyUI project
import os
import sys
import subprocess
from hordelib import COMFYUI_VERSION

if os.path.exists("hordelib/ComfyUI"):
    # Nothing to do
    exit(0)

commands = [
    "git clone https://github.com/comfyanonymous/ComfyUI.git hordelib/ComfyUI",
    f"cd hordelib/ComfyUI && git checkout {COMFYUI_VERSION}",
]

for command in commands:
    result = subprocess.run(command, shell=True, text=True, capture_output=True)
    if result.returncode:
        print(result.stderr, file=sys.stderr)
        exit(result.returncode)
