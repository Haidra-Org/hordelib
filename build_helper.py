# build_helper.py
# This is just a build helper script to build the pypi package.
import os
import shutil
import subprocess

from hordelib import install_comfy
from hordelib.consts import COMFYUI_VERSION


def run(command):
    result = subprocess.run(command, shell=True, text=True)
    if result.returncode:
        raise Exception(result.stderr)


def patch_requirements(unpatch=False):
    with open("requirements.txt") as infile:
        data = infile.readlines()
    newfile = []
    for line in data:
        if not unpatch and line.startswith("--"):
            newfile.append(f"#{line}")
        elif unpatch and line.startswith("#--"):
            newfile.append(f"{line[1:]}")
        else:
            newfile.append(line)
    with open("requirements.txt", "w") as outfile:
        outfile.writelines(newfile)


def patch_toml(unpatch=False):
    with open("pyproject.toml") as infile:
        data = infile.readlines()
    newfile = []
    for line in data:
        if not unpatch and line.startswith('dynamic=["version"]'):
            newfile.append(f"#{line}")
        elif not unpatch and line.startswith('#dynamic=["version", "dependencies"]'):
            newfile.append(f"{line[1:]}")
        elif unpatch and line.startswith('#dynamic=["version"]'):
            newfile.append(f"{line[1:]}")
        elif unpatch and line.startswith('dynamic=["version", "dependencies"]'):
            newfile.append(f"#{line}")
        else:
            newfile.append(line)
    with open("pyproject.toml", "w") as outfile:
        outfile.writelines(newfile)


def static_package_comfyui():
    installer = install_comfy.Installer()
    installer.install(COMFYUI_VERSION)

    if not os.path.exists("ComfyUI"):
        raise Exception("ComfyUI not found")
    shutil.copytree("ComfyUI", "hordelib/_comfyui")


if __name__ == "__main__":
    static_package_comfyui()
    patch_requirements()
    patch_toml()
