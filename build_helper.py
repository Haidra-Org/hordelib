# build_helper.py
# This is just a build helper script to build the pypi package.
import argparse
import os
import shutil
import subprocess

# NOTE If you have come here looking for an explanation of why the CI is failing,
# know that build_helper.py can be and is run without a full hordelib installation.
# Accordingly, any imports that ultimately resolve as a result of the following imports
# must be included in release.yml. The section at the time of writing is `🛠 Install pypa/build`.


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
        elif not unpatch and line.startswith("#write_to ="):
            newfile.append(f"{line[1:]}")
        elif unpatch and line.startswith("write_to ="):
            newfile.append(f"#{line}")
        else:
            newfile.append(line)
    with open("pyproject.toml", "w") as outfile:
        outfile.writelines(newfile)


def cleanup_version_file(unpatch=False):
    # ComfyUI is no longer statically bundled into wheels: release installs fetch the
    # manifest-pinned ComfyUI on first initialise() (see hordelib/installation/).
    if unpatch and os.path.exists("hordelib/_version.py"):
        os.remove("hordelib/_version.py")

    # Clean up any stale static bundle left behind by older build versions
    if os.path.exists("hordelib/_comfyui"):
        try:
            shutil.rmtree("hordelib/_comfyui")
        except PermissionError:
            print("Can't delete `hordelib/_comfyui/` please delete it manually and try again")
            exit(1)


def unpatch():
    cleanup_version_file(True)
    patch_requirements(True)
    patch_toml(True)


def patch():
    cleanup_version_file()
    patch_requirements()
    patch_toml()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fix", action="store_true", help="Cleanup after the build")
    args = parser.parse_args()

    if not args.fix:
        patch()
    else:
        unpatch()
