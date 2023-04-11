# build_helper.py
# This is just a build helper script to build the pypi package.
import subprocess


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


patch_requirements()
patch_toml()

# try:
#     run(["python", "-m", "build"])
# finally:
#     patch_requirements(unpatch=True)
#     patch_toml(unpatch=True)
