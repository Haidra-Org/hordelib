# run_comfyui.py
# This is a helper to run the embedded comfyui during development.
# This is not required for runtime use of hordelib.
# Run this only with: tox -e comfyui
import os
import subprocess
import webbrowser

from hordelib.config_path import get_comfyui_path


class ComfyWebAppLauncher:
    # I know what you're thinking. Feel free to replace this with a pure python
    # implementation applying a unified diff. Good luck.
    PATCH = [
        '    with open("../comfy-prompt.json", "wt", encoding="utf-8") as f:',
        "        f.write(json.dumps(prompt, indent=4))",
    ]

    @classmethod
    def run_comfyui(cls):
        # If we're running the embedded version, it's likely we want
        # to create or edit pipelines for hordelib, so patch comfyui
        # to save it's backend pipelines as JSON when they are run, into
        # the project root as "comfy-prompt.json"
        cls.patch()

        # Launch a browser
        webbrowser.open("http://127.0.0.1:8188/")

        # Now launch the comfyui process and replace our current process
        os.chdir(get_comfyui_path())
        subprocess.run(
            ["python", "main.py"],
            shell=True,
            text=True,
            cwd=get_comfyui_path(),
        )

    @classmethod
    def patch(cls):
        sourcefile = os.path.join(get_comfyui_path(), "execution.py")

        with open(sourcefile, encoding="utf-8") as infile:
            source = infile.readlines()

        # We just want to inject a couple of lines at the end of the
        # validate_prompt method.
        patched = False
        for i, line in enumerate(source):
            if line.startswith("def validate_prompt(prompt):"):
                j = i + 1
                while j < len(source) and not source[j].startswith(
                    '    return (True, "")',
                ):
                    # If we pass an already patched line, abort, we've already done this
                    if source[j].startswith(ComfyWebAppLauncher.PATCH[0]):
                        j = len(source)
                        break
                    j += 1
                if j >= len(source):
                    break
                for patchline in ComfyWebAppLauncher.PATCH:
                    source.insert(j, f"{patchline}\n")
                    j += 1
                patched = True
                break

        if patched:
            with open(sourcefile, "wt") as outfile:
                outfile.writelines(source)


if __name__ == "__main__":
    ComfyWebAppLauncher.run_comfyui()
