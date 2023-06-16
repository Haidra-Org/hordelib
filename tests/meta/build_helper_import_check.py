import sys


def import_check():
    # NOTE These imports should *exactly* match any hordelib imports in build_helper.py
    # If this test fails, (particularly during CI) include the imports missing in `release.yaml`.
    # and update [testenv:test-build-helper] in `tox.ini`.

    from hordelib import install_comfy
    from hordelib.consts import COMFYUI_VERSION


if __name__ == "__main__":
    try:
        import_check()
    except ImportError:
        exit(1)
