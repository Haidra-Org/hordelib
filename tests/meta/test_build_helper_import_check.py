import sys


def test_import_check():
    # NOTE These imports should *exactly* match any hordelib imports in build_helper.py
    # If this test fails, include the imports missing in `release.yaml`.
    # and update [testenv:test-build-helper] in `tox.ini`.

    from hordelib import install_comfy
    from hordelib.consts import COMFYUI_VERSION
