import pathlib


def get_hook_dirs() -> list[str]:
    return [str(pathlib.Path(__file__).parent / "pyinstaller_hooks")]


def get_PyInstaller_tests() -> list[str]:
    return []  # FIXME
    return [str(pathlib.Path(__file__).parent / "pyinstaller_tests")]
