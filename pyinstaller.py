from pathlib import Path

import PyInstaller.__main__  # type: ignore

HERE = Path(__file__).parent.absolute()
entry_point = str(HERE / "pyinstaller_test_entrypoint.py")


def install():
    PyInstaller.__main__.run(
        [
            "--clean",
            "--log-level=DEBUG",
            "--additional-hooks-dir",
            str(HERE / "pyinstaller_hooks"),
            entry_point,
        ],
    )


install()
