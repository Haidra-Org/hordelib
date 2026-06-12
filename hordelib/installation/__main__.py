"""CLI entry point: install/verify the ComfyUI environment without starting hordelib.

Usage::

    python -m hordelib.installation [--root <dir>] [--export-snapshot <file>]

Intended for pre-baking docker images and offline environments.
"""

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m hordelib.installation",
        description="Install or update the ComfyUI environment pinned by hordelib's manifest.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="The ComfyUI checkout directory. Defaults to hordelib's standard resolution.",
    )
    parser.add_argument(
        "--export-snapshot",
        type=Path,
        default=None,
        help="Write a ComfyUI-Manager-compatible snapshot JSON to this path and exit.",
    )
    args = parser.parse_args(argv)

    from hordelib.installation.installer import EnvironmentInstaller
    from hordelib.installation.manifest import load_packaged_manifest

    manifest = load_packaged_manifest()

    if args.export_snapshot is not None:
        args.export_snapshot.write_text(
            json.dumps(manifest.to_manager_snapshot(), indent=4) + "\n",
            encoding="utf-8",
        )
        print(f"Snapshot written to {args.export_snapshot}")
        return 0

    root = args.root
    if root is None:
        from hordelib.config_path import get_comfyui_path

        root = get_comfyui_path()

    installer = EnvironmentInstaller(manifest)
    installer.ensure(root)
    print(f"ComfyUI environment at {root} matches manifest (ComfyUI {manifest.comfyui_ref[:8]}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
