"""Test-only helpers for importing ComfyUI without running full hordelib initialization."""

import importlib
import sys
from types import ModuleType

import torch

from hordelib.config_path import set_system_path


def import_comfy_module(module_name: str) -> ModuleType:
    """Import a synchronized ComfyUI module with a minimal, isolated CLI configuration."""
    set_system_path()
    argv = sys.argv
    sys.argv = [argv[0]]
    if torch.version.cuda is None and getattr(torch.version, "hip", None) is None:
        sys.argv.append("--cpu")
    try:
        from comfy.options import enable_args_parsing

        enable_args_parsing()
        return importlib.import_module(module_name)
    finally:
        sys.argv = argv
