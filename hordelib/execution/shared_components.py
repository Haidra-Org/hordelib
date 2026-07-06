"""Content-hashing of a loaded model component's state dict.

The cross-machine component-identity work (:mod:`hordelib.execution.component_stability_probe`) needs a
hash of a ComfyUI-loaded module's *runtime* state dict, distinct from the torch-free file-content hash
the model reference ships. This module provides that single hash. It folds ``str(tensor.dtype)`` into the
digest, and ComfyUI's load-time dtype selection is device-dependent, so this hash may differ between
machines: it is only meaningful as a machine-local identity, never a shippable one.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping


def hash_state_dict(state_dict: Mapping[str, Any]) -> str:
    """Content-hash a state dict's tensor bytes (keys, shapes, dtypes, and data).

    Intended for the CPU (mmap-view) state dict, where reading the bytes is cheap; byte-identical
    components in different checkpoint files hash equal.
    """
    import torch

    digest = hashlib.sha256()
    for key in sorted(state_dict.keys()):
        value = state_dict[key]
        if not isinstance(value, torch.Tensor):
            continue
        digest.update(key.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        tensor = value.detach()
        if tensor.device.type != "cpu":
            tensor = tensor.cpu()
        # reshape(-1) before viewing as bytes: a 0-dim scalar (e.g. an fp8 scale factor in a Qwen text
        # encoder) cannot be reinterpreted with view(torch.uint8), which raises on a 0-dim tensor. The shape
        # is already folded into the digest above, so flattening loses nothing.
        digest.update(tensor.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()
