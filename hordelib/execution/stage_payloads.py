"""Wire serialization for disaggregated-stage intermediates (CONDITIONING and LATENT).

Pipeline disaggregation cuts a single generation into stages that run in separate processes:
a text-encode stage emits CONDITIONING, a sample stage consumes it and emits a LATENT, a
decode stage consumes the LATENT. The intermediates must cross a process boundary (and be
checkpointed in the parent for fault re-dispatch), so they are serialized to a self-contained
bytes blob here. The stage IO nodes (:mod:`hordelib.nodes.node_stage_io`) and the worker's IPC
both use these functions, so the cut points stay byte-identical to a monolithic run (verified:
the same tensors fed to the same sampler/VAE reproduce the exact image at a fixed seed).

Wire format (self-describing, no pickle):

    [8 bytes big-endian uint64: header length H]
    [H bytes: utf-8 JSON header describing structure + scalar dict entries + tensor key map]
    [safetensors bytes: every tensor, keyed by the names the header references]

CONDITIONING is ComfyUI's ``list[[cond_tensor, dict]]`` where the dict carries tensors
(``pooled_output`` and friends) alongside float scalars; LATENT is ``{"samples": tensor, ...}``.
Tensors serialize via safetensors (CPU); the caller moves them to the target device on load.
Non-tensor, non-JSON-scalar dict entries are dropped (the disaggregated v1 families carry none;
controlnet, which embeds a live model ref in its conditioning, stays on the monolithic path).
"""

from __future__ import annotations

import json
import struct
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    import torch

_HEADER_LEN_STRUCT = struct.Struct(">Q")
CONDITIONING_TYPE = "CONDITIONING"
LATENT_TYPE = "LATENT"


def _is_tensor(value: Any) -> bool:
    return hasattr(value, "shape") and hasattr(value, "dtype") and hasattr(value, "cpu")


def _is_json_scalar(value: Any) -> bool:
    return isinstance(value, (bool, int, float, str)) or value is None


def _pack(header: dict[str, Any], tensors: dict[str, torch.Tensor]) -> bytes:
    from safetensors.torch import save

    header_bytes = json.dumps(header).encode("utf-8")
    payload = save(tensors) if tensors else b""
    return _HEADER_LEN_STRUCT.pack(len(header_bytes)) + header_bytes + payload


def _unpack(blob: bytes) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    from safetensors.torch import load

    (header_len,) = _HEADER_LEN_STRUCT.unpack_from(blob, 0)
    start = _HEADER_LEN_STRUCT.size
    header = json.loads(blob[start : start + header_len].decode("utf-8"))
    payload = blob[start + header_len :]
    tensors = load(payload) if payload else {}
    return header, tensors


def serialize_conditioning(conditioning: list[Any]) -> bytes:
    """Serialize a ComfyUI CONDITIONING (``list[[cond_tensor, dict]]``) to a self-contained blob."""
    tensors: dict[str, torch.Tensor] = {}
    pairs: list[dict[str, Any]] = []
    for i, pair in enumerate(conditioning):
        cond_tensor, cond_dict = pair[0], pair[1]
        cond_key = f"c{i}"
        tensors[cond_key] = cond_tensor.contiguous().cpu()
        dict_tensors: dict[str, str] = {}
        dict_scalars: dict[str, Any] = {}
        for key, value in cond_dict.items():
            if _is_tensor(value):
                tkey = f"c{i}_{key}"
                tensors[tkey] = value.contiguous().cpu()
                dict_tensors[key] = tkey
            elif _is_json_scalar(value):
                dict_scalars[key] = value
            else:
                # Dropped on purpose (see module docstring); warn so an unexpected carrier (e.g. a
                # controlnet conditioning routed here by mistake) is visible rather than silent.
                logger.warning("stage_payloads: dropping non-tensor/non-scalar conditioning entry {!r}", key)
        pairs.append({"cond_key": cond_key, "dict_tensors": dict_tensors, "dict_scalars": dict_scalars})
    return _pack({"kind": "conditioning", "pairs": pairs}, tensors)


def deserialize_conditioning(blob: bytes, *, device: str = "cpu") -> list[Any]:
    """Reconstruct a CONDITIONING from a blob, moving tensors onto ``device``."""
    header, tensors = _unpack(blob)
    if header.get("kind") != "conditioning":
        raise ValueError(f"expected conditioning blob, got {header.get('kind')!r}")
    conditioning: list[Any] = []
    for pair in header["pairs"]:
        cond_dict: dict[str, Any] = dict(pair["dict_scalars"])
        for key, tkey in pair["dict_tensors"].items():
            cond_dict[key] = tensors[tkey].to(device)
        conditioning.append([tensors[pair["cond_key"]].to(device), cond_dict])
    return conditioning


def serialize_latent(latent: dict[str, Any]) -> bytes:
    """Serialize a ComfyUI LATENT (``{"samples": tensor, ...}``) to a self-contained blob."""
    tensors: dict[str, torch.Tensor] = {"samples": latent["samples"].contiguous().cpu()}
    scalars: dict[str, Any] = {}
    extra_tensors: dict[str, str] = {}
    for key, value in latent.items():
        if key == "samples":
            continue
        if _is_tensor(value):
            tkey = f"extra_{key}"
            tensors[tkey] = value.contiguous().cpu()
            extra_tensors[key] = tkey
        elif _is_json_scalar(value):
            scalars[key] = value
    return _pack({"kind": "latent", "scalars": scalars, "extra_tensors": extra_tensors}, tensors)


def deserialize_latent(blob: bytes, *, device: str = "cpu") -> dict[str, Any]:
    """Reconstruct a LATENT from a blob, moving tensors onto ``device``."""
    header, tensors = _unpack(blob)
    if header.get("kind") != "latent":
        raise ValueError(f"expected latent blob, got {header.get('kind')!r}")
    latent: dict[str, Any] = dict(header["scalars"])
    for key, tkey in header.get("extra_tensors", {}).items():
        latent[key] = tensors[tkey].to(device)
    latent["samples"] = tensors["samples"].to(device)
    return latent
