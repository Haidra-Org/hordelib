"""Routing for the standalone-VAE load path: prefer a pre-extracted VAE over a monolithic subset-load.

A disaggregated decode stage needs only a checkpoint's VAE. Subset-loading it from the monolithic
checkpoint reads (and caches) the whole multi-gigabyte file under the model's name, so two models that
embed byte-identical VAE weights never share a cached VAE. horde_model_reference's component-identity
sidecar records the embedded VAE's content hash and, when extracted, the standalone VAE file's name; this
module turns a checkpoint path into a plan to load that small standalone file and cache it by content
identity, so those two models hit the same cache entry.

The module is deliberately free of torch and ComfyUI: it only reads the sidecar (torch-free) and resolves a
file name to a path through an injected locator, so the routing decision is unit-testable without a GPU or an
initialised ComfyUI. The weight load, cache insertion and seamless-tiling normalization live in the loader
node, which delegates the decision here.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from horde_model_reference.component_hash import ComponentKind
from horde_model_reference.component_identity import read_sidecar

_DISABLE_ENV_VAR = "HORDE_DISABLE_STANDALONE_VAE_PATH"
"""Set truthy to restore the prior monolithic subset-load behaviour for every VAE-only request."""

_TRUTHY_VALUES = frozenset({"1", "true", "yes", "on"})


@dataclass(frozen=True)
class StandaloneVaePlan:
    """A resolved plan to load a checkpoint's VAE from its pre-extracted standalone file."""

    content_hash: str
    """The embedded VAE's container-independent content hash (from the sidecar)."""
    vae_file_path: Path
    """The on-disk standalone VAE file to load instead of subset-loading the monolithic checkpoint."""
    vae_tensor_bytes: int
    """The VAE's tensor byte size (from the sidecar), for the loader's approximate RAM-cost estimate."""

    @property
    def cache_key(self) -> str:
        """The content-addressed in-RAM cache key, identical for every checkpoint embedding this VAE.

        Keying by content hash rather than model name is what lets two models that share byte-identical VAE
        weights resolve to the same cached VAE entry (a cross-model cache hit).
        """
        return f"vae@{self.content_hash}"


def standalone_vae_path_disabled() -> bool:
    """Return whether the kill-switch env var disables the standalone-VAE path (default: enabled)."""
    return os.environ.get(_DISABLE_ENV_VAR, "").strip().lower() in _TRUTHY_VALUES


def plan_standalone_vae_load(
    ckpt_path: Path,
    locate_vae_file: Callable[[str], Path | None],
) -> StandaloneVaePlan | None:
    """Return a standalone-VAE plan for *ckpt_path*, or None to fall back to the subset load.

    A None result always means "fall back to the existing subset load", never an error. It is returned when
    the checkpoint has no fresh sidecar (absent or stale), the sidecar records no embedded VAE (or one whose
    standalone file was never extracted), or the extracted file is not found by *locate_vae_file*.

    Args:
        ckpt_path: The monolithic checkpoint the VAE would otherwise be subset-loaded from. Its sidecar is
            read from beside it; staleness (a size mismatch) is handled by the reader as "no sidecar".
        locate_vae_file: Maps an extracted VAE file name to its on-disk path, or None when absent. The loader
            wires this to ComfyUI's ``vae`` folder search so the file is found exactly where the
            download-time extraction wrote it.

    Returns:
        A :class:`StandaloneVaePlan` when the standalone path can serve the VAE, else None.
    """
    sidecar = read_sidecar(ckpt_path)
    if sidecar is None:
        return None

    vae_identity = sidecar.embedded.get(ComponentKind.VAE.value)
    if vae_identity is None or not vae_identity.extracted_file_name:
        return None

    located = locate_vae_file(vae_identity.extracted_file_name)
    if located is None:
        return None

    return StandaloneVaePlan(
        content_hash=vae_identity.content_hash,
        vae_file_path=located,
        vae_tensor_bytes=vae_identity.tensor_bytes,
    )
