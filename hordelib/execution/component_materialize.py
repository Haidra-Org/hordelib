"""Materialise a canonical shared component into a comfy module the lane can publish.

The component lane holds a hot-set of canonical components (VAEs and text encoders) resident and publishes
them for inference slots to adopt. This module turns one :class:`CanonicalComponent` from the reference into
the loaded torch module whose state dict gets published, choosing the cheapest available source and the right
comfy load path for it.

Identity: the reference's cross-machine ``content_hash`` is carried through unchanged as the component's
stable identity; the machine-local ``hash_state_dict`` of the loaded module is what intra-card adoption keys
on. For a monolithic VAE the standalone file is sliced torch-free by :mod:`horde_model_reference`, whose
content hash equals the embedded one by construction; a monolithic text encoder is obtained by loading a
representative checkpoint and keeping its text-encoder module (comfy renames the keys at load, so it is
deliberately never sliced to a standalone file).

comfy/torch imports are deferred into the load helpers, so this module imports GPU-free: the source-selection
logic and the data types are unit-testable without a backend. The actual comfy load paths are rig-validated.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from horde_model_reference.component_hash import ComponentKind

if TYPE_CHECKING:
    from collections.abc import Callable

    from horde_model_reference.canonical_components import CanonicalComponent, CanonicalComponentSource

_LABEL_FOR_KIND: dict[ComponentKind, str] = {
    ComponentKind.VAE: "vae",
    ComponentKind.TEXT_ENCODERS: "clip",
}
"""The label per kind identifying where a component's weights live on the loaded ComfyUI objects: a VAE's
``first_stage_model`` (label ``vae``), a CLIP's ``cond_stage_model`` (label ``clip``)."""


class NoMaterializationSourceError(LookupError):
    """Raised when a canonical component declares no usable source to obtain it from."""


@dataclass
class MaterializedComponent:
    """A loaded component ready to publish: the shared module plus its stable identity."""

    module: Any
    """The ``torch.nn.Module`` whose ``state_dict()`` is published (a VAE's ``first_stage_model`` or a CLIP's
    ``cond_stage_model``)."""
    content_hash: str
    """The reference's cross-machine content hash for this component (its stable identity)."""
    label: str
    """The publish label: ``"vae"`` or ``"clip"``."""


def _choose_source(component: CanonicalComponent) -> tuple[CanonicalComponentSource, bool]:
    """Pick the cheapest source for *component*, returning ``(source, is_embedded)``.

    A standalone (non-embedded) source is preferred: it loads directly, whereas an embedded source must slice
    a VAE out of a checkpoint or load a whole checkpoint for a text encoder. Falls back to an embedded source
    when that is all the reference offers.

    Raises:
        NoMaterializationSourceError: The component declares no sources.
    """
    if not component.sources:
        raise NoMaterializationSourceError(f"Canonical component {component.content_hash[:12]} has no sources")
    standalone = next((source for source in component.sources if not source.embedded), None)
    if standalone is not None:
        return standalone, False
    return component.sources[0], True


def materialize(
    component: CanonicalComponent,
    resolve_path: Callable[[str], str],
    *,
    temp_dir: str | Path,
    clip_type_resolver: Callable[[str], str | None] | None = None,
) -> MaterializedComponent:
    """Load *component* into a comfy module ready to publish.

    Args:
        component: The canonical component to materialise.
        resolve_path: Maps a source key to an absolute on-disk path. For a standalone source the key is its
            ``file_name``; for an embedded source it is the carrier's ``model_name`` (the checkpoint to load
            or slice). The lane owns file layout, so it supplies this.
        temp_dir: Directory a sliced monolithic VAE's standalone file is written to.
        clip_type_resolver: Maps a source model name to the comfy ``CLIPType`` name its pipeline loads the
            text encoder with (e.g. ``"qwen_image"``, ``"lumina2"``). A split-file text encoder must be loaded
            with the same type the consumer's pipeline uses, or the wrapped module differs and no consumer
            adopts it. ``None`` (or an unknown name) falls back to ``stable_diffusion``. Only consulted for a
            split-file text encoder; the source model is the one ``materialize`` actually chose.

    Returns:
        The loaded module, the component's content hash, and its publish label.

    Raises:
        NoMaterializationSourceError: The component has no source to load from.
    """
    source, is_embedded = _choose_source(component)
    module = _load_component_module(
        component.kind,
        source,
        is_embedded=is_embedded,
        content_hash=component.content_hash,
        resolve_path=resolve_path,
        temp_dir=Path(temp_dir),
        clip_type_resolver=clip_type_resolver,
    )
    return MaterializedComponent(
        module=module,
        content_hash=component.content_hash,
        label=_LABEL_FOR_KIND[component.kind],
    )


def _load_component_module(
    kind: ComponentKind,
    source: CanonicalComponentSource,
    *,
    is_embedded: bool,
    content_hash: str,
    resolve_path: Callable[[str], str],
    temp_dir: Path,
    clip_type_resolver: Callable[[str], str | None] | None = None,
) -> Any:
    """Load the shared submodule for one component via comfy (rig-only path; comfy/torch imported lazily)."""
    import comfy.sd
    import comfy.utils

    from hordelib.execution.zero_copy_load import zero_copy_state_dict_assignment

    with zero_copy_state_dict_assignment():
        if kind is ComponentKind.VAE:
            vae_path = _standalone_vae_path(source, is_embedded, content_hash, resolve_path, temp_dir)
            vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(vae_path))
            return vae.first_stage_model

        if is_embedded:
            # A monolithic text encoder: load the representative checkpoint and keep only its CLIP module.
            # load_checkpoint_guess_config returns (model, clip, vae, ...); the loader node relies on this.
            checkpoint_path = resolve_path(source.model_name)
            loaded = comfy.sd.load_checkpoint_guess_config(
                checkpoint_path,
                output_vae=False,
                output_clip=True,
                embedding_directory=None,
            )
            return loaded[1].cond_stage_model

        # A split-file text encoder (e.g. Qwen/Z-Image). It must be loaded with the same CLIPType the
        # consumer's pipeline uses (qwen_image, lumina2, ...); the wrong type wraps the encoder differently
        # and no consumer's module hash would match. The resolver maps the chosen source model to that type.
        clip_path = resolve_path(_require_file_name(source))
        clip = comfy.sd.load_clip(
            ckpt_paths=[clip_path],
            embedding_directory=None,
            clip_type=_resolve_clip_type(source.model_name, clip_type_resolver),
        )
        return clip.cond_stage_model


def _resolve_clip_type(model_name: str, clip_type_resolver: Callable[[str], str | None] | None) -> Any:
    """Return the comfy ``CLIPType`` for a split-file text encoder, defaulting to ``STABLE_DIFFUSION``.

    ``clip_type_resolver`` maps the source model name to a comfy type name (case-insensitive); an unknown or
    missing name falls back to ``STABLE_DIFFUSION``, matching comfy's own ``CLIPLoader`` default.
    """
    import comfy.sd

    type_name = clip_type_resolver(model_name) if clip_type_resolver is not None else None
    return getattr(comfy.sd.CLIPType, (type_name or "stable_diffusion").upper(), comfy.sd.CLIPType.STABLE_DIFFUSION)


def _standalone_vae_path(
    source: CanonicalComponentSource,
    is_embedded: bool,
    content_hash: str,
    resolve_path: Callable[[str], str],
    temp_dir: Path,
) -> str:
    """Return a path to a standalone VAE file, slicing it out of a monolithic checkpoint when embedded.

    The sliced file's content hash equals the embedded VAE's by construction (see
    :func:`horde_model_reference.component_hash.extract_embedded_vae_file`).
    """
    if not is_embedded:
        return resolve_path(_require_file_name(source))
    from horde_model_reference.component_hash import extract_embedded_vae_file

    checkpoint_path = resolve_path(source.model_name)
    dest = temp_dir / f"{content_hash}.vae.safetensors"
    if not dest.exists():
        extract_embedded_vae_file(checkpoint_path, dest)
    return str(dest)


def _require_file_name(source: CanonicalComponentSource) -> str:
    """Return a standalone source's ``file_name``, or raise if the reference did not record one."""
    if source.file_name is None:
        raise NoMaterializationSourceError(
            f"Standalone source for model {source.model_name!r} has no file_name to load",
        )
    return source.file_name
