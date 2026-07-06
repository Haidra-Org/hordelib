"""Graph-cut helpers for the disaggregated stage entry points.

Each function takes a fully materialized family graph (a :class:`~hordelib.pipeline.graph.ComfyGraph`)
and cuts it down to a single stage: it points the checkpoint loader at only the component the stage
runs, drops the original image output (ComfyUI executes every OUTPUT_NODE, so a stray one would drag
the whole pipeline back in), and adds/rewires the stage IO nodes so ancestor-only execution runs just
that stage.

The cut is only valid on the v1 family shape: a single combined ``model_loader``
(``HordeCheckpointLoader`` carrying model+clip+vae), ``prompt``/``negative_prompt`` CLIPTextEncode
nodes, a ``sampler`` KSampler (plus an optional ``upscale_sampler`` for hires), ``vae_encode`` for
img2img source encode, ``vae_decode``, and an ``output_image``. Each helper validates that shape
structurally (never by family name) and raises :class:`StageGraphUnsupportedError` otherwise, so a
family that does not present it (Flux's ``SamplerCustomAdvanced`` sampler, Qwen/Z-Image's split
``CLIPLoader``/``VAELoader`` where the loader-subset flags are no-ops) fails loudly instead of running
mis-wired and silently re-encoding text or decoding with a subset-disabled VAE.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

from hordelib.execution.interface import OutputKind, OutputSpec, StageGraphUnsupportedError
from hordelib.pipeline.graph import NodeRef

if TYPE_CHECKING:
    from hordelib.pipeline.graph import ComfyGraph

_SPLIT_LOADER_CLASS_TYPES = ("CLIPLoader", "VAELoader")
"""Class types whose presence means the family splits its component loaders, so the combined
``HordeCheckpointLoader`` subset flags cannot govern which components load."""


def set_loader_subset(graph: ComfyGraph, *, model: bool, clip: bool, vae: bool) -> None:
    """Point the checkpoint loader at only the components this stage runs."""
    loader_inputs = graph.raw["model_loader"]["inputs"]
    loader_inputs["output_model"] = model
    loader_inputs["output_clip"] = clip
    loader_inputs["output_vae"] = vae


def _live_ancestors(graph: ComfyGraph, roots: Iterable[str]) -> set[str]:
    """Titles reachable by walking node input connections upward from ``roots`` (roots excluded).

    An input value is a connection when it is a ``[title, output_index]`` list whose title names a
    node in this graph; scalar inputs are ignored. Used to reason about what a cut stage would
    actually execute, independent of node titles.
    """
    raw = graph.raw
    seen: set[str] = set()
    stack = list(roots)
    while stack:
        node = raw.get(stack.pop())
        if node is None:
            continue
        for value in node.get("inputs", {}).values():
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str) and value[0] in raw:
                if value[0] not in seen:
                    seen.add(value[0])
                    stack.append(value[0])
    return seen


def _class_type(graph: ComfyGraph, title: str) -> str | None:
    return graph.node(title).get("class_type")


def _require_supported_loader(graph: ComfyGraph) -> None:
    """Verify the graph loads its components through a single combined checkpoint loader.

    Raises:
        StageGraphUnsupportedError: If ``model_loader`` is absent, is not a ``HordeCheckpointLoader``,
            or if the graph carries split ``CLIPLoader``/``VAELoader`` nodes (whose presence makes the
            loader-subset flags no-ops).
    """
    if not graph.has_node("model_loader"):
        raise StageGraphUnsupportedError("stage graph has no 'model_loader' node")
    loader_class = _class_type(graph, "model_loader")
    if loader_class != "HordeCheckpointLoader":
        raise StageGraphUnsupportedError(
            f"stage graph 'model_loader' is {loader_class!r}, not a combined HordeCheckpointLoader "
            "carrying model+clip+vae",
        )
    split = sorted(
        {
            class_type
            for class_type in (_class_type(graph, title) for title in graph.node_titles())
            if class_type is not None and class_type in _SPLIT_LOADER_CLASS_TYPES
        },
    )
    if split:
        raise StageGraphUnsupportedError(
            f"stage graph uses split component loaders {split}; the checkpoint-loader subset flags are no-ops",
        )


def _require_node(graph: ComfyGraph, title: str, description: str) -> None:
    if not graph.has_node(title):
        raise StageGraphUnsupportedError(f"stage graph is missing its {description} (expected node titled {title!r})")


def _has_class_ancestor(graph: ComfyGraph, ancestors: set[str], class_type: str) -> bool:
    return any(_class_type(graph, title) == class_type for title in ancestors)


def cut_encode_text_stage(graph: ComfyGraph) -> tuple[OutputSpec, ...]:
    """Cut a materialized graph down to the text-encode stage (loads only the CLIP).

    Points the loader at the CLIP subset, drops the image output, and appends a
    ``HordeConditioningOutput`` on each of ``prompt``/``negative_prompt``.

    Raises:
        StageGraphUnsupportedError: If the loader shape is unsupported or a prompt node is absent.
    """
    _require_supported_loader(graph)
    _require_node(graph, "prompt", "positive prompt CLIPTextEncode")
    _require_node(graph, "negative_prompt", "negative prompt CLIPTextEncode")
    set_loader_subset(graph, model=False, clip=True, vae=False)
    del graph.raw["output_image"]
    graph.add_node("cond_positive_output", "HordeConditioningOutput", {"conditioning": NodeRef("prompt")})
    graph.add_node("cond_negative_output", "HordeConditioningOutput", {"conditioning": NodeRef("negative_prompt")})
    return (
        OutputSpec(node="cond_positive_output", kind=OutputKind.CONDITIONING),
        OutputSpec(node="cond_negative_output", kind=OutputKind.CONDITIONING),
    )


def cut_sample_stage(
    graph: ComfyGraph,
    *,
    positive_bytes: bytes,
    negative_bytes: bytes,
    source_latent_bytes: bytes | None,
) -> tuple[OutputSpec, ...]:
    """Cut a materialized graph down to the sample stage (loads only the UNet).

    Injects the received conditioning in place of the graph's own text encoders and (for img2img) the
    received source latent in place of the graph's VAE-encode, then repoints the output at the final
    latent. After rewiring it asserts, structurally, that no ``CLIPTextEncode`` remains a live ancestor
    of the executed output (otherwise the stage would re-encode text with a subset-disabled CLIP).

    Raises:
        StageGraphUnsupportedError: If the loader shape is unsupported; if there is no ``sampler``
            node; if the graph VAE-encodes a source image (img2img) but ``source_latent_bytes`` is
            None; or if conditioning injection failed to displace the text encoders.
    """
    _require_supported_loader(graph)
    _require_node(graph, "sampler", "sampler (KSampler)")

    if source_latent_bytes is None and _has_class_ancestor(graph, _live_ancestors(graph, ["sampler"]), "VAEEncode"):
        raise StageGraphUnsupportedError(
            "sample_stage: graph VAE-encodes a source image (img2img) but no source_latent_bytes was provided",
        )

    set_loader_subset(graph, model=True, clip=False, vae=False)
    graph.add_node("inject_positive", "HordeConditioningInput", {"conditioning_bytes": positive_bytes})
    graph.add_node("inject_negative", "HordeConditioningInput", {"conditioning_bytes": negative_bytes})
    for sampler_title in ("sampler", "upscale_sampler"):
        if graph.has_node(sampler_title):
            graph.connect(f"{sampler_title}.positive", NodeRef("inject_positive"))
            graph.connect(f"{sampler_title}.negative", NodeRef("inject_negative"))
    if source_latent_bytes is not None:
        graph.add_node("inject_source_latent", "HordeLatentInput", {"latent_bytes": source_latent_bytes})
        graph.connect("sampler.latent_image", NodeRef("inject_source_latent"))

    final_latent = graph.node("vae_decode")["inputs"]["samples"]
    del graph.raw["output_image"]
    graph.add_node("latent_output", "HordeLatentOutput", {"samples": NodeRef(final_latent[0], final_latent[1])})

    stray = sorted(t for t in _live_ancestors(graph, ["latent_output"]) if _class_type(graph, t) == "CLIPTextEncode")
    if stray:
        raise StageGraphUnsupportedError(
            f"sample_stage: conditioning injection did not take; CLIPTextEncode nodes still live: {stray}",
        )
    return (OutputSpec(node="latent_output", kind=OutputKind.LATENT),)


def cut_vae_encode_stage(graph: ComfyGraph) -> tuple[OutputSpec, ...]:
    """Cut a materialized graph down to the VAE-encode stage (loads only the VAE), for img2img/remix.

    Raises:
        StageGraphUnsupportedError: If the loader shape is unsupported or there is no ``vae_encode``
            node (a txt2img graph has none, so it cannot be the source of an img2img start latent).
    """
    _require_supported_loader(graph)
    _require_node(graph, "vae_encode", "VAEEncode (img2img source encode)")
    set_loader_subset(graph, model=False, clip=False, vae=True)
    del graph.raw["output_image"]
    graph.add_node("latent_output", "HordeLatentOutput", {"samples": NodeRef("vae_encode")})
    return (OutputSpec(node="latent_output", kind=OutputKind.LATENT),)


def cut_decode_stage(graph: ComfyGraph, *, latent_bytes: bytes) -> None:
    """Cut a materialized graph down to the decode stage (loads only the VAE), reusing the image output.

    Injects the received latent in place of the graph's own sample source at ``vae_decode``; the graph's
    original image output is kept and reused (the caller runs the materialize-declared outputs).

    Raises:
        StageGraphUnsupportedError: If the loader shape is unsupported or there is no ``vae_decode`` node.
    """
    _require_supported_loader(graph)
    _require_node(graph, "vae_decode", "VAEDecode")
    set_loader_subset(graph, model=False, clip=False, vae=True)
    graph.add_node("inject_latent", "HordeLatentInput", {"latent_bytes": latent_bytes})
    graph.connect("vae_decode.samples", NodeRef("inject_latent"))
