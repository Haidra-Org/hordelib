"""GPU-free tests for the stage graph-cut helpers (:mod:`hordelib.execution.stage_graph`).

These build constructed ComfyUI API-format graphs (title-keyed, no model files) that mimic the shapes
the family materialization produces, then drive the cut helpers to assert the cut is correct: the loader
subset flags are set, the stage IO nodes are added, the outputs point at the stage outputs, and the
original image output is gone for the non-decode stages. They also assert the typed
``StageGraphUnsupportedError`` for the shapes the cut cannot support (Flux's ``SamplerCustomAdvanced``,
split-loader families, and an img2img sample with no injected source latent).
"""

from __future__ import annotations

import pytest

from hordelib.execution.interface import OutputKind, StageGraphUnsupportedError
from hordelib.execution.stage_graph import (
    _class_type,
    _live_ancestors,
    cut_decode_stage,
    cut_encode_text_stage,
    cut_sample_stage,
    cut_vae_encode_stage,
)
from hordelib.pipeline.graph import ComfyGraph


def _node(class_type: str, inputs: dict, title: str) -> dict:
    return {"class_type": class_type, "inputs": inputs, "_meta": {"title": title}}


def _combined_loader() -> dict:
    return _node(
        "HordeCheckpointLoader",
        {"horde_model_name": "m", "output_model": True, "output_clip": True, "output_vae": True},
        "model_loader",
    )


def _sd_txt2img() -> ComfyGraph:
    graph = {
        "model_loader": _combined_loader(),
        "prompt": _node("CLIPTextEncode", {"clip": ["model_loader", 1], "text": "a"}, "prompt"),
        "negative_prompt": _node("CLIPTextEncode", {"clip": ["model_loader", 1], "text": "b"}, "negative_prompt"),
        "empty_latent_image": _node(
            "EmptyLatentImage", {"width": 512, "height": 512, "batch_size": 1}, "empty_latent_image"
        ),
        "sampler": _node(
            "KSampler",
            {
                "model": ["model_loader", 0],
                "positive": ["prompt", 0],
                "negative": ["negative_prompt", 0],
                "latent_image": ["empty_latent_image", 0],
            },
            "sampler",
        ),
        "vae_decode": _node("VAEDecode", {"vae": ["model_loader", 2], "samples": ["sampler", 0]}, "vae_decode"),
        "output_image": _node("HordeImageOutput", {"images": ["vae_decode", 0]}, "output_image"),
    }
    return ComfyGraph(graph)


def _sd_img2img(*, source_image: object = "<PIL>") -> ComfyGraph:
    graph = _sd_txt2img()
    graph.raw["image_loader"] = _node("HordeImageLoader", {"image": source_image}, "image_loader")
    graph.raw["vae_encode"] = _node(
        "VAEEncode", {"vae": ["model_loader", 2], "pixels": ["image_loader", 0]}, "vae_encode"
    )
    graph.raw["sampler"]["inputs"]["latent_image"] = ["vae_encode", 0]
    return graph


def _clip_skip(graph: ComfyGraph) -> ComfyGraph:
    """Route both CLIPTextEncode nodes through a ``CLIPSetLastLayer`` (clip_skip > 1), as materialization does."""
    graph.raw["clip_skip"] = _node(
        "CLIPSetLastLayer", {"clip": ["model_loader", 1], "stop_at_clip_layer": -2}, "clip_skip"
    )
    graph.raw["prompt"]["inputs"]["clip"] = ["clip_skip", 0]
    graph.raw["negative_prompt"]["inputs"]["clip"] = ["clip_skip", 0]
    return graph


def _sd_transparency_decode(graph: ComfyGraph) -> ComfyGraph:
    """Interpose a transparency decode node that reads the sample latent AND the decoded image.

    Mirrors the layerdiffuse shape where the reused image output is fed by a node that consumes the
    sampler latent through a second edge (``samples``), so a decode cut that only repoints
    ``vae_decode.samples`` would leave the sampler (and its CLIP-side ancestors) live.
    """
    graph.raw["transparency_decode"] = _node(
        "LayeredDiffusionDecodeRGBA",
        {"samples": ["sampler", 0], "images": ["vae_decode", 0]},
        "transparency_decode",
    )
    graph.raw["output_image"]["inputs"]["images"] = ["transparency_decode", 0]
    return graph


def _clip_side_live(graph: ComfyGraph, roots: list[str]) -> list[str]:
    return sorted(
        t for t in _live_ancestors(graph, roots) if _class_type(graph, t) in ("CLIPTextEncode", "CLIPSetLastLayer")
    )


def _sd_hires() -> ComfyGraph:
    graph = _sd_txt2img()
    graph.raw["latent_upscale"] = _node(
        "LatentUpscale", {"samples": ["sampler", 0], "width": 1024, "height": 1024}, "latent_upscale"
    )
    graph.raw["upscale_sampler"] = _node(
        "KSampler",
        {
            "model": ["model_loader", 0],
            "positive": ["prompt", 0],
            "negative": ["negative_prompt", 0],
            "latent_image": ["latent_upscale", 0],
        },
        "upscale_sampler",
    )
    graph.raw["vae_decode"]["inputs"]["samples"] = ["upscale_sampler", 0]
    return graph


def _flux() -> ComfyGraph:
    """A Flux-shaped graph: combined loader and prompts, but a SamplerCustomAdvanced (no ``sampler``)."""
    graph = {
        "model_loader": _combined_loader(),
        "prompt": _node("CLIPTextEncode", {"clip": ["model_loader", 1], "text": "a"}, "prompt"),
        "negative_prompt": _node("CLIPTextEncode", {"clip": ["model_loader", 1], "text": "b"}, "negative_prompt"),
        "sampler_custom_advanced": _node(
            "SamplerCustomAdvanced", {"latent_image": ["empty", 0]}, "sampler_custom_advanced"
        ),
        "vae_decode": _node(
            "VAEDecode", {"vae": ["model_loader", 2], "samples": ["sampler_custom_advanced", 0]}, "vae_decode"
        ),
        "output_image": _node("HordeImageOutput", {"images": ["vae_decode", 0]}, "output_image"),
    }
    return ComfyGraph(graph)


def _qwen_split_loader() -> ComfyGraph:
    """A Qwen/Z-Image-shaped graph: a ``sampler`` KSampler but split CLIPLoader/VAELoader nodes."""
    graph = {
        "model_loader": _node(
            "HordeCheckpointLoader", {"horde_model_name": "m", "output_model": True}, "model_loader"
        ),
        "clip_loader": _node("CLIPLoader", {"clip_name": "c", "type": "qwen"}, "clip_loader"),
        "vae_loader": _node("VAELoader", {"vae_name": "v"}, "vae_loader"),
        "prompt": _node("CLIPTextEncode", {"clip": ["clip_loader", 0], "text": "a"}, "prompt"),
        "negative_prompt": _node("CLIPTextEncode", {"clip": ["clip_loader", 0], "text": "b"}, "negative_prompt"),
        "sampler": _node(
            "KSampler",
            {"model": ["model_loader", 0], "positive": ["prompt", 0], "negative": ["negative_prompt", 0]},
            "sampler",
        ),
        "vae_decode": _node("VAEDecode", {"vae": ["vae_loader", 0], "samples": ["sampler", 0]}, "vae_decode"),
        "output_image": _node("HordeImageOutput", {"images": ["vae_decode", 0]}, "output_image"),
    }
    return ComfyGraph(graph)


def _loader_flags(graph: ComfyGraph) -> tuple[bool, bool, bool]:
    inputs = graph.node("model_loader")["inputs"]
    return inputs["output_model"], inputs["output_clip"], inputs["output_vae"]


class TestEncodeTextCut:
    def test_txt2img_cut(self) -> None:
        graph = _sd_txt2img()
        outputs = cut_encode_text_stage(graph)

        assert _loader_flags(graph) == (False, True, False)
        assert graph.has_node("cond_positive_output")
        assert graph.has_node("cond_negative_output")
        assert not graph.has_node("output_image")
        assert {o.node for o in outputs} == {"cond_positive_output", "cond_negative_output"}
        assert all(o.kind == OutputKind.CONDITIONING for o in outputs)
        assert graph.node("cond_positive_output")["inputs"]["conditioning"] == ["prompt", 0]

    def test_split_loader_rejected(self) -> None:
        with pytest.raises(StageGraphUnsupportedError, match="split component loaders"):
            cut_encode_text_stage(_qwen_split_loader())


class TestSampleCut:
    def test_txt2img_cut(self) -> None:
        graph = _sd_txt2img()
        outputs = cut_sample_stage(graph, positive_bytes=b"p", negative_bytes=b"n", source_latent_bytes=None)

        assert _loader_flags(graph) == (True, False, False)
        assert graph.has_node("inject_positive")
        assert graph.has_node("inject_negative")
        assert graph.node("sampler")["inputs"]["positive"] == ["inject_positive", 0]
        assert graph.node("sampler")["inputs"]["negative"] == ["inject_negative", 0]
        assert not graph.has_node("output_image")
        assert [o.node for o in outputs] == ["latent_output"]
        assert outputs[0].kind == OutputKind.LATENT

    def test_hires_rewires_both_samplers(self) -> None:
        graph = _sd_hires()
        cut_sample_stage(graph, positive_bytes=b"p", negative_bytes=b"n", source_latent_bytes=None)

        for title in ("sampler", "upscale_sampler"):
            assert graph.node(title)["inputs"]["positive"] == ["inject_positive", 0]
            assert graph.node(title)["inputs"]["negative"] == ["inject_negative", 0]

    def test_img2img_with_source_latent(self) -> None:
        graph = _sd_img2img()
        cut_sample_stage(graph, positive_bytes=b"p", negative_bytes=b"n", source_latent_bytes=b"L")

        assert graph.has_node("inject_source_latent")
        assert graph.node("sampler")["inputs"]["latent_image"] == ["inject_source_latent", 0]

    def test_img2img_without_source_latent_rejected(self) -> None:
        with pytest.raises(StageGraphUnsupportedError, match="source_latent_bytes"):
            cut_sample_stage(_sd_img2img(), positive_bytes=b"p", negative_bytes=b"n", source_latent_bytes=None)

    def test_txt2img_with_source_latent_rejected(self) -> None:
        # A txt2img job must never have its empty latent displaced by an injected source latent.
        with pytest.raises(StageGraphUnsupportedError, match="txt2img-shaped"):
            cut_sample_stage(_sd_txt2img(), positive_bytes=b"p", negative_bytes=b"n", source_latent_bytes=b"L")

    def test_flux_sampler_rejected(self) -> None:
        with pytest.raises(StageGraphUnsupportedError, match="sampler"):
            cut_sample_stage(_flux(), positive_bytes=b"p", negative_bytes=b"n", source_latent_bytes=None)

    def test_split_loader_rejected(self) -> None:
        with pytest.raises(StageGraphUnsupportedError, match="split component loaders"):
            cut_sample_stage(_qwen_split_loader(), positive_bytes=b"p", negative_bytes=b"n", source_latent_bytes=None)


class TestVaeEncodeCut:
    def test_img2img_cut(self) -> None:
        graph = _sd_img2img()
        outputs = cut_vae_encode_stage(graph)

        assert _loader_flags(graph) == (False, False, True)
        assert graph.node("latent_output")["inputs"]["samples"] == ["vae_encode", 0]
        assert not graph.has_node("output_image")
        assert [o.node for o in outputs] == ["latent_output"]

    def test_txt2img_rejected(self) -> None:
        with pytest.raises(StageGraphUnsupportedError, match="VAEEncode"):
            cut_vae_encode_stage(_sd_txt2img())

    def test_effective_txt2img_no_source_image_rejected(self) -> None:
        # Materialization keeps a vae_encode node even for effective txt2img (img2img routing with no
        # source image); the image loader input is None, so the encode has no usable source to run on.
        with pytest.raises(StageGraphUnsupportedError, match="no usable source image"):
            cut_vae_encode_stage(_sd_img2img(source_image=None))


class TestDecodeCut:
    def test_cut_keeps_image_output(self) -> None:
        graph = _sd_txt2img()
        cut_decode_stage(graph, latent_bytes=b"L")

        assert _loader_flags(graph) == (False, False, True)
        assert graph.has_node("inject_latent")
        assert graph.node("vae_decode")["inputs"]["samples"] == ["inject_latent", 0]
        # decode reuses the family's image output, so it must remain.
        assert graph.has_node("output_image")

    def test_split_loader_rejected(self) -> None:
        with pytest.raises(StageGraphUnsupportedError, match="split component loaders"):
            cut_decode_stage(_qwen_split_loader(), latent_bytes=b"L")

    def test_clip_skip_txt2img_orphans_clip_side(self) -> None:
        # With clip_skip > 1 a CLIPSetLastLayer sits between the loader and the encoders; the decode
        # cut must leave none of the CLIP-side chain a live ancestor of the reused image output.
        graph = _clip_skip(_sd_txt2img())
        cut_decode_stage(graph, latent_bytes=b"L")

        assert graph.node("vae_decode")["inputs"]["samples"] == ["inject_latent", 0]
        assert _clip_side_live(graph, ["output_image"]) == []

    def test_transparency_decode_detaches_second_latent_consumer(self) -> None:
        # The reused image output reaches the sampler through the transparency node's ``samples`` edge;
        # a decode cut that only repoints vae_decode would leave clip_skip live against a null CLIP.
        graph = _sd_transparency_decode(_clip_skip(_sd_txt2img()))
        cut_decode_stage(graph, latent_bytes=b"L")

        assert graph.node("vae_decode")["inputs"]["samples"] == ["inject_latent", 0]
        assert graph.node("transparency_decode")["inputs"]["samples"] == ["inject_latent", 0]
        # The transparency node still composites over the decoded image, but no longer via the sampler.
        assert graph.node("transparency_decode")["inputs"]["images"] == ["vae_decode", 0]
        assert _clip_side_live(graph, ["output_image"]) == []
        assert "sampler" not in _live_ancestors(graph, ["output_image"])
