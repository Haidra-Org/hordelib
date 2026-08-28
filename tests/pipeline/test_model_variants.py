"""One graph per family: the per-model variables that let family members share it. No GPU required.

A family's graph carries one member's component filenames, CLIP type and shift as defaults; what a
given model loads comes from its reference record plus the profile/override tables in
``families/image_gen/baselines.py``. These tests pin that resolution and the materialized result of
the three graph-compatible baselines, which remain architecturally distinct.
"""

from typing import Any

import pytest
from horde_model_reference.meta_consts import KNOWN_IMAGE_GENERATION_BASELINE

from hordelib.pipeline.context import ModelContext
from hordelib.pipeline.families.image import build_default_registry
from hordelib.pipeline.families.image_gen.baselines import resolve_clip_type, resolve_flow_shift
from hordelib.pipeline.patches import FlowShiftNode
from hordelib.pipeline.payload import ImageGenPayload

QWEN = KNOWN_IMAGE_GENERATION_BASELINE.qwen_image
KREA2 = KNOWN_IMAGE_GENERATION_BASELINE.krea2_turbo
ANIMA = KNOWN_IMAGE_GENERATION_BASELINE.anima
Z_IMAGE = KNOWN_IMAGE_GENERATION_BASELINE.z_image_turbo
SD1 = KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_1
FLUX = KNOWN_IMAGE_GENERATION_BASELINE.flux_1

KREA2_MODEL_NAME = "Krea2-Turbo_fp8"
QWEN_DEFAULT_SHIFT = 3.1000000000000005

QWEN_FILES = {
    "vae": "../vae/qwen_image_vae.safetensors",
    "text_encoders": "../text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors",
}
KREA2_FILES = {
    "vae": "../vae/qwen_image_vae.safetensors",
    "text_encoders": "../text_encoders/qwen3vl_4b_fp8_scaled.safetensors",
}
ANIMA_FILES = {
    "vae": "../vae/qwen_image_vae.safetensors",
    "text_encoders": "../text_encoders/qwen_3_06b_base.safetensors",
}


def _materialize(
    model_name: str,
    extra_files: dict[str, str],
    *,
    baseline: KNOWN_IMAGE_GENERATION_BASELINE = QWEN,
    **payload_overrides: Any,
):
    payload = ImageGenPayload.from_horde_dict({"seed": 1, "prompt": "variant test", **payload_overrides})
    context = ModelContext(
        horde_model_name=model_name,
        baseline=baseline,
        main_file="variant_model.safetensors",
        extra_files=extra_files,
    )
    definition = build_default_registry().select(payload, context)
    assert definition is not None
    return definition, definition.materialize(payload, context)


@pytest.mark.parametrize(
    ("baseline", "model_name", "expected"),
    [
        (QWEN, "Qwen-Image_fp8", "qwen_image"),
        (Z_IMAGE, "Z-Image-Turbo", "lumina2"),
        (KREA2, KREA2_MODEL_NAME, "krea2"),
        (ANIMA, "Anima-Turbo-v1.1", "stable_diffusion"),
        (SD1, "Deliberate", None),
        ("not_a_baseline", "Deliberate", None),
        (None, None, None),
    ],
)
def test_resolve_clip_type(baseline, model_name, expected):
    assert resolve_clip_type(baseline, model_name) == expected


@pytest.mark.parametrize(
    ("baseline", "model_name", "requested", "expected"),
    [
        (QWEN, "Qwen-Image_fp8", None, (FlowShiftNode.AURA_FLOW, QWEN_DEFAULT_SHIFT)),
        (QWEN, "Qwen-Image_fp8", 2.0, (FlowShiftNode.AURA_FLOW, 2.0)),
        # Krea 2 takes no shift node, so a requested shift is carried back for the caller to warn about.
        (KREA2, KREA2_MODEL_NAME, 2.0, (None, 2.0)),
        (KREA2, KREA2_MODEL_NAME, None, (None, None)),
        (ANIMA, "Anima-Turbo-v1.1", 2.0, (None, 2.0)),
        (ANIMA, "Anima-Turbo-v1.1", None, (None, None)),
        (FLUX, "Flux.1-Schnell fp8", 2.5, (FlowShiftNode.FLUX, 2.5)),
        (FLUX, "Flux.1-Schnell fp8", None, (FlowShiftNode.FLUX, None)),
        (SD1, "Deliberate", 2.0, (None, 2.0)),
    ],
)
def test_resolve_flow_shift(baseline, model_name, requested, expected):
    assert resolve_flow_shift(baseline, model_name, requested) == expected


def test_qwen_image_loads_its_own_components_and_is_shifted():
    definition, graph = _materialize("Qwen-Image_fp8", QWEN_FILES)

    assert definition.name == "qwen"
    assert graph.node("clip_loader")["inputs"]["clip_name"] == "qwen_2.5_vl_7b_fp8_scaled.safetensors"
    assert graph.node("clip_loader")["inputs"]["type"] == "qwen_image"
    assert graph.node("vae_loader")["inputs"]["vae_name"] == "qwen_image_vae.safetensors"
    assert graph.node("model_sampling_aura_flow")["inputs"]["shift"] == QWEN_DEFAULT_SHIFT
    assert graph.node("sampler")["inputs"]["model"][0] == "model_sampling_aura_flow"


def test_qwen_image_takes_the_requested_shift():
    _, graph = _materialize("Qwen-Image_fp8", QWEN_FILES, flow_shift=2.0)

    assert graph.node("model_sampling_aura_flow")["inputs"]["shift"] == 2.0


def test_krea2_runs_the_qwen_graph_with_its_own_encoder_and_no_shift():
    definition, graph = _materialize(KREA2_MODEL_NAME, KREA2_FILES, baseline=KREA2)

    assert definition.name == "qwen"
    assert graph.node("clip_loader")["inputs"]["clip_name"] == "qwen3vl_4b_fp8_scaled.safetensors"
    assert graph.node("clip_loader")["inputs"]["type"] == "krea2"
    assert graph.node("vae_loader")["inputs"]["vae_name"] == "qwen_image_vae.safetensors"
    assert not graph.has_node("model_sampling_aura_flow")
    assert graph.node("sampler")["inputs"]["model"][0] == "model_loader"


def test_a_requested_shift_is_ignored_for_krea2():
    _, graph = _materialize(KREA2_MODEL_NAME, KREA2_FILES, baseline=KREA2, flow_shift=2.0)

    assert not graph.has_node("model_sampling_aura_flow")
    assert graph.node("sampler")["inputs"]["model"][0] == "model_loader"


def test_anima_runs_the_shared_graph_with_its_own_encoder_and_native_model_shift():
    definition, graph = _materialize("Anima-Turbo-v1.1", ANIMA_FILES, baseline=ANIMA)

    assert definition.name == "qwen"
    assert graph.node("clip_loader")["inputs"]["clip_name"] == "qwen_3_06b_base.safetensors"
    assert graph.node("clip_loader")["inputs"]["type"] == "stable_diffusion"
    assert graph.node("vae_loader")["inputs"]["vae_name"] == "qwen_image_vae.safetensors"
    assert not graph.has_node("model_sampling_aura_flow")
    assert graph.node("sampler")["inputs"]["model"][0] == "model_loader"
