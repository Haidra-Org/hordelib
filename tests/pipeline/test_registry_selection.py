"""Table-driven pipeline selection tests, replacing implicit knowledge of the legacy if/elif
tree. No GPU required."""

import PIL.Image
import pytest
from horde_model_reference.meta_consts import KNOWN_IMAGE_GENERATION_BASELINE

from hordelib.pipeline.context import ModelContext
from hordelib.pipeline.families.image import build_default_registry
from hordelib.pipeline.payload import ImageGenPayload

SD1 = KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_1
SDXL = KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_xl
CASCADE = KNOWN_IMAGE_GENERATION_BASELINE.stable_cascade
FLUX = KNOWN_IMAGE_GENERATION_BASELINE.flux_1
QWEN = KNOWN_IMAGE_GENERATION_BASELINE.qwen_image


def _rgba_image() -> PIL.Image.Image:
    return PIL.Image.new("RGBA", (8, 8))


def _rgb_image() -> PIL.Image.Image:
    return PIL.Image.new("RGB", (8, 8))


SELECTION_CASES = [
    # (test id, payload dict, baseline, expected pipeline)
    ("plain_txt2img", {}, SD1, "stable_diffusion"),
    ("hires_fix", {"hires_fix": True}, SD1, "stable_diffusion_hires_fix"),
    ("img2img_no_mask", {"source_processing": "img2img", "source_image": _rgb_image()}, SD1, "stable_diffusion"),
    (
        "img2img_with_mask",
        {"source_processing": "img2img", "source_image": _rgb_image(), "source_mask": _rgb_image()},
        SD1,
        "stable_diffusion_img2img_mask",
    ),
    (
        "img2img_alpha_mask",
        {"source_processing": "img2img", "source_image": _rgba_image()},
        SD1,
        "stable_diffusion_img2img_mask",
    ),
    ("inpainting", {"source_processing": "inpainting"}, SD1, "stable_diffusion_paint"),
    ("outpainting", {"source_processing": "outpainting"}, SD1, "stable_diffusion_paint"),
    ("controlnet", {"control_type": "canny"}, SD1, "controlnet"),
    ("controlnet_hires", {"control_type": "canny", "hires_fix": True}, SD1, "controlnet_hires_fix"),
    (
        "controlnet_annotator",
        {"control_type": "canny", "return_control_map": True},
        SD1,
        "controlnet_annotator",
    ),
    (
        "annotator_beats_hires",
        {"control_type": "canny", "return_control_map": True, "hires_fix": True},
        SD1,
        "controlnet_annotator",
    ),
    ("cascade", {}, CASCADE, "stable_cascade"),
    ("cascade_hires", {"hires_fix": True}, CASCADE, "stable_cascade_2pass"),
    ("cascade_remix", {"source_processing": "remix"}, CASCADE, "stable_cascade_remix"),
    ("cascade_remix_beats_hires", {"source_processing": "remix", "hires_fix": True}, CASCADE, "stable_cascade_remix"),
    # Baseline families beat controlnet (matches the legacy tree's check order)
    ("cascade_beats_controlnet", {"control_type": "canny"}, CASCADE, "stable_cascade"),
    ("flux", {}, FLUX, "flux"),
    ("flux_ignores_hires", {"hires_fix": True}, FLUX, "flux"),
    ("qwen", {}, QWEN, "qwen"),
    ("qr_workflow", {"workflow": "qr_code"}, SD1, "qr_code"),
    ("qr_beats_everything", {"workflow": "qr_code", "control_type": "canny", "hires_fix": True}, CASCADE, "qr_code"),
    (
        "creative_upscale_workflow",
        {"workflow": "creative_upscale", "source_image": _rgb_image()},
        SD1,
        "creative_upscale",
    ),
    (
        "creative_upscale_beats_features",
        {"workflow": "creative_upscale", "source_image": _rgb_image(), "control_type": "canny", "hires_fix": True},
        SDXL,
        "creative_upscale",
    ),
    ("sdxl_plain", {}, SDXL, "stable_diffusion"),
    ("unknown_baseline", {}, None, "stable_diffusion"),
]


@pytest.mark.parametrize(
    ("payload_dict", "baseline", "expected"),
    [case[1:] for case in SELECTION_CASES],
    ids=[case[0] for case in SELECTION_CASES],
)
def test_selection(payload_dict, baseline, expected):
    registry = build_default_registry()
    payload = ImageGenPayload.from_horde_dict(payload_dict)
    context = ModelContext(horde_model_name="some model", baseline=baseline)

    definition = registry.select(payload, context)

    assert definition is not None
    assert definition.name == expected


EXPECTED_SELECTION_ORDER = [
    "qr_code",
    "creative_upscale",
    "stable_cascade_remix",
    "stable_cascade_2pass",
    "stable_cascade",
    "flux",
    "qwen",
    "z_image",
    "controlnet_annotator",
    "controlnet_hires_fix",
    "controlnet",
    "stable_diffusion_img2img_mask",
    "stable_diffusion_paint",
    "stable_diffusion_hires_fix",
    "stable_diffusion",
]
"""The full selection precedence order, pinned. A tier/order change that reshuffles the
registry is a behavior change and must be made deliberately here."""


def test_selection_precedence_order_is_pinned():
    registry = build_default_registry()
    assert [definition.name for definition in registry.all_definitions()] == EXPECTED_SELECTION_ORDER


def test_every_registered_graph_file_exists():
    registry = build_default_registry()
    for definition in registry.all_definitions():
        assert definition.graph_file.exists(), f"Missing graph file for {definition.name}"


def test_all_graphs_load_and_have_an_image_output():
    registry = build_default_registry()
    for definition in registry.all_definitions():
        graph = definition.load_graph()
        # Results are collected from HordeImageOutput nodes (SaveImage pre-replacement),
        # regardless of their title
        assert "HordeImageOutput" in graph.class_types(), f"{definition.name} has no image output node"
