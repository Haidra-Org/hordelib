"""Materialized-graph snapshot corpus: the image-identity gate for pipeline refactors.

Each case runs the same pure path as ``HordeLib._materialize_image_graph`` after resolution
(registry selection + ``template.materialize``) against a canned payload and a hand-built
``ModelContext``, then compares the resulting graph byte-for-byte against a committed JSON
snapshot in ``materialized_expected/``. No GPU and no ComfyUI initialisation are required.

The snapshots pin *everything* the pipeline layer emits, including historically spurious
inputs (e.g. a ``noise_seed`` set on plain KSampler nodes). That is deliberate: any change to
a snapshot means the ComfyUI prompts we submit changed, which is exactly what a structural
refactor must not do. To intentionally change materialization behavior, regenerate with::

    pytest tests/pipeline/test_materialize_snapshots.py --snapshot-update

and review the resulting diff like production code.

Notes:
    - PIL images are replaced with deterministic ``<image mode=... size=...>`` sentinels.
    - The qr_code cases stub :func:`hordelib.comfy_horde.get_node_class` so the real QR
      layout path (not the too-large fallback) is snapshotted without initialising ComfyUI.
"""

import json
from pathlib import Path
from typing import Any

import PIL.Image
import pytest
from horde_model_reference.meta_consts import KNOWN_IMAGE_GENERATION_BASELINE

from hordelib.pipeline.context import ModelContext
from hordelib.pipeline.families.image import build_default_registry
from hordelib.pipeline.patches import ResolvedLora
from hordelib.pipeline.payload import ImageGenPayload

SNAPSHOT_DIR = Path(__file__).parent / "materialized_expected"

SD1 = KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_1
SDXL = KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_xl
SD2 = KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_2_768
CASCADE = KNOWN_IMAGE_GENERATION_BASELINE.stable_cascade
FLUX = KNOWN_IMAGE_GENERATION_BASELINE.flux_1
QWEN = KNOWN_IMAGE_GENERATION_BASELINE.qwen_image
Z_IMAGE = KNOWN_IMAGE_GENERATION_BASELINE.z_image_turbo

_FIXED_SEED = 123456789

BASE_PAYLOAD: dict[str, Any] = {
    "seed": _FIXED_SEED,
    "prompt": "a deterministic snapshot prompt",
    "negative_prompt": "blurry, deterministic snapshot negative",
}


def _rgb_image() -> PIL.Image.Image:
    return PIL.Image.new("RGB", (512, 512))


def _rgba_image() -> PIL.Image.Image:
    return PIL.Image.new("RGBA", (512, 512))


def _context(
    baseline: KNOWN_IMAGE_GENERATION_BASELINE | None,
    **overrides: Any,
) -> ModelContext:
    """Create a canned resolved context, as `resolve_image_generation` would produce."""
    context_kwargs: dict[str, Any] = {
        "horde_model_name": "Snapshot Model",
        "baseline": baseline,
        "main_file": "snapshot_model.safetensors",
    }
    context_kwargs.update(overrides)
    return ModelContext(**context_kwargs)


def _cascade_context(**overrides: Any) -> ModelContext:
    return _context(
        CASCADE,
        extra_files={
            "stable_cascade_stage_b": "snapshot_stage_b.safetensors",
            "stable_cascade_stage_c": "snapshot_stage_c.safetensors",
        },
        **overrides,
    )


_TWO_LORAS = [
    ResolvedLora(filename="snapshot_lora_a.safetensors", strength_model=0.75, strength_clip=0.5),
    ResolvedLora(filename="snapshot_lora_b.safetensors", strength_model=1.0, strength_clip=1.0),
]

# (case id, payload overrides, resolved context). Cases cover every selection route plus the
# materialization variants (loras, layerdiffuse, remix, QR layout, batching, samplers).
SNAPSHOT_CASES: list[tuple[str, dict[str, Any], ModelContext]] = [
    ("plain_txt2img", {}, _context(SD1)),
    ("sdxl_plain", {}, _context(SDXL)),
    ("sd2_plain", {}, _context(SD2)),
    ("unknown_baseline", {}, _context(None)),
    ("hires_fix_sd1", {"hires_fix": True}, _context(SD1)),
    ("hires_fix_sdxl", {"hires_fix": True, "width": 1536, "height": 1536}, _context(SDXL)),
    ("hires_fix_unknown_baseline", {"hires_fix": True}, _context(None)),
    (
        "img2img",
        {"source_processing": "img2img", "source_image": _rgb_image(), "denoising_strength": 0.6},
        _context(SD1),
    ),
    (
        "img2img_mask",
        {"source_processing": "img2img", "source_image": _rgb_image(), "source_mask": _rgb_image()},
        _context(SD1),
    ),
    (
        "img2img_alpha_mask",
        {"source_processing": "img2img", "source_image": _rgba_image()},
        _context(SD1),
    ),
    (
        "inpainting",
        {"source_processing": "inpainting", "source_image": _rgba_image()},
        _context(SD1, is_inpainting_model=True),
    ),
    (
        "outpainting",
        {"source_processing": "outpainting", "source_image": _rgba_image()},
        _context(SD1, is_inpainting_model=True),
    ),
    (
        "controlnet_canny",
        {"control_type": "canny", "source_image": _rgb_image(), "source_processing": "img2img"},
        _context(SD1),
    ),
    (
        "controlnet_hires_fix",
        {"control_type": "canny", "hires_fix": True, "source_image": _rgb_image(), "source_processing": "img2img"},
        _context(SD1),
    ),
    (
        "controlnet_annotator",
        {"control_type": "canny", "return_control_map": True, "source_image": _rgb_image()},
        _context(SD1),
    ),
    (
        "controlnet_image_is_control",
        {"control_type": "canny", "image_is_control": True, "source_image": _rgb_image()},
        _context(SD1),
    ),
    ("cascade_txt2img", {}, _cascade_context()),
    ("cascade_2pass", {"hires_fix": True, "width": 1024, "height": 1024}, _cascade_context()),
    (
        "cascade_img2img",
        {"source_processing": "img2img", "source_image": _rgb_image(), "denoising_strength": 0.55},
        _cascade_context(),
    ),
    (
        "cascade_remix",
        {
            "source_processing": "remix",
            "source_image": _rgb_image(),
            "extra_source_images": [
                {"image": _rgb_image(), "strength": 0.6},
                {"image": _rgba_image(), "strength": 1.2},
            ],
        },
        _cascade_context(),
    ),
    ("flux", {"cfg_scale": 1.0, "ddim_steps": 4}, _context(FLUX)),
    # Flux "ignores" hires fix at selection time, but the shared hires patch step still
    # shrinks its empty latent to the first-pass resolution; pinned so step changes preserve it.
    ("flux_hires_fix", {"hires_fix": True, "width": 1024, "height": 1024}, _context(FLUX)),
    # Baseline families beat the controlnet tier; the controlnet step must stay a no-op here.
    (
        "cascade_controlnet",
        {"control_type": "canny", "source_image": _rgb_image()},
        _cascade_context(),
    ),
    # img2img on a graph without an image_loader: the rewire must stay a silent no-op.
    (
        "qwen_img2img",
        {"source_processing": "img2img", "source_image": _rgb_image()},
        _context(QWEN),
    ),
    (
        "flux_img2img",
        {"source_processing": "img2img", "source_image": _rgb_image(), "denoising_strength": 0.8},
        _context(FLUX),
    ),
    (
        "flux_loras",
        {},
        _context(FLUX, resolved_loras=_TWO_LORAS, will_load_loras=True),
    ),
    ("qwen", {}, _context(QWEN)),
    ("z_image", {}, _context(Z_IMAGE)),
    (
        "loras_sd",
        {},
        _context(SD1, resolved_loras=_TWO_LORAS, will_load_loras=True),
    ),
    (
        "loras_hires_fix",
        {"hires_fix": True},
        _context(SD1, resolved_loras=_TWO_LORAS[:1], will_load_loras=True),
    ),
    ("layerdiffuse_sd15", {"transparent": True}, _context(SD1)),
    ("layerdiffuse_sdxl_hires", {"transparent": True, "hires_fix": True}, _context(SDXL)),
    ("layerdiffuse_unsupported_baseline", {"transparent": True}, _cascade_context()),
    (
        "qr_default",
        {
            "workflow": "qr_code",
            "extra_texts": [{"text": "https://aihorde.net", "reference": "qr_code"}],
        },
        _context(SD1),
    ),
    (
        "qr_custom_layout",
        {
            "workflow": "qr_code",
            "width": 768,
            "height": 768,
            "extra_texts": [
                {"text": "https://aihorde.net", "reference": "qr_code"},
                {"text": "https", "reference": "protocol"},
                {"text": "circle", "reference": "module_drawer"},
                {"text": "a hidden village", "reference": "function_layer_prompt"},
                {"text": "128", "reference": "x_offset"},
                {"text": "96", "reference": "y_offset"},
                {"text": "2", "reference": "qr_border"},
            ],
        },
        _context(SDXL),
    ),
    ("batch_n_iter", {"n_iter": 3}, _context(SD1)),
    (
        "clip_skip_tiling",
        {"clip_skip": 2, "tiling": True},
        _context(SD1),
    ),
    (
        "sampler_scheduler_variant",
        {"sampler_name": "k_dpmpp_2m", "scheduler": "karras", "cfg_scale": 4.5, "ddim_steps": 24},
        _context(SDXL),
    ),
]

_STUB_QR_SIZE = 384


class _StubQRNode:
    """Stands in for ComfyQR's split node so QR layout is computed without ComfyUI."""

    def generate_qr(self, **kwargs: Any) -> tuple[None, None, None, None, None, int]:
        return (None, None, None, None, None, _STUB_QR_SIZE)


@pytest.fixture(autouse=True)
def _stub_qr_node_class(monkeypatch):
    monkeypatch.setattr("hordelib.comfy_horde.get_node_class", lambda name: _StubQRNode)


def _sanitize(value: Any) -> Any:
    """Replace non-JSON values (PIL images) with deterministic sentinels, recursively."""
    if isinstance(value, PIL.Image.Image):
        return f"<image mode={value.mode} size={value.width}x{value.height}>"
    if isinstance(value, dict):
        return {key: _sanitize(entry) for key, entry in value.items()}
    if isinstance(value, list):
        return [_sanitize(entry) for entry in value]
    return value


def _materialize_snapshot(payload_overrides: dict[str, Any], context: ModelContext) -> dict[str, Any]:
    registry = build_default_registry()
    payload = ImageGenPayload.from_horde_dict({**BASE_PAYLOAD, **payload_overrides})
    template = registry.select(payload, context)
    assert template is not None

    graph = template.materialize(payload, context)

    snapshot = {"pipeline": template.name, "graph": _sanitize(graph.to_api_dict())}
    # A snapshot that cannot round-trip through JSON means a sentinel is missing; fail loudly.
    json.dumps(snapshot)
    return snapshot


@pytest.mark.parametrize(
    ("payload_overrides", "context"),
    [case[1:] for case in SNAPSHOT_CASES],
    ids=[case[0] for case in SNAPSHOT_CASES],
)
def test_materialized_graph_matches_snapshot(payload_overrides, context, request):
    case_id = request.node.callspec.id
    snapshot_file = SNAPSHOT_DIR / f"{case_id}.json"

    snapshot = _materialize_snapshot(payload_overrides, context)

    if request.config.getoption("--snapshot-update"):
        SNAPSHOT_DIR.mkdir(exist_ok=True)
        snapshot_file.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return

    assert (
        snapshot_file.exists()
    ), f"No committed snapshot for case {case_id!r}; generate it with --snapshot-update and commit it."
    expected = json.loads(snapshot_file.read_text(encoding="utf-8"))
    assert snapshot == expected, (
        f"Materialized graph for {case_id!r} drifted from the committed snapshot. If this change "
        "is intentional, regenerate with --snapshot-update and review the diff."
    )


def test_every_committed_snapshot_has_a_case():
    """A renamed/removed case must not leave a stale snapshot silently passing."""
    if not SNAPSHOT_DIR.exists():
        pytest.skip("No snapshots generated yet")
    case_ids = {case[0] for case in SNAPSHOT_CASES}
    committed = {path.stem for path in SNAPSHOT_DIR.glob("*.json")}
    assert committed <= case_ids, f"Stale snapshots without cases: {sorted(committed - case_ids)}"
