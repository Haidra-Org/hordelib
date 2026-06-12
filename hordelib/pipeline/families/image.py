"""Pipeline selection for the image-generation families.

The predicates and priorities reproduce the legacy decision tree in
``HordeLib._get_appropriate_pipeline`` exactly:

1. qr_code workflow
2. cascade (remix / 2pass / base)
3. flux
4. qwen
5. controlnet (annotator / hires / base)
6. img2img-with-mask / inpainting / outpainting
7. hires fix
8. generic stable diffusion (also plain img2img)

Bindings are introduced family-by-family as parameterization moves out of
``_final_pipeline_adjustments``; selection comes first.
"""

from pathlib import Path

from hordelib.pipeline.payload import ImageGenPayload
from hordelib.pipeline.registry import PipelineRegistry, PipelineSpec
from hordelib.pipeline.template import PipelineTemplate

PIPELINES_DIR = Path(__file__).parent.parent.parent / "pipelines"


def _has_img2img_mask(payload: ImageGenPayload) -> bool:
    if payload.source_processing != "img2img":
        return False
    if payload.source_mask is not None:
        return True
    return payload.source_image is not None and len(payload.source_image.getbands()) == 4


def _template(name: str) -> PipelineTemplate:
    return PipelineTemplate(
        name=name,
        graph_file=PIPELINES_DIR / f"pipeline_{name}.json",
        bindings=(),
    )


def build_default_registry() -> PipelineRegistry:
    registry = PipelineRegistry()

    specs = [
        # Workflow overrides beat everything else
        PipelineSpec(
            template=_template("qr_code"),
            predicate=lambda p, c: p.workflow == "qr_code",
            priority=100,
        ),
        # Baseline-specific families must be checked before the generic SD fallback
        PipelineSpec(
            template=_template("stable_cascade_remix"),
            predicate=lambda p, c: c.is_cascade and p.source_processing == "remix",
            priority=90,
        ),
        PipelineSpec(
            template=_template("stable_cascade_2pass"),
            predicate=lambda p, c: c.is_cascade and p.hires_fix,
            priority=89,
        ),
        PipelineSpec(
            template=_template("stable_cascade"),
            predicate=lambda p, c: c.is_cascade,
            priority=88,
        ),
        PipelineSpec(
            template=_template("flux"),
            predicate=lambda p, c: c.is_flux,
            priority=87,
        ),
        PipelineSpec(
            template=_template("qwen"),
            predicate=lambda p, c: c.is_qwen,
            priority=86,
        ),
        # ControlNet
        PipelineSpec(
            template=_template("controlnet_annotator"),
            predicate=lambda p, c: bool(p.control_type) and p.return_control_map,
            priority=80,
        ),
        PipelineSpec(
            template=_template("controlnet_hires_fix"),
            predicate=lambda p, c: bool(p.control_type) and p.hires_fix,
            priority=79,
        ),
        PipelineSpec(
            template=_template("controlnet"),
            predicate=lambda p, c: bool(p.control_type),
            priority=78,
        ),
        # Masked / painting modes
        PipelineSpec(
            template=_template("stable_diffusion_img2img_mask"),
            predicate=lambda p, c: _has_img2img_mask(p),
            priority=70,
        ),
        PipelineSpec(
            template=_template("stable_diffusion_paint"),
            predicate=lambda p, c: p.source_processing in ("inpainting", "outpainting"),
            priority=69,
        ),
        # Generic stable diffusion
        PipelineSpec(
            template=_template("stable_diffusion_hires_fix"),
            predicate=lambda p, c: p.hires_fix,
            priority=10,
        ),
        PipelineSpec(
            template=_template("stable_diffusion"),
            predicate=lambda p, c: True,
            priority=0,
        ),
    ]

    for spec in specs:
        registry.register(spec)

    return registry


DEFAULT_REGISTRY = build_default_registry()
