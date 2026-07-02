"""The image-generation pipeline family.

One module per pipeline (or per variant group) colocates everything a pipeline declares:
graph reference, exposed parameter bindings (composed from the named groups in
:mod:`.bindings`), declarative selection (:mod:`.features`), patch steps (:mod:`.steps`),
and outputs. Adding a pipeline means adding a module and listing its definition in
:data:`IMAGE_PIPELINES`; see ``docs/adding-a-pipeline.md``.

Selection precedence is pinned by ``tests/pipeline/test_registry_selection.py`` and the
materialized-graph snapshots in ``tests/pipeline/materialized_expected/``.
"""

from hordelib.pipeline.context import ModelContext
from hordelib.pipeline.definition import PipelineDefinition
from hordelib.pipeline.families.image_gen import (
    cascade,
    controlnet,
    creative_upscale,
    flux,
    qr_code,
    qwen,
    stable_diffusion,
    z_image,
)
from hordelib.pipeline.identifiers import ImagePipeline
from hordelib.pipeline.payload import ImageGenPayload
from hordelib.pipeline.registry import PipelineRegistry

__all__ = [
    "DEFAULT_REGISTRY",
    "IMAGE_PIPELINES",
    "build_default_registry",
]

type ImageGenDefinition = PipelineDefinition[ImageGenPayload, ModelContext]

IMAGE_PIPELINES: tuple[ImageGenDefinition, ...] = (
    qr_code.QR_CODE,
    creative_upscale.CREATIVE_UPSCALE,
    cascade.STABLE_CASCADE_REMIX,
    cascade.STABLE_CASCADE_2PASS,
    cascade.STABLE_CASCADE,
    flux.FLUX,
    qwen.QWEN,
    z_image.Z_IMAGE,
    controlnet.CONTROLNET_ANNOTATOR,
    controlnet.CONTROLNET_HIRES_FIX,
    controlnet.CONTROLNET,
    stable_diffusion.STABLE_DIFFUSION_IMG2IMG_MASK,
    stable_diffusion.STABLE_DIFFUSION_PAINT,
    stable_diffusion.STABLE_DIFFUSION_HIRES_FIX,
    stable_diffusion.STABLE_DIFFUSION,
)
"""Every image pipeline, explicitly listed (no discovery magic)."""


def build_default_registry() -> PipelineRegistry[ImageGenPayload, ModelContext]:
    """Create a registry holding all image pipelines, auditing each against its graph."""
    registry: PipelineRegistry[ImageGenPayload, ModelContext] = PipelineRegistry(payload_types=(ImageGenPayload,))
    for definition in IMAGE_PIPELINES:
        registry.register(definition)
    return registry


DEFAULT_REGISTRY = build_default_registry()

# The public ImagePipeline vocabulary must be exactly the registered pipelines; drift is a
# programming error surfaced at import time, in the same spirit as the registry's own
# definition audits (mirrored by a named test in tests/pipeline/).
_registered_pipeline_names = {definition.name for definition in IMAGE_PIPELINES}
_enumerated_pipeline_names = {member.value for member in ImagePipeline}
if _registered_pipeline_names != _enumerated_pipeline_names:
    raise RuntimeError(
        "ImagePipeline enum is out of sync with IMAGE_PIPELINES: "
        f"{sorted(_registered_pipeline_names ^ _enumerated_pipeline_names)}",
    )
