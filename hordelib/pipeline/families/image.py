"""Transitional shim: the image family moved to :mod:`hordelib.pipeline.families.image_gen`.

Kept for one release so existing imports of the registry keep working; import from the
``image_gen`` package (and its ``bindings``/``features``/``steps``/``baselines`` modules)
instead.
"""

from hordelib.pipeline.families.image_gen import DEFAULT_REGISTRY, IMAGE_PIPELINES, build_default_registry
from hordelib.pipeline.families.image_gen.baselines import FLUX_BASELINES, UNET_LOADER_BASELINES

__all__ = [
    "DEFAULT_REGISTRY",
    "FLUX_BASELINES",
    "IMAGE_PIPELINES",
    "UNET_LOADER_BASELINES",
    "build_default_registry",
]
