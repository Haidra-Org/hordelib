import sys
from hordelib.nodes.comfy_controlnet_preprocessors.uniformer.mmcv.utils import Registry, build_from_cfg

PIXEL_SAMPLERS = Registry("pixel sampler", sys.modules[__name__])


def build_pixel_sampler(cfg, **default_args):
    """Build pixel sampler for segmentation map."""
    return build_from_cfg(cfg, PIXEL_SAMPLERS, default_args)
