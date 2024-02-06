# Copyright (c) OpenMMLab. All rights reserved.
import sys
from torch.nn.parallel import DataParallel, DistributedDataParallel

from hordelib.nodes.comfy_controlnet_preprocessors.uniformer.mmcv.utils import Registry

MODULE_WRAPPERS = Registry("module wrapper", sys.modules[__name__])
MODULE_WRAPPERS.register_module(module=DataParallel)
MODULE_WRAPPERS.register_module(module=DistributedDataParallel)
