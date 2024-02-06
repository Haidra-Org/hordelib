import os

from hordelib.nodes.comfy_controlnet_preprocessors.uniformer.mmseg.apis import (
    init_segmentor,
    inference_segmentor,
    show_result_pyplot,
)
from hordelib.nodes.comfy_controlnet_preprocessors.uniformer.mmseg.core.evaluation import get_palette
import builtins

import model_management


checkpoint_file = "https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/upernet_global_small.pth"


class UniformerDetector:
    def __init__(self):
        modelpath = os.path.join(builtins.annotator_ckpts_path, "upernet_global_small.pth")
        if not os.path.exists(modelpath):
            from hordelib.nodes.comfy_controlnet_preprocessors.util import load_file_from_url

            load_file_from_url(checkpoint_file, model_dir=builtins.annotator_ckpts_path)
        config_file = os.path.join(os.path.dirname(__file__), "exp", "upernet_global_small", "config.py")
        self.model = init_segmentor(config_file, modelpath).to(model_management.get_torch_device())

    def __call__(self, img):
        result = inference_segmentor(self.model, img)
        res_img = show_result_pyplot(self.model, img, result, get_palette("ade"), opacity=1)
        return res_img
