import os
import torch
import numpy as np
from einops import rearrange
from .model import pidinet
from hordelib.nodes.comfy_controlnet_preprocessors.util import load_state_dict, load_file_from_url
import builtins
import model_management

netNetwork = None
remote_model_path = "https://huggingface.co/TencentARC/T2I-Adapter/resolve/main/third-party-models/table5_pidinet.pth"
modeldir = "pidinet"


def download_if_not_existed():
    modelpath = os.path.join(builtins.annotator_ckpts_path, modeldir, "table5_pidinet.pth")
    if not os.path.exists(modelpath):
        load_file_from_url(remote_model_path, model_dir=modelpath)
    return modelpath


def apply_pidinet(input_image):
    global netNetwork
    if netNetwork is None:
        netNetwork = pidinet()
        ckp = load_state_dict(download_if_not_existed())
        netNetwork.load_state_dict({k.replace("module.", ""): v for k, v in ckp.items()})

    netNetwork = netNetwork.to(model_management.get_torch_device())
    netNetwork.eval()
    assert input_image.ndim == 3
    input_image = input_image[:, :, ::-1].copy()
    with torch.no_grad():
        image_pidi = torch.from_numpy(input_image).float().to(model_management.get_torch_device())
        image_pidi = image_pidi / 255.0
        image_pidi = rearrange(image_pidi, "h w c -> 1 c h w")
        edge = netNetwork(image_pidi)[-1]
        edge = edge > 0.5
        edge = (edge * 255.0).clip(0, 255).cpu().numpy().astype(np.uint8)

    return edge[0][0]
