import os
from urllib.parse import urlparse

import cv2
import numpy as np
import torch

from .enums import ResizeMode
from hordelib.shared_model_manager import SharedModelManager


def rgba2rgbfp32(x):
    rgb = x[..., :3].astype(np.float32) / 255.0
    a = x[..., 3:4].astype(np.float32) / 255.0
    return 0.5 + (rgb - 0.5) * a


def to255unit8(x):
    return (x * 255.0).clip(0, 255).astype(np.uint8)


def safe_numpy(x):
    # A very safe method to make sure that Apple/Mac works
    y = x

    # below is very boring but do not change these. If you change these Apple or Mac may fail.
    y = y.copy()
    y = np.ascontiguousarray(y)
    y = y.copy()
    return y


def high_quality_resize(x, size):
    if x.shape[0] != size[1] or x.shape[1] != size[0]:
        if (size[0] * size[1]) < (x.shape[0] * x.shape[1]):
            interpolation = cv2.INTER_AREA
        else:
            interpolation = cv2.INTER_LANCZOS4

        y = cv2.resize(x, size, interpolation=interpolation)
    else:
        y = x
    return y


def crop_and_resize_image(detected_map, resize_mode, h, w):
    if resize_mode == ResizeMode.RESIZE:
        detected_map = high_quality_resize(detected_map, (w, h))
        detected_map = safe_numpy(detected_map)
        return detected_map

    old_h, old_w, _ = detected_map.shape
    old_w = float(old_w)
    old_h = float(old_h)
    k0 = float(h) / old_h
    k1 = float(w) / old_w

    def safeint(x):
        return int(np.round(x))

    if resize_mode == ResizeMode.RESIZE_AND_FILL:
        k = min(k0, k1)
        borders = np.concatenate(
            [detected_map[0, :, :], detected_map[-1, :, :], detected_map[:, 0, :], detected_map[:, -1, :]], axis=0,
        )
        high_quality_border_color = np.median(borders, axis=0).astype(detected_map.dtype)
        high_quality_background = np.tile(high_quality_border_color[None, None], [h, w, 1])
        detected_map = high_quality_resize(detected_map, (safeint(old_w * k), safeint(old_h * k)))
        new_h, new_w, _ = detected_map.shape
        pad_h = max(0, (h - new_h) // 2)
        pad_w = max(0, (w - new_w) // 2)
        high_quality_background[pad_h : pad_h + new_h, pad_w : pad_w + new_w] = detected_map
        detected_map = high_quality_background
        detected_map = safe_numpy(detected_map)
        return detected_map
    else:
        k = max(k0, k1)
        detected_map = high_quality_resize(detected_map, (safeint(old_w * k), safeint(old_h * k)))
        new_h, new_w, _ = detected_map.shape
        pad_h = max(0, (new_h - h) // 2)
        pad_w = max(0, (new_w - w) // 2)
        detected_map = detected_map[pad_h : pad_h + h, pad_w : pad_w + w]
        detected_map = safe_numpy(detected_map)
        return detected_map


def pytorch_to_numpy(x):
    return [np.clip(255.0 * y.cpu().numpy(), 0, 255).astype(np.uint8) for y in x]


def numpy_to_pytorch(x):
    y = x.astype(np.float32) / 255.0
    y = y[None]
    y = np.ascontiguousarray(y.copy())
    y = torch.from_numpy(y).float()
    return y

# IMPORTANT: Edited by hordelib devs to use model loader!
# To keep things more maintainable in case we need to update the code
# We let the same incoming variables, but ignore them in favour of our own
# model manager paths
def load_file_from_url(
    url: str,
    *,
    model_dir: str,
    progress: bool = True,
    file_name: str | None = None,
) -> str:
    """Download a file from `url` into `model_dir`, using the file present if possible.

    Returns the path to the downloaded file.
    """
    return str(SharedModelManager.manager.miscellaneous.model_folder_path / file_name)


def to_lora_patch_dict(state_dict: dict) -> dict:
    """Convert raw lora state_dict to patch_dict that can be applied on
    modelpatcher."""
    patch_dict = {}
    for k, w in state_dict.items():
        model_key, patch_type, weight_index = k.split("::")
        if model_key not in patch_dict:
            patch_dict[model_key] = {}
        if patch_type not in patch_dict[model_key]:
            patch_dict[model_key][patch_type] = [None] * 16
        patch_dict[model_key][patch_type][int(weight_index)] = w

    patch_flat = {}
    for model_key, v in patch_dict.items():
        for patch_type, weight_list in v.items():
            patch_flat[model_key] = (patch_type, weight_list)

    return patch_flat
