"""Vocabularies shared by the payload models and pipeline templates.

These are the canonical copies; ``HordeLib``'s class attributes alias them for backwards
compatibility until the legacy payload path is removed.
"""

from collections.abc import Iterable

SAMPLERS_MAP = {
    "k_euler": "euler",
    "k_euler_a": "euler_ancestral",
    "k_heun": "heun",
    "k_dpm_2": "dpm_2",
    "k_dpm_2_a": "dpm_2_ancestral",
    "k_lms": "lms",
    "k_dpm_fast": "dpm_fast",
    "k_dpm_adaptive": "dpm_adaptive",
    "k_dpmpp_2s_a": "dpmpp_2s_ancestral",
    "k_dpmpp_sde": "dpmpp_sde",
    "k_dpmpp_2m": "dpmpp_2m",
    "ddim": "ddim",
    "uni_pc": "uni_pc",
    "uni_pc_bh2": "uni_pc_bh2",
    "plms": "euler",
    "lcm": "lcm",
}
"""Horde sampler names to ComfyUI sampler names."""

# Horde control_type on the left, comfyui_controlnet_aux preprocessor on the right
CONTROLNET_IMAGE_PREPROCESSOR_MAP = {
    "canny": "CannyEdgePreprocessor",
    "hed": "HEDPreprocessor",
    "depth": "LeReS-DepthMapPreprocessor",
    "normal": "MiDaS-NormalMapPreprocessor",
    "openpose": "OpenposePreprocessor",
    "seg": "SemSegPreprocessor",
    "scribble": "ScribblePreprocessor",
    "fakescribbles": "FakeScribblePreprocessor",
    "hough": "M-LSDPreprocessor",  # horde backward compatibility
    "mlsd": "M-LSDPreprocessor",
}

CONTROLNET_MODEL_MAP = {
    "canny": "diff_control_sd15_canny_fp16.safetensors",
    "hed": "diff_control_sd15_hed_fp16.safetensors",
    "depth": "diff_control_sd15_depth_fp16.safetensors",
    "normal": "control_normal_fp16.safetensors",
    "openpose": "control_openpose_fp16.safetensors",
    "seg": "control_seg_fp16.safetensors",
    "scribble": "control_scribble_fp16.safetensors",
    "fakescribble": "control_scribble_fp16.safetensors",
    "fakescribbles": "control_scribble_fp16.safetensors",
    "mlsd": "control_mlsd_fp16.safetensors",
    "hough": "control_mlsd_fp16.safetensors",
}
"""Horde control_type to controlnet model filename."""

# Rough order-of-magnitude of the checkpoint(s) each control_type's comfyui_controlnet_aux detector
# downloads from the HuggingFace hub on first use, into AUX_ANNOTATOR_CKPTS_PATH (see hordelib.preload).
# These are *annotator* (preprocessor) weights, distinct from the controlnet model weights in
# CONTROLNET_MODEL_MAP. Used only for download/disk previews (e.g. the worker's benchmark planner), so a
# ROM estimate is sufficient; the real fetch verifies actual sizes. canny/scribble/mlsd are pure-cv2 and
# download nothing (0). Keep keys aligned with CONTROLNET_IMAGE_PREPROCESSOR_MAP.
CONTROLNET_ANNOTATOR_DOWNLOAD_BYTES = {
    "canny": 0,
    "hed": 56 * 1024**2,  # ControlNetHED.pth
    "depth": 800 * 1024**2,  # LeReS: res101.pth (~470MB) + latest_net_G.pth (~320MB)
    "normal": 470 * 1024**2,  # Intel/dpt-hybrid-midas pytorch_model.bin
    "openpose": 200 * 1024**2,  # body/hand/face pose models (needs the controlnet/onnxruntime extra)
    "seg": 170 * 1024**2,  # UniFormer segmentation
    "scribble": 0,
    "fakescribbles": 56 * 1024**2,  # FakeScribble runs the HED detector underneath
    "hough": 6 * 1024**2,  # M-LSD
    "mlsd": 6 * 1024**2,
}
"""Horde control_type to an estimated annotator-checkpoint download size in bytes (ROM)."""


def controlnet_annotator_download_bytes(control_types: Iterable[str | None]) -> int:
    """Return the summed ROM annotator-download size for *control_types*.

    Unknown or ``None`` control types contribute 0 (they may be pure-cv2 or not annotator-backed), so a
    consumer can pass a level's raw control types without filtering. Duplicates are de-duplicated first:
    an annotator is fetched once and shared on disk, so sweeping the same type twice costs it once.

    Args:
        control_types: Horde ``control_type`` values (``None`` entries allowed and ignored).

    Returns:
        The total estimated annotator download size in bytes (0 when none are annotator-backed).
    """
    distinct = {control_type for control_type in control_types if control_type}
    return sum(CONTROLNET_ANNOTATOR_DOWNLOAD_BYTES.get(control_type, 0) for control_type in distinct)


SOURCE_IMAGE_PROCESSING_OPTIONS = ["img2img", "inpainting", "outpainting", "remix"]

SCHEDULERS = ["normal", "karras", "simple", "ddim_uniform", "sgm_uniform", "exponential"]
