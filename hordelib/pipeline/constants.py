"""Vocabularies shared by the payload models and pipeline templates.

These are the canonical copies; ``HordeLib``'s class attributes alias them for backwards
compatibility until the legacy payload path is removed.
"""

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

SOURCE_IMAGE_PROCESSING_OPTIONS = ["img2img", "inpainting", "outpainting", "remix"]

SCHEDULERS = ["normal", "karras", "simple", "ddim_uniform", "sgm_uniform", "exponential"]
