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
    "binary": "BinaryPreprocessor",
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
    "standard_lineart": "LineartStandardPreprocessor",
    "lineart": "LineArtPreprocessor",
    "lineart_anime": "AnimeLineArtPreprocessor",
    "lineart_anime_denoise": "Manga2Anime_LineArt_Preprocessor",
    "pidinet": "PiDiNetPreprocessor",
    "scribble_xdog": "Scribble_XDoG_Preprocessor",
    "scribble_pidinet": "Scribble_PiDiNet_Preprocessor",
    "teed": "TEEDPreprocessor",
    "pyracanny": "PyraCannyPreprocessor",
    "midas_depth": "MiDaS-DepthMapPreprocessor",
    "zoe_depth": "Zoe-DepthMapPreprocessor",
    "depth_anything": "DepthAnythingPreprocessor",
    "depth_anything_v2": "DepthAnythingV2Preprocessor",
    "normal_bae": "BAE-NormalMapPreprocessor",
    "oneformer_ade20k": "OneFormer-ADE20K-SemSegPreprocessor",
    "oneformer_coco": "OneFormer-COCO-SemSegPreprocessor",
    "recolor_luminance": "ImageLuminanceDetector",
    "recolor_intensity": "ImageIntensityDetector",
    "tile": "TilePreprocessor",
    "tile_ttplanet_guided": "TTPlanet_TileGF_Preprocessor",
    "tile_ttplanet_simple": "TTPlanet_TileSimple_Preprocessor",
}

ONNXRUNTIME_GATED_PREPROCESSORS: frozenset[str] = frozenset()
"""comfyui_controlnet_aux preprocessors that need the onnxruntime-backed ``controlnet`` extra to run.

No currently exposed preprocessor requires ONNX Runtime. ``OpenposePreprocessor`` is the classic Torch
implementation at the pinned auxiliary-node revision; DWPose is a separate, unexposed node.
"""

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
    "binary": "control_scribble_fp16.safetensors",
    "standard_lineart": "control_v11p_sd15_lineart_fp16.safetensors",
    "lineart": "control_v11p_sd15_lineart_fp16.safetensors",
    "lineart_anime": "control_v11p_sd15s2_lineart_anime_fp16.safetensors",
    "lineart_anime_denoise": "control_v11p_sd15s2_lineart_anime_fp16.safetensors",
    "pidinet": "diff_control_sd15_hed_fp16.safetensors",
    "scribble_xdog": "control_scribble_fp16.safetensors",
    "scribble_pidinet": "control_scribble_fp16.safetensors",
    "teed": "diff_control_sd15_hed_fp16.safetensors",
    "pyracanny": "diff_control_sd15_canny_fp16.safetensors",
    "midas_depth": "diff_control_sd15_depth_fp16.safetensors",
    "zoe_depth": "diff_control_sd15_depth_fp16.safetensors",
    "depth_anything": "diff_control_sd15_depth_fp16.safetensors",
    "depth_anything_v2": "diff_control_sd15_depth_fp16.safetensors",
    "normal_bae": "control_v11p_sd15_normalbae_fp16.safetensors",
    "oneformer_ade20k": "control_seg_fp16.safetensors",
    "oneformer_coco": "control_seg_fp16.safetensors",
    "recolor_luminance": "ioclab_sd15_recolor.safetensors",
    "recolor_intensity": "ioclab_sd15_recolor.safetensors",
    "tile": "control_v11f1e_sd15_tile_fp16.safetensors",
    "tile_ttplanet_guided": "control_v11f1e_sd15_tile_fp16.safetensors",
    "tile_ttplanet_simple": "control_v11f1e_sd15_tile_fp16.safetensors",
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
    "openpose": 200 * 1024**2,  # body/hand/face pose models (pure Torch at the pinned aux revision)
    "seg": 170 * 1024**2,  # UniFormer segmentation
    "scribble": 0,
    "fakescribbles": 56 * 1024**2,  # FakeScribble runs the HED detector underneath
    "hough": 6 * 1024**2,  # M-LSD
    "mlsd": 6 * 1024**2,
    "binary": 0,
    "standard_lineart": 0,
    "lineart": 35 * 1024**2,
    "lineart_anime": 208 * 1024**2,
    "lineart_anime_denoise": 165 * 1024**2,
    "pidinet": 3 * 1024**2,
    "scribble_xdog": 0,
    "scribble_pidinet": 3 * 1024**2,
    "teed": 1024**2,
    "pyracanny": 0,
    "midas_depth": 470 * 1024**2,
    "zoe_depth": 1_400 * 1024**2,
    "depth_anything": 1_400 * 1024**2,
    "depth_anything_v2": 1_280 * 1024**2,
    "normal_bae": 280 * 1024**2,
    "oneformer_ade20k": 850 * 1024**2,
    "oneformer_coco": 850 * 1024**2,
    "recolor_luminance": 0,
    "recolor_intensity": 0,
    "tile": 0,
    "tile_ttplanet_guided": 0,
    "tile_ttplanet_simple": 0,
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


# Horde upscaler name (KNOWN_UPSCALERS value) to its linear scale factor. The upscaler enlarges the
# generated image by this factor on each axis, so the post-processing activation peak scales with the
# *output* megapixels (factor**2 the generation megapixels), not the generation resolution. A 4x upscale
# of a 1 MP image produces a 16 MP tensor, which is the dominant VRAM cost of the post-processing phase
# (the model weights themselves are tens of MB). Used by the feature-impact estimate to size that peak.
UPSCALER_SCALE_FACTORS = {
    "BACKEND_DEFAULT": 4,  # the worker's default upscaler is a 4x ESRGAN; assume the larger factor
    "RealESRGAN_x4plus": 4,
    "RealESRGAN_x2plus": 2,
    "RealESRGAN_x4plus_anime_6B": 4,
    "NMKD_Siax": 4,
    "4x_AnimeSharp": 4,
    "4xNomos8kSC": 4,
    "4xLSDIRplus": 4,
    "4xNomosWebPhoto_RealPLKSR": 4,
    "4xNomos2_realplksr_dysample": 4,
    "4xNomos2_hq_dat2": 4,
    "2xModernSpanimationV1": 2,
}
"""Horde upscaler name to its linear (per-axis) scale factor (ROM)."""

_DEFAULT_UPSCALE_FACTOR = 4
"""Assumed factor for an upscaler absent from the ROM: err high so the activation peak is not under-sized."""


def upscaler_scale_factor(name: str | None) -> int:
    """Return the linear scale factor for upscaler *name*, or the conservative default when unknown.

    ``None`` returns 1 (no upscaler, no enlargement). An unrecognised upscaler returns
    :data:`_DEFAULT_UPSCALE_FACTOR` rather than 1, so a new upscaler the ROM has not learned yet
    over-reserves rather than under-reserving the post-processing activation peak.
    """
    if name is None:
        return 1
    return UPSCALER_SCALE_FACTORS.get(name, _DEFAULT_UPSCALE_FACTOR)


def max_upscale_factor(names: Iterable[str | None]) -> int:
    """Return the largest linear scale factor among *names*, or 1 when none enlarge the image.

    A job may request several upscalers; the output size (and thus the activation peak) is driven by the
    largest factor. ``None`` entries and an empty iterable contribute the no-op factor of 1.
    """
    factors = [upscaler_scale_factor(name) for name in names if name is not None]
    return max(factors) if factors else 1


SOURCE_IMAGE_PROCESSING_OPTIONS = ["img2img", "inpainting", "outpainting", "remix"]

SCHEDULERS = ["normal", "karras", "simple", "ddim_uniform", "sgm_uniform", "exponential"]
