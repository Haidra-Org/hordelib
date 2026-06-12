"""Audit: every class_type in the packaged pipelines must be a known, installable node.

This converts "pipeline references a node that no longer exists" from a cryptic runtime
validation error into a CI failure. If you add a node type to a pipeline, add it here too —
and make sure it is provided by ComfyUI itself, hordelib's first-party nodes, or one of the
custom-node packages pinned in hordelib/installation/manifest.json.
"""

import json
from pathlib import Path

PIPELINES_DIR = Path(__file__).parent.parent.parent / "hordelib" / "pipelines"

COMFYUI_BUILTIN_NODES = {
    "BasicScheduler",
    "CFGGuider",
    "CheckpointLoaderSimple",
    "CLIPLoader",
    "CLIPSetLastLayer",
    "CLIPTextEncode",
    "CLIPVisionEncode",
    "ConditioningZeroOut",
    "ControlNetApply",
    "ControlNetApplyAdvanced",
    "ControlNetLoader",
    "DiffControlNetLoader",
    "EmptyLatentImage",
    "EmptySD3LatentImage",
    "GrowMask",
    "ImageBlur",
    "ImageCompositeMasked",
    "ImageScale",
    "ImageToMask",
    "ImageUpscaleWithModel",
    "KSampler",
    "KSamplerAdvanced",
    "KSamplerSelect",
    "LatentCompositeMasked",
    "LatentUpscale",
    "LoadImage",
    "MaskComposite",
    "MaskToImage",
    "ModelSamplingAuraFlow",
    "RandomNoise",
    "RepeatImageBatch",
    "SamplerCustomAdvanced",
    "SaveImage",
    "SetLatentNoiseMask",
    "SolidMask",
    "StableCascade_EmptyLatentImage",
    "StableCascade_StageB_Conditioning",
    "StableCascade_StageC_VAEEncode",
    "unCLIPCheckpointLoader",
    "unCLIPConditioning",
    "UNETLoader",
    "UpscaleModelLoader",
    "VAEDecode",
    "VAEEncode",
    "VAEEncodeForInpaint",
    "VAELoader",
}

HORDE_FIRST_PARTY_NODES = {
    "HordeCheckpointLoader",
    "HordeImageLoader",
    "HordeImageOutput",
    "HordeLoraLoader",
    "HordeDiffControlNetLoader",
    "HordeUpscaleModelLoader",
}

# Vendored under hordelib/nodes (facerestore_cf, comfyui_layerdiffuse)
VENDORED_NODES = {
    "FaceRestoreCFWithModel",
    "FaceRestoreModelLoader",
    "LayeredDiffusionApply",
    "LayeredDiffusionDecodeRGBA",
}

# Provided by the custom-node packages pinned in hordelib/installation/manifest.json
MANIFEST_NODES = {
    # comfyui_controlnet_aux
    "AIO_Preprocessor",
    # ComfyQR
    "comfy-qr-by-module-split",
    "comfy-qr-by-module-size",
    "comfy-qr-by-image-size",
    "comfy-qr-mask_errors",
}

ALLOWED_CLASS_TYPES = COMFYUI_BUILTIN_NODES | HORDE_FIRST_PARTY_NODES | VENDORED_NODES | MANIFEST_NODES


def test_all_pipeline_class_types_are_known():
    unknown: dict[str, set[str]] = {}
    for pipeline_file in sorted(PIPELINES_DIR.glob("pipeline_*.json")):
        graph = json.loads(pipeline_file.read_text(encoding="utf-8"))
        class_types = {node["class_type"] for node in graph.values() if "class_type" in node}
        unexpected = class_types - ALLOWED_CLASS_TYPES
        if unexpected:
            unknown[pipeline_file.name] = unexpected

    assert not unknown, f"Pipelines reference unknown node class_types: {unknown}"
