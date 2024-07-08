# horde.py
# Main interface for the horde to this library.
from __future__ import annotations

import base64
import io
import json
import random
import sys
import typing
from collections.abc import Callable
from copy import deepcopy
from enum import Enum, auto
from types import FunctionType

from horde_sdk.ai_horde_api.apimodels import ImageGenerateJobPopResponse
from horde_sdk.ai_horde_api.apimodels.base import (
    GenMetadataEntry,
)
from horde_sdk.ai_horde_api.consts import KNOWN_FACEFIXERS, KNOWN_UPSCALERS, METADATA_TYPE, METADATA_VALUE
from loguru import logger
from PIL import Image
from pydantic import BaseModel

from hordelib.comfy_horde import Comfy_Horde
from hordelib.consts import MODEL_CATEGORY_NAMES
from hordelib.nodes.comfy_qr.qr_nodes import QRByModuleSizeSplitFunctionPatterns
from hordelib.shared_model_manager import SharedModelManager
from hordelib.utils.dynamicprompt import DynamicPromptParser
from hordelib.utils.image_utils import ImageUtils
from hordelib.utils.ioredirect import ComfyUIProgress


class ProgressState(Enum):
    """The state of the progress report"""

    started = auto()
    progress = auto()
    post_processing = auto()
    finished = auto()


class ProgressReport(BaseModel):
    """A progress message sent to a callback"""

    hordelib_progress_state: ProgressState
    comfyui_progress: ComfyUIProgress | None = None
    progress: float | None = None
    hordelib_message: str | None = None


class ResultingImageReturn:
    image: Image.Image | None
    rawpng: io.BytesIO | None
    faults: list[GenMetadataEntry]

    def __init__(
        self,
        image: Image.Image | None,
        rawpng: io.BytesIO | None,
        faults: list[GenMetadataEntry],
    ):
        if faults is None:
            faults = []

        for fault in faults:
            if not isinstance(fault, GenMetadataEntry):
                raise TypeError("faults must be a list of GenMetadataEntry")

        if image is not None and not isinstance(image, Image.Image):
            raise TypeError("image must be a PIL.Image.Image")

        if rawpng is not None and not isinstance(rawpng, io.BytesIO):
            raise TypeError("rawpng must be a io.BytesIO")

        self.image = image
        self.rawpng = rawpng
        self.faults = faults


def _calc_upscale_sampler_steps(payload):
    """Calculates the amount of hires_fix upscaler steps based on the denoising used and the steps used for the
    primary image"""
    upscale_steps = round(payload["ddim_steps"] * (0.9 - payload["hires_fix_denoising_strength"]))
    if upscale_steps < 3:
        upscale_steps = 3

    logger.debug(f"Upscale steps calculated as {upscale_steps}")
    return upscale_steps


class HordeLib:
    _instance: HordeLib | None = None
    _initialised = False

    # Horde to comfy sampler mapping
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

    # Horde names on the left, our node names on the right
    # We use this to dynamically route the image through the
    # right node by reconnect inputs.
    CONTROLNET_IMAGE_PREPROCESSOR_MAP = {
        "canny": "canny",
        "hed": "hed",
        "depth": "depth",
        "normal": "normal",
        "openpose": "openpose",
        "seg": "seg",
        "scribble": "scribble",
        "fakescribbles": "fakescribble",
        "hough": "mlsd",  # horde backward compatibility
        "mlsd": "mlsd",
        # "<unused>": "MiDaS-DepthMapPreprocessor",
        # "<unused>": "MediaPipe-HandPosePreprocessor",
        # "<unused>": "MediaPipe-FaceMeshPreprocessor",
        # "<unused>": "BinaryPreprocessor",
        # "<unused>": "ColorPreprocessor",
        # "<unused>": "PiDiNetPreprocessor",
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
        "mlsd": "control_mlsd_fp16.safetensors",
        "hough": "control_mlsd_fp16.safetensors",
    }

    SOURCE_IMAGE_PROCESSING_OPTIONS = ["img2img", "inpainting", "outpainting", "remix"]

    SCHEDULERS = ["normal", "karras", "simple", "ddim_uniform", "sgm_uniform", "exponential"]

    # Describe a valid payload, it's types and bounds. All incoming payload data is validated against,
    # and normalised to, this schema.
    PAYLOAD_SCHEMA = {
        "sampler_name": {"datatype": str, "values": list(SAMPLERS_MAP.keys()), "default": "k_euler"},
        "cfg_scale": {"datatype": float, "min": 1, "max": 100, "default": 8.0},
        "denoising_strength": {"datatype": float, "min": 0.01, "max": 1.0, "default": 1.0},
        "control_strength": {"datatype": float, "min": 0.01, "max": 3.0, "default": 1.0},
        "seed": {"datatype": int, "default": random.randint(0, sys.maxsize)},
        "width": {"datatype": int, "min": 64, "max": 8192, "default": 512, "divisible": 64},
        "height": {"datatype": int, "min": 64, "max": 8192, "default": 512, "divisible": 64},
        "hires_fix": {"datatype": bool, "default": False},
        "clip_skip": {"datatype": int, "min": 1, "max": 20, "default": 1},
        "control_type": {"datatype": str, "values": list(CONTROLNET_IMAGE_PREPROCESSOR_MAP.keys()), "default": None},
        "image_is_control": {"datatype": bool, "default": False},
        "return_control_map": {"datatype": bool, "default": False},
        "prompt": {"datatype": str, "default": ""},
        "negative_prompt": {"datatype": str, "default": ""},
        "loras": {"datatype": list, "default": []},
        "tis": {"datatype": list, "default": []},
        "ddim_steps": {"datatype": int, "min": 1, "max": 500, "default": 30},
        "n_iter": {"datatype": int, "min": 1, "max": 100, "default": 1},
        "model": {"datatype": str, "default": "stable_diffusion"},
        "source_mask": {"datatype": Image.Image, "default": None},
        "source_image": {"datatype": Image.Image, "default": None},
        "source_processing": {"datatype": str, "values": SOURCE_IMAGE_PROCESSING_OPTIONS, "default": None},
        "hires_fix_denoising_strength": {"datatype": float, "min": 0.01, "max": 1.0, "default": 0.65},
        "scheduler": {"datatype": str, "values": SCHEDULERS, "default": "normal"},
        "tiling": {"datatype": bool, "default": False},
        "model_name": {"datatype": str, "default": "stable_diffusion"},  # Used internally by hordelib
        "stable_cascade_stage_b": {"datatype": str, "default": None},  # Stable Cascade
        "stable_cascade_stage_c": {"datatype": str, "default": None},  # Stable Cascade
        "extra_source_images": {"datatype": list, "default": []},  # Stable Cascade Remix
        "extra_texts": {"datatype": list, "default": []},  # QR Codes (for now)
        "workflow": {"datatype": str, "default": "auto_detect"},
        "transparent": {"datatype": bool, "default": False},
    }

    EXTRA_IMAGES_SCHEMA = {
        "image": {"datatype": Image.Image, "default": None},
        "strength": {"datatype": float, "min": 0.0, "max": 5.0, "default": 1.0},
    }

    EXTRA_TEXTS_SCHEMA = {
        "text": {"datatype": str, "default": ""},
        "reference": {"datatype": str, "default": None},
    }

    LORA_SCHEMA = {
        "name": {"datatype": str, "default": ""},
        "model": {"datatype": float, "min": -10.0, "max": 10.0, "default": 1.0},
        "clip": {"datatype": float, "min": -10.0, "max": 10.0, "default": 1.0},
        "inject_trigger": {"datatype": str},
        "is_version": {"datatype": bool},
    }

    TIS_SCHEMA = {
        "name": {"datatype": str, "default": ""},
        "inject_ti": {"datatype": str},
        "strength": {"datatype": float, "min": -10, "max": 10.0, "default": 1.0},
    }

    # pipeline parameter <- hordelib payload parameter mapping
    PAYLOAD_TO_PIPELINE_PARAMETER_MAPPING = {  # FIXME
        "sampler.sampler_name": "sampler_name",
        "sampler.cfg": "cfg_scale",
        "sampler.denoise": "denoising_strength",
        "sampler.seed": "seed",
        "sampler.noise_seed": "seed",
        "empty_latent_image.height": "height",
        "empty_latent_image.width": "width",
        "sampler.scheduler": "scheduler",
        "clip_skip.stop_at_clip_layer": "clip_skip",
        "prompt.text": "prompt",
        "negative_prompt.text": "negative_prompt",
        "sampler.steps": "ddim_steps",
        "empty_latent_image.batch_size": "n_iter",
        "repeat_image_batch.amount": "n_iter",
        "model_loader.ckpt_name": "model",
        "model_loader.model_name": "model",
        "model_loader.horde_model_name": "model_name",
        "model_loader.seamless_tiling_enabled": "tiling",
        "image_loader.image": "source_image",
        "loras": "loras",
        "tis": "tis",
        "upscale_sampler.denoise": "hires_fix_denoising_strength",
        "upscale_sampler.seed": "seed",
        "upscale_sampler.cfg": "cfg_scale",
        "upscale_sampler.steps": _calc_upscale_sampler_steps,
        "upscale_sampler.sampler_name": "sampler_name",
        "controlnet_apply.strength": "control_strength",
        "controlnet_model_loader.control_net_name": "control_type",
        # Stable Cascade
        "stable_cascade_empty_latent_image.width": "width",
        "stable_cascade_empty_latent_image.height": "height",
        "stable_cascade_empty_latent_image.batch_size": "n_iter",
        "sc_image_loader.image": "source_image",
        "sc_image_loader_0.image": "source_image",
        "sampler_stage_c.sampler_name": "sampler_name",
        "sampler_stage_b.sampler_name": "sampler_name",
        "sampler_stage_c.cfg": "cfg_scale",
        "sampler_stage_c.denoise": "denoising_strength",
        "sampler_stage_b.seed": "seed",
        "sampler_stage_c.seed": "seed",
        "sampler_stage_b.steps": "ddim_steps*0.33",
        "sampler_stage_c.steps": "ddim_steps*0.67",
        "model_loader_stage_c.ckpt_name": "stable_cascade_stage_c",
        "model_loader_stage_c.model_name": "stable_cascade_stage_c",
        "model_loader_stage_c.horde_model_name": "model_name",
        "model_loader_stage_b.ckpt_name": "stable_cascade_stage_b",
        "model_loader_stage_b.model_name": "stable_cascade_stage_b",
        "model_loader_stage_b.horde_model_name": "model_name",
        # Stable Cascade 2pass
        "2pass_sampler_stage_c.sampler_name": "sampler_name",
        "2pass_sampler_stage_c.steps": "ddim_steps*0.67",
        "2pass_sampler_stage_c.denoise": "hires_fix_denoising_strength",
        "2pass_sampler_stage_b.sampler_name": "sampler_name",
        "2pass_sampler_stage_b.steps": "ddim_steps*0.33",
        # QR Codes
        "sampler_bg.sampler_name": "sampler_name",
        "sampler_bg.cfg": "cfg_scale",
        "sampler_bg.denoise": "denoising_strength",
        "sampler_bg.seed": "seed",
        "sampler_bg.steps": "ddim_steps",
        "sampler_bg.noise_seed": "seed",
        "sampler_fg.sampler_name": "sampler_name",
        "sampler_fg.cfg": "cfg_scale",
        "sampler_fg.denoise": "denoising_strength",
        "sampler_fg.seed": "seed",
        "sampler_fg.steps": "ddim_steps",
        "sampler_fg.noise_seed": "seed",
        "controlnet_bg.strength": "control_strength",
        "solidmask_grey.width": "width",
        "solidmask_grey.height": "height",
        "solidmask_white.width": "width",
        "solidmask_white.height": "height",
        "solidmask_black.width": "width",
        "solidmask_black.height": "height",
        "qr_code_split.max_image_size": "width",
    }

    _comfyui_callback: Callable[[str, dict, str], None] | None = None

    # We are a singleton
    def __new__(
        cls,
        *,
        comfyui_callback: Callable[[str, dict, str], None] | None = None,
    ):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._comfyui_callback = comfyui_callback

        return cls._instance

    # We initialise only ever once (in the lifetime of the singleton)
    def __init__(
        self,
        *,
        comfyui_callback: Callable[[str, dict, str], None] | None = None,
    ):
        if not self._initialised:
            self.generator = Comfy_Horde(
                comfyui_callback=comfyui_callback if comfyui_callback else self._comfyui_callback,
            )
            self.__class__._initialised = True

    def _json_hack(self, obj):
        # Helper to serialise json which contains non-serialisable types
        if hasattr(obj, "__class__"):
            return f"{obj.__class__.__name__} instance"
        return f"Object of type {type(obj).__name__}"

    def dump_json(self, adict):
        logger.warning(json.dumps(adict, indent=4, default=self._json_hack))

    def _validate(self, value, datatype, min=None, max=None, default=None, values=None, divisible=None):
        """Check the given value against the given constraints. Return the fixed value."""

        # First, if we are passed no value at all, use the default
        if value is None:
            return default

        # We have a value, check the type
        if not isinstance(value, datatype):
            # try to coerce the type
            try:
                value = datatype(value)
            except (ValueError, TypeError):
                # epic fail, use the default
                return default

        # If the value must be divisible by some amount, assert that
        if divisible:
            if value % divisible != 0:
                value = ((value + (divisible - 1)) // divisible) * divisible

        # If we have a minimum, assert it
        if min is not None and value < min:
            value = min

        # If we have a maximum, assert it
        if max and value > max:
            value = max

        # If we have a list of allowed values, make sure the value is permitted
        if values:
            if isinstance(value, str):
                if value.lower() not in values:
                    # not an allowed value, use the default
                    return default
                value = value.lower()
            else:
                if value not in values:
                    # not an allowed value, use the default
                    return default

        return value

    def _validate_data_structure(self, data, schema_definition=PAYLOAD_SCHEMA):
        """Validate a data structure, assert parameters fall within the allowed bounds."""
        data = deepcopy(data)

        # Remove anything from the payload that isn't in our schema
        for key in data.copy().keys():
            if key.lower() not in schema_definition.keys():
                del data[key]

        # Build a valid schema payload, create attributes that don't exist using default values
        for key, schema in schema_definition.items():
            data[key.lower()] = self._validate(data.get(key.lower()), **schema)

        # Do the same for loras, if we have loras in this data structure
        if data.get("loras"):
            for i, lora in enumerate(data.get("loras")):
                data["loras"][i] = self._validate_data_structure(lora, HordeLib.LORA_SCHEMA)
            # Remove invalid loras
            data["loras"] = [x for x in data["loras"] if x.get("name")]

        # Do the same for tis, if we have tis in this data structure
        if data.get("tis"):
            for i, ti in enumerate(data.get("tis")):
                data["tis"][i] = self._validate_data_structure(ti, HordeLib.TIS_SCHEMA)
            # Remove invalid tis
            data["tis"] = [x for x in data["tis"] if x.get("name")]

        # Do the same for extra images, if we have them in this data structure
        if data.get("extra_source_images"):
            for i, img in enumerate(data.get("extra_source_images")):
                data["extra_source_images"][i] = self._validate_data_structure(img, HordeLib.EXTRA_IMAGES_SCHEMA)
            data["extra_source_images"] = [x for x in data["extra_source_images"] if x.get("image")]

        # Do the same for extra texts, if we have them in this data structure
        if data.get("extra_texts"):
            for i, img in enumerate(data.get("extra_texts")):
                data["extra_texts"][i] = self._validate_data_structure(img, HordeLib.EXTRA_TEXTS_SCHEMA)
            data["extra_texts"] = [x for x in data["extra_texts"] if x.get("text")]

        return data

    def _apply_aihorde_compatibility_hacks(self, payload: dict) -> tuple[dict, list[GenMetadataEntry]]:
        """For use by the AI Horde worker we require various counterintuitive hacks to the payload data.

        We encapsulate all of this implicit witchcraft in one function, here.
        """
        faults: list[GenMetadataEntry] = []

        if SharedModelManager.manager.compvis is None:
            raise RuntimeError("Cannot use AI Horde compatibility hacks without compvis loaded!")

        payload = deepcopy(payload)

        model = payload.get("model")

        if model is None:
            raise RuntimeError("No model specified in payload")

        # This is translated to "horde_model_name" later for compvis models and used as is for post processors
        payload["model_name"] = model

        found_model_in_ref = False
        found_model_on_disk = False
        model_files: list[dict] = [{}]

        if model in SharedModelManager.manager.compvis.model_reference:
            found_model_in_ref = True

        if SharedModelManager.manager.compvis.is_model_available(model):
            model_files = SharedModelManager.manager.compvis.get_model_filenames(model)
            found_model_on_disk = True

            if SharedModelManager.manager.compvis.model_reference[model].get("inpainting") is True:
                if payload.get("source_processing") not in ["inpainting", "outpainting"]:
                    logger.warning(
                        "Inpainting model detected, but source processing not set to inpainting or outpainting.",
                    )

                    payload["source_processing"] = "inpainting"

                source_image = payload.get("source_image")
                source_mask = payload.get("source_mask")

                if source_image is None or not isinstance(source_image, Image.Image):
                    logger.warning(
                        "Inpainting model detected, but source image is not a valid image. Using a noise image.",
                    )
                    faults.append(
                        GenMetadataEntry(
                            type=METADATA_TYPE.source_image,
                            value=METADATA_VALUE.parse_failed,
                        ),
                    )
                    payload["source_image"] = ImageUtils.create_noise_image(
                        payload["width"],
                        payload["height"],
                    )

                source_image = payload.get("source_image")

                if source_mask is None and (
                    source_image is None
                    or (isinstance(source_image, Image.Image) and not ImageUtils.has_alpha_channel(source_image))
                ):
                    logger.warning(
                        "Inpainting model detected, but no source mask provided. Using an all white mask.",
                    )
                    faults.append(
                        GenMetadataEntry(
                            type=METADATA_TYPE.source_mask,
                            value=METADATA_VALUE.parse_failed,
                        ),
                    )
                    payload["source_mask"] = ImageUtils.create_white_image(
                        source_image.width if source_image else payload["width"],
                        source_image.height if source_image else payload["height"],
                    )

        else:
            # The node may be a post processor, so we check the other model managers
            post_processor_model_managers = SharedModelManager.manager.get_model_manager_instances(
                [MODEL_CATEGORY_NAMES.codeformer, MODEL_CATEGORY_NAMES.esrgan, MODEL_CATEGORY_NAMES.gfpgan],
            )

            for post_processor_model_manager in post_processor_model_managers:
                if model in post_processor_model_manager.model_reference:
                    found_model_in_ref = True
                if post_processor_model_manager.is_model_available(model):
                    model_files = post_processor_model_manager.get_model_filenames(model)
                    found_model_on_disk = True
                    break

        if not found_model_in_ref:
            raise RuntimeError(f"Model {model} not found in model reference!")

        if not found_model_on_disk:
            raise RuntimeError(f"Model {model} not found on disk!")

        if len(model_files) == 0 or (not isinstance(model_files[0], dict)) or "file_path" not in model_files[0]:
            raise RuntimeError(f"Model {model} has no files in its reference entry!")

        payload["model"] = model_files[0]["file_path"]
        for file_entry in model_files:
            if "file_type" in file_entry:
                payload[file_entry["file_type"]] = file_entry["file_path"]

        # Rather than specify a scheduler, only karras or not karras is specified
        if payload.get("karras", False):
            payload["scheduler"] = "karras"
        else:
            payload["scheduler"] = "normal"

        prompt = payload.get("prompt")

        # Negative and positive prompts are merged together
        if prompt is not None:
            if "###" in prompt:
                split_prompts = prompt.split("###")
                payload["prompt"] = split_prompts[0]
                payload["negative_prompt"] = split_prompts[1]
        elif prompt == "":
            logger.warning("Empty prompt detected, this is likely to produce poor results")

        # Turn off hires fix if we're not generating a hires image, or if the params are just confused
        try:
            if "hires_fix" in payload:
                if SharedModelManager.manager.compvis.model_reference[model].get(
                    "baseline",
                ) == "stable diffusion 1" and (payload["width"] <= 512 or payload["height"] <= 512):
                    payload["hires_fix"] = False
                elif SharedModelManager.manager.compvis.model_reference[model].get(
                    "baseline",
                ) == "stable_diffusion_xl" and (payload["width"] <= 1024 or payload["height"] <= 1024):
                    payload["hires_fix"] = False
        except (TypeError, KeyError):
            payload["hires_fix"] = False

        # Turn off hires fix if we're inpainting as the dimensions are from the source image
        if "hires_fix" in payload and (
            payload.get("source_processing") == "inpainting" or payload.get("source_processing") == "outpainting"
        ):
            payload["hires_fix"] = False

        # Use denoising strength for both samplers if no second denoiser specified
        # but not for txt2img where denoising will always generally be 1.0
        if payload.get("hires_fix"):
            if payload.get("source_processing") and payload.get("source_processing") != "txt2img":
                if not payload.get("hires_fix_denoising_strength"):
                    payload["hires_fix_denoising_strength"] = payload.get("denoising_strength")

        if payload.get("workflow") == "qr_code":
            if payload.get("source_processing") and payload.get("source_processing") != "txt2img":
                if not payload.get("hires_fix_denoising_strength"):
                    payload["hires_fix_denoising_strength"] = payload.get("denoising_strength")

        # # Remap "denoising" to "controlnet strength", historical hack
        # if payload.get("control_type"):
        #     if payload.get("denoising_strength"):
        #         if not payload.get("control_strength"):
        #             payload["control_strength"] = payload["denoising_strength"]
        #             del payload["denoising_strength"]
        #         else:
        #             del payload["denoising_strength"]
        return payload, faults

    def _final_pipeline_adjustments(self, payload, pipeline_data) -> tuple[dict, list[GenMetadataEntry]]:
        payload = deepcopy(payload)
        faults: list[GenMetadataEntry] = []

        # Process dynamic prompts
        if new_prompt := DynamicPromptParser(payload["seed"]).parse(payload.get("prompt", "")):
            payload["prompt"] = new_prompt

        # Clip skip inversion, comfy uses -1, -2, etc
        if payload.get("clip_skip", 0) > 0:
            payload["clip_skip"] = -payload["clip_skip"]

        # Remap sampler name, horde to comfy
        payload["sampler_name"] = HordeLib.SAMPLERS_MAP.get(payload["sampler_name"])

        # Remap controlnet models, horde to comfy
        if payload.get("control_type"):
            payload["control_type"] = HordeLib.CONTROLNET_IMAGE_PREPROCESSOR_MAP.get(payload.get("control_type"))
        if payload.get("tis") and SharedModelManager.manager.ti:
            # Remove any requested TIs that we don't have
            for ti in payload.get("tis"):
                # Determine the actual ti filename
                if not SharedModelManager.manager.ti.is_local_model(str(ti["name"])):
                    try:
                        adhoc_ti = SharedModelManager.manager.ti.fetch_adhoc_ti(str(ti["name"]))
                    except Exception as e:
                        logger.info(f"Error fetching adhoc TI {ti['name']}: ({type(e).__name__}) {e}")
                        faults.append(
                            GenMetadataEntry(
                                type=METADATA_TYPE.ti,
                                value=METADATA_VALUE.download_failed,
                                ref=ti["name"],
                            ),
                        )
                        adhoc_ti = None
                    if not adhoc_ti:
                        logger.info(f"Adhoc TI requested '{ti['name']}' could not be found in CivitAI. Ignoring!")
                        faults.append(
                            GenMetadataEntry(
                                type=METADATA_TYPE.ti,
                                value=METADATA_VALUE.download_failed,
                                ref=ti["name"],
                            ),
                        )
                        continue
                ti_name = SharedModelManager.manager.ti.get_ti_name(str(ti["name"]))
                if ti_name:
                    logger.debug(f"Found valid TI {ti_name}")
                    if SharedModelManager.manager.compvis is None:
                        raise RuntimeError("Cannot use TIs without compvis loaded!")
                    model_details = SharedModelManager.manager.compvis.get_model_reference_info(payload["model"])
                    # If the ti and model do not match baseline, we ignore the TI
                    if not SharedModelManager.manager.ti.do_baselines_match(ti_name, model_details):
                        logger.info(f"Skipped TI {ti_name} because its baseline does not match the model's")
                        faults.append(
                            GenMetadataEntry(
                                type=METADATA_TYPE.ti,
                                value=METADATA_VALUE.baseline_mismatch,
                                ref=ti_name,
                            ),
                        )
                        continue
                    ti_inject = ti.get("inject_ti")
                    ti_strength = ti.get("strength", 1.0)
                    if type(ti_strength) not in [float, int]:
                        ti_strength = 1.0
                    ti_id = SharedModelManager.manager.ti.get_ti_id(str(ti["name"]))
                    if ti_inject == "prompt":
                        payload["prompt"] = f'(embedding:{ti_id}:{ti_strength}),{payload["prompt"]}'
                    elif ti_inject == "negprompt":
                        # create negative prompt if empty
                        if "negative_prompt" not in payload:
                            payload["negative_prompt"] = ""

                        had_leading_comma = payload["negative_prompt"].startswith(",")

                        payload["negative_prompt"] = f'{payload["negative_prompt"]},(embedding:{ti_id}:{ti_strength})'
                        if not had_leading_comma:
                            payload["negative_prompt"] = payload["negative_prompt"].strip(",")
                    SharedModelManager.manager.ti.touch_ti(ti_name)
        # Setup controlnet if required

        # For LORAs we completely build the LORA section of the pipeline dynamically, as we have
        # to handle n LORA models which form chained nodes in the pipeline.
        # Note that we build this between several nodes, the model_loader, clip_skip and the sampler,
        # plus the upscale sampler (used in hires fix) if there is one
        if payload.get("loras") and SharedModelManager.manager.lora:
            # Remove any requested LORAs that we don't have
            valid_loras = []
            for lora in payload.get("loras"):
                # Determine the actual lora filename
                is_version: bool = lora.get("is_version", False)
                verstext = ""
                if is_version:
                    verstext = " version"
                if not SharedModelManager.manager.lora.is_lora_available(str(lora["name"]), is_version=is_version):
                    logger.debug(f"Adhoc lora requested '{lora['name']}' not yet downloaded. Downloading...")
                    try:
                        adhoc_lora = SharedModelManager.manager.lora.fetch_adhoc_lora(
                            str(lora["name"]),
                            is_version=is_version,
                        )
                    except Exception as e:
                        logger.info(f"Error fetching adhoc lora {lora['name']}: ({type(e).__name__}) {e}")
                        faults.append(
                            GenMetadataEntry(
                                type=METADATA_TYPE.lora,
                                value=METADATA_VALUE.download_failed,
                                ref=lora["name"],
                            ),
                        )
                        adhoc_lora = None
                    if not adhoc_lora:
                        logger.info(
                            f"Adhoc lora requested{verstext} '{lora['name']} "
                            "could not be found in CivitAI. Ignoring!",
                        )
                        faults.append(
                            GenMetadataEntry(
                                type=METADATA_TYPE.lora,
                                value=METADATA_VALUE.download_failed,
                                ref=lora["name"],
                            ),
                        )
                        continue
                # We store the actual lora name to search for the trigger
                # If a version is requested, the lora name we need is the exact version
                if is_version:
                    lora_name = str(lora["name"])
                else:
                    lora_name = SharedModelManager.manager.lora.get_lora_name(str(lora["name"]))
                if lora_name is None:
                    faults.append(
                        GenMetadataEntry(
                            type=METADATA_TYPE.lora,
                            value=METADATA_VALUE.download_failed,
                            ref=lora_name,
                        ),
                    )
                    continue
                logger.debug(f"Found valid lora{verstext} {lora_name}")
                if SharedModelManager.manager.compvis is None:
                    raise RuntimeError("Cannot use LORAs without a compvis loaded!")
                model_details = SharedModelManager.manager.compvis.get_model_reference_info(payload["model"])
                # If the lora and model do not match baseline, we ignore the lora
                if not SharedModelManager.manager.lora.do_baselines_match(
                    lora_name,
                    model_details,
                    is_version=is_version,
                ):
                    logger.info(
                        f"Skipped lora{verstext} {lora_name} because its baseline does not match the model's",
                    )
                    faults.append(
                        GenMetadataEntry(
                            type=METADATA_TYPE.lora,
                            value=METADATA_VALUE.baseline_mismatch,
                            ref=lora_name,
                        ),
                    )
                    continue
                trigger_inject = lora.get("inject_trigger")
                trigger = None
                if trigger_inject == "any":
                    triggers = SharedModelManager.manager.lora.get_lora_triggers(lora_name, is_version=is_version)
                    if triggers:
                        trigger = random.choice(triggers)
                elif trigger_inject == "all":
                    triggers = SharedModelManager.manager.lora.get_lora_triggers(lora_name, is_version=is_version)
                    if triggers:
                        trigger = ", ".join(triggers)
                elif trigger_inject is not None:
                    trigger = SharedModelManager.manager.lora.find_lora_trigger(
                        lora_name,
                        trigger_inject,
                        is_version,
                    )
                if trigger:
                    # We inject at the start, to avoid throwing it in a negative prompt
                    payload["prompt"] = f'{trigger}, {payload["prompt"]}'
                # the fixed up and validated filename (Comfy expect the "name" key to be the filename)
                lora["name"] = SharedModelManager.manager.lora.get_lora_filename(lora_name, is_version=is_version)
                SharedModelManager.manager.lora._touch_lora(lora_name, is_version=is_version)
                valid_loras.append(lora)
            payload["loras"] = valid_loras
            for lora_index, lora in enumerate(payload.get("loras")):
                # Inject a lora node (first lora)
                if lora_index == 0:
                    pipeline_data[f"lora_{lora_index}"] = {
                        "inputs": {
                            "model": ["model_loader", 0],
                            "clip": ["model_loader", 1],
                            "lora_name": lora["name"],
                            "strength_model": lora["model"],
                            "strength_clip": lora["clip"],
                            # "model_manager": SharedModelManager,
                        },
                        "class_type": "HordeLoraLoader",
                    }
                else:
                    # Subsequent chained loras
                    pipeline_data[f"lora_{lora_index}"] = {
                        "inputs": {
                            "model": [f"lora_{lora_index-1}", 0],
                            "clip": [f"lora_{lora_index-1}", 1],
                            "lora_name": lora["name"],
                            "strength_model": lora["model"],
                            "strength_clip": lora["clip"],
                            # "model_manager": SharedModelManager,
                        },
                        "class_type": "HordeLoraLoader",
                    }

            for lora_index in range(len(payload.get("loras"))):
                # The first LORA always connects to the model loader
                if lora_index == 0:
                    self.generator.reconnect_input(pipeline_data, "lora_0.model", "model_loader")
                    self.generator.reconnect_input(pipeline_data, "lora_0.clip", "model_loader")
                else:
                    # Other loras connect to the previous lora
                    self.generator.reconnect_input(
                        pipeline_data,
                        f"lora_{lora_index}.model",
                        f"lora_{lora_index-1}.model",
                    )
                    self.generator.reconnect_input(
                        pipeline_data,
                        f"lora_{lora_index}.clip",
                        f"lora_{lora_index-1}.clip",
                    )

                # The last LORA always connects to the sampler and clip text encoders (via the clip_skip)
                if lora_index == len(payload.get("loras")) - 1:
                    self.generator.reconnect_input(pipeline_data, "sampler.model", f"lora_{lora_index}")
                    self.generator.reconnect_input(pipeline_data, "upscale_sampler.model", f"lora_{lora_index}")
                    self.generator.reconnect_input(pipeline_data, "clip_skip.clip", f"lora_{lora_index}")

        # Translate the payload parameters into pipeline parameters
        pipeline_params = {}
        for newkey, key in HordeLib.PAYLOAD_TO_PIPELINE_PARAMETER_MAPPING.items():
            multiplier = None
            # We allow a multiplier in the param, so that I can adjust easily the
            # values for steps on things like stable cascade
            if isinstance(key, FunctionType):
                pipeline_params[newkey] = key(payload)
            elif "*" in key:
                key, multiplier = key.split("*", 1)
            elif key in payload:
                if multiplier:
                    pipeline_params[newkey] = round(payload.get(key) * float(multiplier))
                else:
                    pipeline_params[newkey] = payload.get(key)
            else:
                logger.error(f"Parameter {key} not found")
        # We inject these parameters to ensure the HordeCheckpointLoader knows what file to load, if necessary
        # We don't want to hardcode this into the pipeline.json as we export this directly from ComfyUI
        # and don't want to have to rememebr to re-add those keys
        if "model_loader_stage_c.ckpt_name" in pipeline_params:
            pipeline_params["model_loader_stage_c.file_type"] = "stable_cascade_stage_c"
        if "model_loader_stage_b.ckpt_name" in pipeline_params:
            pipeline_params["model_loader_stage_b.file_type"] = "stable_cascade_stage_b"
        pipeline_params["model_loader.file_type"] = None  # To allow normal SD pipelines to keep working

        # Inject our model manager
        # pipeline_params["model_loader.model_manager"] = SharedModelManager
        pipeline_params["model_loader.will_load_loras"] = bool(payload.get("loras"))
        pipeline_params["model_loader_stage_c.will_load_loras"] = False  # FIXME: Once we support loras
        # Does this have to be required var in the modelloader?
        pipeline_params["model_loader_stage_c.seamless_tiling_enabled"] = False
        pipeline_params["model_loader_stage_b.will_load_loras"] = False  # FIXME: Once we support loras
        # Does this have to be required var in the modelloader?
        pipeline_params["model_loader_stage_b.seamless_tiling_enabled"] = False

        # For hires fix, change the image sizes as we create an intermediate image first
        if payload.get("hires_fix", False):
            model_details = (
                SharedModelManager.manager.compvis.get_model_reference_info(payload["model_name"])
                if SharedModelManager.manager.compvis
                else None
            )

            original_width = pipeline_params.get("empty_latent_image.width")
            original_height = pipeline_params.get("empty_latent_image.height")

            if original_width is None or original_height is None:
                if model_details and model_details.get("baseline") == "stable diffusion 1":
                    logger.error("empty_latent_image.width or empty_latent_image.height not found. Using 512x512.")
                    original_width, original_height = (512, 512)
                else:
                    logger.error("empty_latent_image.width or empty_latent_image.height not found. Using 1024x1024.")
                    original_width, original_height = (1024, 1024)

            new_width, new_height = (None, None)

            if model_details and model_details.get("baseline") == "stable_cascade":
                new_width, new_height = ImageUtils.get_first_pass_image_resolution_max(
                    original_width,
                    original_height,
                )
            else:
                new_width, new_height = ImageUtils.get_first_pass_image_resolution_min(
                    original_width,
                    original_height,
                )

            # This is the *target* resolution
            pipeline_params["latent_upscale.width"] = original_width
            pipeline_params["latent_upscale.height"] = original_height

            if new_width and new_height:
                # This is the *first pass* resolution
                pipeline_params["empty_latent_image.width"] = new_width
                pipeline_params["empty_latent_image.height"] = new_height
            else:
                logger.error("Could not determine new image size for hires fix. Using 1024x1024.")
                pipeline_params["empty_latent_image.width"] = 1024
                pipeline_params["empty_latent_image.height"] = 1024

        if payload.get("control_type"):
            # Inject control net model manager
            # pipeline_params["controlnet_model_loader.model_manager"] = SharedModelManager
            model_name = self.CONTROLNET_MODEL_MAP.get(payload.get("control_type"))
            if not model_name:
                logger.error(f"Controlnet model for {payload.get('control_type')} not found")
            pipeline_params["controlnet_model_loader.control_net_name"] = model_name

            # Dynamically reconnect nodes in the pipeline to connect the correct pre-processor node
            if payload.get("return_control_map"):
                # Connect annotator to output image directly if we need to return the control map
                self.generator.reconnect_input(
                    pipeline_data,
                    "output_image.images",
                    payload["control_type"],
                )
            elif payload.get("image_is_control"):
                # Connect source image directly to controlnet apply node
                self.generator.reconnect_input(
                    pipeline_data,
                    "controlnet_apply.image",
                    "image_loader.image",
                )
            else:
                # Connect annotator to controlnet apply node
                self.generator.reconnect_input(
                    pipeline_data,
                    "controlnet_apply.image",
                    payload["control_type"],
                )

        # If we have a source image, use that rather than latent noise (i.e. img2img)
        # We do this by reconnecting the nodes in the pipeline to make the input to the vae encoder
        # the source image instead of the latent noise generator
        if pipeline_params.get("image_loader.image"):
            self.generator.reconnect_input(pipeline_data, "sampler.latent_image", "vae_encode")
        if pipeline_params.get("sc_image_loader.image"):
            self.generator.reconnect_input(
                pipeline_data,
                "sampler_stage_c.latent_image",
                "stablecascade_stagec_vaeencode",
            )
            self.generator.reconnect_input(
                pipeline_data,
                "sampler_stage_b.latent_image",
                "stablecascade_stagec_vaeencode",
            )

        # If we have a remix request, we check for extra images to add to the pipeline
        if payload.get("source_processing") == "remix":
            logger.debug([payload.get("source_image"), payload.get("extra_source_images")])
            for image_index in range(len(payload.get("extra_source_images", []))):
                # The first image is always taken from the source_image param
                # That will sit on the 0 spot.
                # Therefore we want the extra images to start iterating from 1
                extra_image = payload["extra_source_images"][image_index]["image"]
                extra_image_strength = payload["extra_source_images"][image_index].get("strength", 1)
                node_index = image_index + 1
                pipeline_data[f"sc_image_loader_{node_index}"] = {
                    "inputs": {"image": extra_image, "upload": "image"},
                    "class_type": "HordeImageLoader",
                }
                pipeline_data[f"clip_vision_encode_{node_index}"] = {
                    "inputs": {
                        "clip_vision": ["model_loader_stage_c", 3],
                        "image": [f"sc_image_loader_{node_index}", 0],
                    },
                    "class_type": "CLIPVisionEncode",
                }
                pipeline_data[f"unclip_conditioning_{node_index}"] = {
                    "inputs": {
                        "strength": extra_image_strength,
                        "noise_augmentation": 0,
                        # Each conditioning ingests the conditioning before it like a chain
                        "conditioning": [f"unclip_conditioning_{node_index-1}", 0],
                        "clip_vision_output": [f"clip_vision_encode_{node_index}", 0],
                    },
                    "class_type": "unCLIPConditioning",
                }

                # The last extra image always connects to the stage_c sampler positive prompt
                if image_index == len(payload.get("extra_source_images")) - 1:
                    self.generator.reconnect_input(
                        pipeline_data,
                        "sampler_stage_c.positive",
                        f"unclip_conditioning_{node_index}",
                    )

        # If we have a qr code request, we check for extra texts such as the generation url
        if payload.get("workflow") == "qr_code":
            original_width = pipeline_params.get("empty_latent_image.width", 512)
            original_height = pipeline_params.get("empty_latent_image.height", 512)
            pipeline_params["qr_code_split.max_image_size"] = max(original_width, original_height)
            pipeline_params["qr_code_split.text"] = "https://haidra.net"
            for text in payload.get("extra_texts"):
                if text["reference"] in ["qr_code", "qr_text"]:
                    pipeline_params["qr_code_split.text"] = text["text"]
                if text["reference"] == "protocol" and text["text"].lower() in ["https", "http"]:
                    pipeline_params["qr_code_split.protocol"] = text["text"].capitalize()
                if text["reference"] == "module_drawer" and text["text"].lower() in [
                    "square",
                    "gapped square",
                    "circle",
                    "rounded",
                    "vertical bars",
                    "horizontal bars",
                ]:
                    pipeline_params["qr_code_split.module_drawer"] = text["text"].capitalize()
                if text["reference"] == "function_layer_prompt":
                    pipeline_params["function_layer_prompt.text"] = text["text"]
                if text["reference"] == "x_offset" and text["text"].lstrip("-").isdigit():
                    x_offset = int(text["text"])
                    if x_offset < 0:
                        x_offset = 10
                    pipeline_params["qr_flattened_composite.x"] = x_offset
                if text["reference"] == "y_offset" and text["text"].lstrip("-").isdigit():
                    y_offset = int(text["text"])
                    if y_offset < 0:
                        y_offset = 10
                    pipeline_params["qr_flattened_composite.y"] = y_offset
                if text["reference"] == "qr_border" and text["text"].lstrip("-").isdigit():
                    border = int(text["text"])
                    if border < 0:
                        border = 10
                    pipeline_params["qr_code_split.border"] = border
            if not pipeline_params.get("qr_code_split.protocol"):
                pipeline_params["qr_code_split.protocol"] = "None"
            if not pipeline_params.get("function_layer_prompt.text"):
                pipeline_params["function_layer_prompt.text"] = payload["prompt"]
            try:
                test_qr = QRByModuleSizeSplitFunctionPatterns()
                _, _, _, _, _, qr_size = test_qr.generate_qr(
                    protocol=pipeline_params.get("qr_code_split.protocol"),
                    text=pipeline_params["qr_code_split.text"],
                    module_size=16,
                    max_image_size=pipeline_params["qr_code_split.max_image_size"],
                    fill_hexcolor="#FFFFFF",
                    back_hexcolor="#000000",
                    error_correction="High",
                    border=1,
                    module_drawer="Square",
                )
            except RuntimeError as err:
                logger.error(err)
                pipeline_params["qr_code_split.text"] = "This QR Code is too large for this image"
                test_qr = QRByModuleSizeSplitFunctionPatterns()
                qr_size = 624

            if not pipeline_params.get("qr_flattened_composite.x"):
                x_offset = int((original_width / 2) - qr_size / 2)
                # I don't know why but through trial and error I've discovered that the QR codes
                # are more legible when they're placed in an offset which is a multiple of 64
                x_offset = x_offset - (x_offset % 64) if x_offset % 64 != 0 else x_offset
                pipeline_params["qr_flattened_composite.x"] = x_offset
            if pipeline_params.get("qr_flattened_composite.x", 0) > int((original_width) - qr_size):
                pipeline_params["qr_flattened_composite.x"] = int((original_width) - qr_size) - 10
            if not pipeline_params.get("qr_flattened_composite.y"):
                y_offset = int((original_height / 2) - qr_size / 2)
                y_offset = y_offset - (y_offset % 64) if y_offset % 64 != 0 else y_offset
                pipeline_params["qr_flattened_composite.y"] = y_offset
            if pipeline_params.get("qr_flattened_composite.y", 0) > int((original_height) - qr_size):
                pipeline_params["qr_flattened_composite.y"] = int((original_height) - qr_size) - 10
            pipeline_params["module_layer_composite.x"] = pipeline_params["qr_flattened_composite.x"]
            pipeline_params["module_layer_composite.y"] = pipeline_params["qr_flattened_composite.y"]
            pipeline_params["function_layer_composite.x"] = pipeline_params["qr_flattened_composite.x"]
            pipeline_params["function_layer_composite.y"] = pipeline_params["qr_flattened_composite.y"]
            pipeline_params["mask_composite.x"] = pipeline_params["qr_flattened_composite.x"]
            pipeline_params["mask_composite.y"] = pipeline_params["qr_flattened_composite.y"]
            if SharedModelManager.manager.compvis:
                model_details = SharedModelManager.manager.compvis.get_model_reference_info(payload["model_name"])
                if model_details and model_details.get("baseline") == "stable diffusion 1":
                    pipeline_params["controlnet_qr_model_loader.control_net_name"] = (
                        "control_v1p_sd15_qrcode_monster_v2.safetensors"
                    )
        if payload.get("transparent") is True:
            # A transparent gen is basically a fancy lora
            pipeline_params["model_loader.will_load_loras"] = True
            if SharedModelManager.manager.compvis:
                model_details = SharedModelManager.manager.compvis.get_model_reference_info(payload["model_name"])
                # SD2, Cascade and SD3 not supported
                if model_details and model_details.get("baseline") in ["stable diffusion 1", "stable_diffusion_xl"]:
                    self.generator.reconnect_input(pipeline_data, "sampler.model", "layer_diffuse_apply")
                    self.generator.reconnect_input(pipeline_data, "layer_diffuse_apply.model", "model_loader")
                    self.generator.reconnect_input(pipeline_data, "output_image.images", "layer_diffuse_decode_rgba")
                    self.generator.reconnect_input(pipeline_data, "layer_diffuse_decode_rgba.images", "vae_decode")
                    if payload.get("hires_fix") is True:
                        self.generator.reconnect_input(pipeline_data, "upscale_sampler.model", "layer_diffuse_apply")
                    if model_details.get("baseline") == "stable diffusion 1":
                        pipeline_params["layer_diffuse_apply.config"] = "SD15, Attention Injection, attn_sharing"
                        pipeline_params["layer_diffuse_decode_rgba.sd_version"] = "SD15"
                    else:
                        pipeline_params["layer_diffuse_apply.config"] = "SDXL, Conv Injection"
                        pipeline_params["layer_diffuse_decode_rgba.sd_version"] = "SDXL"

        return pipeline_params, faults

    def _get_appropriate_pipeline(self, params):
        # Determine the correct pipeline based on the parameters we have
        #
        # The pipelines are:
        #
        #     stable_diffusion
        #       stable_diffusion_hires_fix
        #     stable_diffusion_img2img_mask
        #     stable_diffusion_paint
        #     controlnet
        #       controlnet_hires_fix
        #       controlnet_annotator
        #     image_facefix
        #     image_upscale
        #     stable_cascade
        #       stable_cascade_remix
        #       stable_cascade_2pass
        #     qr_code

        # controlnet, controlnet_hires_fix controlnet_annotator
        if params.get("workflow") == "qr_code":
            return "qr_code"
        if params.get("model_name"):
            model_details = SharedModelManager.manager.compvis.get_model_reference_info(params["model_name"])
            if model_details.get("baseline") == "stable_cascade":
                if params.get("source_processing") == "remix":
                    return "stable_cascade_remix"
                if params.get("hires_fix", False):
                    return "stable_cascade_2pass"
                return "stable_cascade"
        if params.get("control_type"):
            if params.get("return_control_map", False):
                return "controlnet_annotator"

            if params.get("hires_fix"):
                return "controlnet_hires_fix"

            return "controlnet"

        # stable_diffusion_paint, stable_diffusion_img2img_mask
        if params.get("source_processing") == "img2img":
            has_mask = params.get("source_mask") or (
                params.get("source_image") and len(params.get("source_image", "").getbands()) == 4
            )
            if has_mask:
                return "stable_diffusion_img2img_mask"
        elif params.get("source_processing") == "inpainting":
            return "stable_diffusion_paint"
        elif params.get("source_processing") == "outpainting":
            return "stable_diffusion_paint"

        # stable_diffusion and stable_diffusion_hires_fix
        if params.get("hires_fix", False):
            return "stable_diffusion_hires_fix"

        return "stable_diffusion"  # also includes img2img mode

    def _process_results(
        self,
        images: list[dict[str, typing.Any]],
    ) -> list[tuple[Image.Image, io.BytesIO]]:
        if images is None:
            logger.error("No images returned from comfy pipeline!")
            raise RuntimeError("No images returned from comfy pipeline!")

        results: list[tuple[Image.Image, io.BytesIO]] = []

        # Return image(s) or raw PNG byte stream
        for image in images:
            results.append(
                (
                    Image.open(image["imagedata"]),
                    image["imagedata"],
                ),
            )
        return results

    def _get_validated_payload_and_pipeline_data(self, payload: dict) -> tuple[dict, dict, list[GenMetadataEntry]]:
        # AIHorde hacks to payload
        payload, compatibility_faults = self._apply_aihorde_compatibility_hacks(payload)
        # Check payload types/values and normalise it's format
        payload = self._validate_data_structure(payload)
        # Resize the source image and mask to actual final width/height requested
        ImageUtils.resize_sources_to_request(payload)
        # Determine the correct pipeline to use
        pipeline = self._get_appropriate_pipeline(payload)
        # Final adjustments to the pipeline
        pipeline_data = self.generator.get_pipeline_data(pipeline)
        payload, finale_adjustment_faults = self._final_pipeline_adjustments(payload, pipeline_data)
        return payload, pipeline_data, compatibility_faults + finale_adjustment_faults

    def _inference(
        self,
        payload: dict,
        *,
        single_image_expected: bool = True,
        comfyui_progress_callback: Callable[[ComfyUIProgress, str], None] | None = None,
    ) -> list[ResultingImageReturn] | ResultingImageReturn:
        payload, pipeline_data, faults = self._get_validated_payload_and_pipeline_data(payload)

        # Run the pipeline

        # Add prefix to loras to avoid name collisions with other models
        loras_being_used = [f"lora-{x['name']}" for x in payload.get("loras", []) if x]
        tis_being_used = [f"ti-{x['name']}" for x in payload.get("tis", []) if x]

        # main model
        main_model = f"{payload.get('model_loader.horde_model_name')}"
        # controlnet model
        controlnet_name = payload.get("controlnet_model_loader.control_net_name")

        resolution = f"{payload.get('empty_latent_image.width')}x{payload.get('empty_latent_image.height')}"
        steps = payload.get("sampler.steps")

        logger.info(f"Generating image {resolution} in size for {steps} steps with model {main_model}.")
        if "latent_upscale.width" in payload:
            hires_fix_final_resolution = (
                f"{payload.get('latent_upscale.width')}x{payload.get('latent_upscale.height')}"
            )
            logger.info(f"Using hiresfix, final resolution {hires_fix_final_resolution}.")
        if controlnet_name:
            logger.info(f"Using controlnet {controlnet_name}")
        if loras_being_used:
            logger.info(f"Using {len(loras_being_used)} LORAs")
        if tis_being_used:
            logger.info(f"Using {len(tis_being_used)} TIs")

        # Call the inference pipeline
        # logger.debug(payload)
        images = self.generator.run_image_pipeline(pipeline_data, payload, comfyui_progress_callback)

        results = self._process_results(images)
        ret_results = [
            ResultingImageReturn(
                image=image,
                rawpng=rawpng,
                faults=faults,
            )
            for image, rawpng in results
        ]

        if single_image_expected:
            if len(results) != 1:
                raise RuntimeError("Expected a single image but got multiple")
            return ret_results[0]

        return ret_results

    def basic_inference(
        self,
        payload: dict | ImageGenerateJobPopResponse,
        *,
        progress_callback: Callable[[ProgressReport], None] | None = None,
    ) -> list[ResultingImageReturn]:
        post_processing_requested: list[str] | None = None
        if isinstance(payload, dict):
            post_processing_requested = payload.get("post_processing")

        faults = []
        if isinstance(payload, ImageGenerateJobPopResponse):  # TODO move this to _inference()
            for post_processor_requested in payload.payload.post_processing:
                if post_processing_requested is None:
                    post_processing_requested = []
                post_processing_requested.append(post_processor_requested)
                logger.debug(f"Post-processing requested: {post_processor_requested}")

            sub_payload = payload.payload.model_dump()

            def handle_images(
                payload: ImageGenerateJobPopResponse,
                image_type: str,
                get_downloaded_image_func: Callable,
            ):
                image = getattr(payload, image_type)

                if image is not None and "http" in image:
                    image = get_downloaded_image_func()

                    if image is None:
                        logger.error(
                            f"{image_type.capitalize()} is a URL but wasn't downloaded, "
                            "this is not supported in this context. Run the `async_download_*` methods first.",
                        )

                        return None

                return image

            source_image = handle_images(
                payload,
                "source_image",
                payload.get_downloaded_source_image,
            )
            if source_image is None:
                logger.info("No source image found in payload.")

            mask_image = handle_images(
                payload,
                "source_mask",
                payload.get_downloaded_source_mask,
            )
            if mask_image is None:
                logger.info("No mask image found in payload.")

            extra_source_images = payload.extra_source_images

            if extra_source_images is not None:
                extra_source_images = payload.get_downloaded_extra_source_images()
                if extra_source_images is not None:
                    logger.info(f"Using {len(extra_source_images)} downloaded extra source images.")
                else:
                    logger.info("No extra source images found in payload.")

            esi_to_remove = []
            if extra_source_images is not None:
                for esi in extra_source_images:
                    if "http" in esi.image:
                        logger.warning("Extra source image is a URL, this is not supported in this context.")
                        esi_to_remove.append(esi)

                extra_source_images = [esi for esi in extra_source_images if esi not in esi_to_remove]
            # If its a base64 encoded image, decode it
            if isinstance(source_image, str):
                try:
                    source_image_bytes = base64.b64decode(source_image)
                    source_image_pil = Image.open(io.BytesIO(source_image_bytes))
                    sub_payload["source_image"] = source_image_pil
                except Exception as err:
                    faults.append(
                        GenMetadataEntry(
                            type=METADATA_TYPE.source_image,
                            value=METADATA_VALUE.parse_failed,
                        ),
                    )
                    logger.warning(f"Failed to parse source image ({err}). Falling back to text2img.")

            if isinstance(mask_image, str):
                try:
                    mask_image_bytes = base64.b64decode(mask_image)
                    mask_image_pil = Image.open(io.BytesIO(mask_image_bytes))
                    sub_payload["source_mask"] = mask_image_pil
                except Exception as err:
                    faults.append(
                        GenMetadataEntry(
                            type=METADATA_TYPE.source_mask,
                            value=METADATA_VALUE.parse_failed,
                        ),
                    )
                    logger.warning(f"Failed to parse source mask ({err}). Ignoring it.")

            if isinstance(extra_source_images, list):
                extra_source_images_sub = []
                for esi_index, esi in enumerate(extra_source_images):
                    try:
                        esi_bytes = base64.b64decode(esi.image)
                        esi_pil = Image.open(io.BytesIO(esi_bytes))
                        extra_source_images_sub.append(
                            {
                                "image": esi_pil,
                                "strength": esi.strength,
                            },
                        )
                    except Exception as err:
                        faults.append(
                            GenMetadataEntry(
                                type=METADATA_TYPE.extra_source_images,
                                value=METADATA_VALUE.parse_failed,
                                ref=str(esi_index),
                            ),
                        )
                        logger.warning(f"Failed to parse extra source image {esi_index} ({err}). Ignoring it.")
                sub_payload["extra_source_images"] = extra_source_images_sub

            sub_payload["source_processing"] = payload.source_processing
            sub_payload["model"] = payload.model
            payload = sub_payload

        if progress_callback is not None:
            try:
                progress_callback(
                    ProgressReport(
                        hordelib_progress_state=ProgressState.started,
                        hordelib_message="Initiating inference...",
                        progress=0,
                    ),
                )
            except Exception as e:
                logger.error(f"Progress callback failed ({type(e)}): {e}")

        def _default_progress_callback(comfyui_progress: ComfyUIProgress, message: str) -> None:
            nonlocal progress_callback
            if progress_callback is not None:
                try:
                    progress_callback(
                        ProgressReport(
                            hordelib_progress_state=ProgressState.progress,
                            hordelib_message=message,
                            comfyui_progress=comfyui_progress,
                        ),
                    )
                except Exception as e:
                    logger.error(f"Progress callback failed ({type(e)}): {e}")

        result = self._inference(
            payload,
            single_image_expected=False,
            comfyui_progress_callback=_default_progress_callback,
        )

        if not isinstance(result, list):
            raise RuntimeError(f"Expected a list of PIL.Image.Image but got {type(result)}")

        return_list = [x for x in result if isinstance(x.image, Image.Image)]
        pptext = ""
        if post_processing_requested is not None:
            pptext = " Initiating post-processing..."
        logger.debug(f"Inference complete. Received {len(return_list)} images.{pptext}")

        post_processed: list[ResultingImageReturn] | None = None
        if post_processing_requested is not None:
            if progress_callback is not None:
                try:
                    progress_callback(
                        ProgressReport(
                            hordelib_progress_state=ProgressState.post_processing,
                            hordelib_message="Post Processing.",
                        ),
                    )
                except Exception as e:
                    logger.error(f"Progress callback failed ({type(e)}): {e}")

            post_processed = []
            for ret in return_list:
                single_image_faults = []
                final_image = ret.image
                final_rawpng = ret.rawpng

                if progress_callback is not None:
                    try:
                        progress_callback(
                            ProgressReport(
                                hordelib_progress_state=ProgressState.progress,
                                hordelib_message="Post Processing new image.",
                            ),
                        )
                    except Exception as e:
                        logger.error(f"Progress callback failed ({type(e)}): {e}")

                # Ensure facefixers always happen first
                post_processing_requested = sorted(
                    post_processing_requested,
                    key=lambda x: 1 if x in KNOWN_FACEFIXERS.__members__ else 0,
                )

                for post_processing in post_processing_requested:
                    if (
                        post_processing in KNOWN_UPSCALERS.__members__
                        or post_processing in KNOWN_UPSCALERS._value2member_map_
                    ):
                        image_ret = self.image_upscale(
                            {
                                "model": post_processing,
                                "source_image": final_image,
                            },
                        )
                        single_image_faults += image_ret.faults
                        final_rawpng = image_ret.rawpng
                        final_image = image_ret.image

                    elif post_processing in KNOWN_FACEFIXERS.__members__:
                        image_ret = self.image_facefix(
                            {
                                "model": post_processing,
                                "source_image": final_image,
                                "facefixer_strength": payload.get("facefixer_strength", 1.0),
                            },
                        )
                        single_image_faults += image_ret.faults
                        final_rawpng = image_ret.rawpng
                        final_image = image_ret.image

                    elif post_processing == "strip_background":
                        if final_image is not None:
                            final_image = ImageUtils.strip_background(final_image)

                if final_image is None:
                    # TODO: Allow to return a partially PP image?
                    logger.error("Post processing failed and there is no output image!")
                else:
                    post_processed.append(
                        ResultingImageReturn(image=final_image, rawpng=final_rawpng, faults=single_image_faults),
                    )

        if progress_callback is not None:
            try:
                progress_callback(
                    ProgressReport(
                        hordelib_progress_state=ProgressState.finished,
                        hordelib_message="Inference complete.",
                        progress=100,
                    ),
                )
            except Exception as e:
                logger.error(f"Progress callback failed ({type(e)}): {e}")

        if post_processed is not None:
            logger.debug(f"Post-processing complete. Returning {len(post_processed)} images.")
            return post_processed

        if len(return_list) == len(result):
            return return_list

        raise RuntimeError("Expected a list of PIL.Image.Image but got a mix of types!")

    def basic_inference_single_image(self, payload: dict) -> ResultingImageReturn:
        result = self._inference(payload, single_image_expected=True)
        if isinstance(result, ResultingImageReturn):
            return result

        raise RuntimeError(f"Expected a PIL.Image.Image but got {type(result)}")

    def basic_inference_rawpng(self, payload: dict) -> list[io.BytesIO]:
        """Return the results directly from comfy as (a) raw PNG byte stream(s)."""
        result = self._inference(payload, single_image_expected=False)

        if isinstance(result, list):
            bytes_io_list = [x.rawpng for x in result if isinstance(x.rawpng, io.BytesIO)]
            if len(bytes_io_list) == len(result):
                return bytes_io_list

            raise RuntimeError("Expected a list of io.BytesIO but got a mix of types!")

        if isinstance(result.image, io.BytesIO):
            return [result.image]

        raise RuntimeError(f"Expected at least one io.BytesIO. Got {result}.")

    def image_upscale(self, payload) -> ResultingImageReturn:
        logger.debug("image_upscale called")

        from hordelib.comfy_horde import log_free_ram

        log_free_ram()

        # AIHorde hacks to payload
        payload, compatibility_faults = self._apply_aihorde_compatibility_hacks(payload)
        # Remember if we were passed width and height, we wouldn't normally be passed width and height
        # because the upscale models upscale to a fixed multiple of image size. However, if we *are*
        # passed a width and height we rescale the upscale output image to this size.
        width = payload.get("height")
        height = payload.get("width")
        # Check payload types/values and normalise it's format
        payload = self._validate_data_structure(payload)
        # Final adjustments to the pipeline
        pipeline_name = "image_upscale"
        pipeline_data = self.generator.get_pipeline_data(pipeline_name)
        payload, final_adjustment_faults = self._final_pipeline_adjustments(payload, pipeline_data)

        # Run the pipeline

        images = self.generator.run_image_pipeline(pipeline_data, payload)

        # Allow arbitrary resizing by shrinking the image back down
        if width or height:
            return ResultingImageReturn(
                ImageUtils.shrink_image(Image.open(images[0]["imagedata"]), width, height),
                rawpng=None,
                faults=final_adjustment_faults,
            )
        result = self._process_results(images)
        if len(result) != 1:
            raise RuntimeError("Expected a single image but got multiple")

        image, rawpng = result[0]
        if not isinstance(image, Image.Image):
            raise RuntimeError(f"Expected a PIL.Image.Image but got {type(image)}")

        log_free_ram()
        return ResultingImageReturn(image=image, rawpng=rawpng, faults=compatibility_faults + final_adjustment_faults)

    def image_facefix(self, payload) -> ResultingImageReturn:
        logger.debug("image_facefix called")

        from hordelib.comfy_horde import log_free_ram

        log_free_ram()

        # AIHorde hacks to payload
        payload, compatibility_faults = self._apply_aihorde_compatibility_hacks(payload)
        # Check payload types/values and normalise it's format
        payload = self._validate_data_structure(payload)
        # Final adjustments to the pipeline
        pipeline_name = "image_facefix"
        pipeline_data = self.generator.get_pipeline_data(pipeline_name)
        payload, final_adjustment_faults = self._final_pipeline_adjustments(payload, pipeline_data)

        # Run the pipeline

        images = self.generator.run_image_pipeline(pipeline_data, payload)

        results = self._process_results(images)
        if len(results) != 1:
            raise RuntimeError("Expected a single image but got multiple")

        image, rawpng = results[0]
        if not isinstance(image, Image.Image):
            raise RuntimeError(f"Expected a PIL.Image.Image but got {type(image)}")

        log_free_ram()

        return ResultingImageReturn(image=image, rawpng=rawpng, faults=compatibility_faults + final_adjustment_faults)
