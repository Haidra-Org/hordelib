# horde.py
# Main interface for the horde to this library.
from __future__ import annotations

import glob
import json
import os
import random
import sys
import time
from copy import deepcopy

from loguru import logger
from PIL import Image

from hordelib.comfy_horde import Comfy_Horde
from hordelib.shared_model_manager import SharedModelManager
from hordelib.utils.dynamicprompt import DynamicPromptParser
from hordelib.utils.image_utils import ImageUtils
from hordelib.utils.sanitizer import Sanitizer


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

    SOURCE_IMAGE_PROCESSING_OPTIONS = ["img2img", "inpainting", "outpainting"]

    SCHEDULERS = ["normal", "karras", "simple", "ddim_uniform"]

    # Describe a valid payload, it's types and bounds. All incoming payload data is validated against,
    # and normalised to, this schema.
    PAYLOAD_SCHEMA = {
        "sampler_name": {"datatype": str, "values": list(SAMPLERS_MAP.keys()), "default": "k_euler"},
        "cfg_scale": {"datatype": float, "min": 1, "max": 100, "default": 8.0},
        "denoising_strength": {"datatype": float, "min": 0.01, "max": 1.0, "default": 1.0},
        "control_strength": {"datatype": float, "min": 0.01, "max": 1.0, "default": 1.0},
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
        "ddim_steps": {"datatype": int, "min": 1, "max": 500, "default": 30},
        "n_iter": {"datatype": int, "min": 1, "max": 100, "default": 1},
        "model": {"datatype": str, "default": "stable_diffusion"},
        "source_mask": {"datatype": Image.Image, "default": None},
        "source_image": {"datatype": Image.Image, "default": None},
        "source_processing": {"datatype": str, "values": SOURCE_IMAGE_PROCESSING_OPTIONS, "default": None},
        "hires_fix_denoising_strength": {"datatype": float, "min": 0.0, "max": 1.0, "default": 0.65},
        "scheduler": {"datatype": str, "values": SCHEDULERS, "default": "normal"},
    }

    LORA_SCHEMA = {
        "name": {"datatype": str, "default": ""},
        "model": {"datatype": float, "min": 0.0, "max": 1.0, "default": 1.0},
        "clip": {"datatype": float, "min": 0.0, "max": 1.0, "default": 1.0},
        "inject_trigger": {"datatype": str},
    }

    # pipeline parameter <- hordelib payload parameter mapping
    PAYLOAD_TO_PIPELINE_PARAMETER_MAPPING = {
        "sampler.sampler_name": "sampler_name",
        "sampler.cfg": "cfg_scale",
        "sampler.denoise": "denoising_strength",
        "sampler.seed": "seed",
        "empty_latent_image.height": "height",
        "empty_latent_image.width": "width",
        "sampler.scheduler": "scheduler",
        "clip_skip.stop_at_clip_layer": "clip_skip",
        "prompt.text": "prompt",
        "negative_prompt.text": "negative_prompt",
        "sampler.steps": "ddim_steps",
        "empty_latent_image.batch_size": "n_iter",
        "model_loader.model_name": "model",
        "image_loader.image": "source_image",
        "loras": "loras",
        "upscale_sampler.denoise": "hires_fix_denoising_strength",
        "upscale_sampler.seed": "seed",
        "upscale_sampler.cfg": "cfg_scale",
        "upscale_sampler.steps": "ddim_steps",
        "upscale_sampler.sampler_name": "sampler_name",
        "controlnet_apply.strength": "control_strength",
        "controlnet_model_loader.control_net_name": "control_type",
    }

    # We are a singleton
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    # We initialise only ever once (in the lifetime of the singleton)
    def __init__(self):
        if not self._initialised:
            self.generator = Comfy_Horde()
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
        if min and value < min:
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
                else:
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

        return data

    def _apply_aihorde_compatibility_hacks(self, payload):
        """For use by the AI Horde worker we require various counterintuitive hacks to the payload data.

        We encapsulate all of this implicit witchcraft in one function, here.
        """
        payload = deepcopy(payload)

        # Rather than specify a scheduler, only karras or not karras is specified
        if payload.get("karras", False):
            payload["scheduler"] = "karras"
        else:
            payload["scheduler"] = "normal"

        # Negative and positive prompts are merged together
        if payload.get("prompt"):
            split_prompts = [x.strip() for x in payload.get("prompt").split("###")][:2]
            if len(split_prompts) == 2:
                payload["prompt"] = split_prompts[0]
                payload["negative_prompt"] = split_prompts[1]

        # Turn off hires fix if we're not generating a hires image, or if the params are just confused
        try:
            if "hires_fix" in payload and (payload["width"] <= 512 or payload["height"] <= 512):
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

        # Remap "denoising" to "controlnet strength", historical hack
        if payload.get("control_type"):
            if payload.get("denoising_strength"):
                if not payload.get("control_strength"):
                    payload["control_strength"] = payload["denoising_strength"]
                    del payload["denoising_strength"]
                else:
                    del payload["denoising_strength"]

        return payload

    def _final_pipeline_adjustments(self, payload, pipeline_data):

        payload = deepcopy(payload)

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
                if not SharedModelManager.manager.lora.is_local_model(str(lora["name"])):
                    adhoc_lora = SharedModelManager.manager.lora.fetch_adhoc_lora(str(lora["name"]))
                    if not adhoc_lora:
                        logger.info(f"Adhoc lora requested '{lora['name']}' could not be found in CivitAI. Ignoring!")
                        continue
                # We store the actual lora name to search for the trigger
                lora_name = SharedModelManager.manager.lora.get_lora_name(str(lora["name"]))
                if lora_name:
                    logger.debug(f"Found valid lora {lora_name}")
                    if SharedModelManager.manager.compvis is None:
                        raise RuntimeError("Cannot use LORAs without a compvis loaded!")
                    model_details = SharedModelManager.manager.compvis.get_model(payload["model"])
                    # If the lora and model do not match baseline, we ignore the lora
                    if not SharedModelManager.manager.lora.do_baselines_match(lora_name, model_details):
                        logger.info(f"Skipped lora {lora_name} because its baseline does not match the model's")
                        continue
                    trigger_inject = lora.get("inject_trigger")
                    trigger = None
                    if trigger_inject == "any":
                        triggers = SharedModelManager.manager.lora.get_lora_triggers(lora_name)
                        if triggers:
                            trigger = random.choice(triggers)
                    elif trigger_inject == "all":
                        triggers = SharedModelManager.manager.lora.get_lora_triggers(lora_name)
                        if triggers:
                            trigger = ", ".join(triggers)
                    elif trigger_inject is not None:
                        trigger = SharedModelManager.manager.lora.find_lora_trigger(lora_name, trigger_inject)
                    if trigger:
                        # We inject at the start, to avoid throwing it in a negative prompt
                        payload["prompt"] = f'{trigger}, {payload["prompt"]}'
                    # the fixed up and validated filename (Comfy expect the "name" key to be the filename)
                    lora["name"] = SharedModelManager.manager.lora.get_lora_filename(lora_name)
                    SharedModelManager.manager.lora.touch_lora(lora_name)
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
                            "model_manager": SharedModelManager,
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
                            "model_manager": SharedModelManager,
                        },
                        "class_type": "HordeLoraLoader",
                    }

            for lora_index, lora in enumerate(payload.get("loras")):

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
            if key in payload:
                pipeline_params[newkey] = payload.get(key)
            else:
                logger.error(f"Parameter {key} not found")

        # Inject our model manager
        pipeline_params["model_loader.model_manager"] = SharedModelManager

        # For hires fix, change the image sizes as we create an intermediate image first
        if payload.get("hires_fix", False):
            width = pipeline_params.get("empty_latent_image.width", 0)
            height = pipeline_params.get("empty_latent_image.height", 0)
            if width > 512 and height > 512:
                newwidth, newheight = ImageUtils.calculate_source_image_size(width, height)
                pipeline_params["latent_upscale.width"] = width
                pipeline_params["latent_upscale.height"] = height
                pipeline_params["empty_latent_image.width"] = newwidth
                pipeline_params["empty_latent_image.height"] = newheight

        if payload.get("control_type"):
            # Inject control net model manager
            pipeline_params["controlnet_model_loader.model_manager"] = SharedModelManager

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

        return pipeline_params

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

        # controlnet, controlnet_hires_fix controlnet_annotator
        if params.get("control_type"):
            if params.get("return_control_map", False):
                return "controlnet_annotator"
            else:
                if params.get("hires_fix"):
                    return "controlnet_hires_fix"
                else:
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
        else:
            return "stable_diffusion"  # also includes img2img mode

    def _process_results(self, images, rawpng):
        if images is None:
            return None  # XXX Log error and/or raise Exception here
        # Return image(s) or raw PNG bytestream
        if not rawpng:
            results = [Image.open(x["imagedata"]) for x in images]
        else:
            results = [x["imagedata"] for x in images]
        if len(results) == 1:
            return results[0]
        else:
            return results

    def lock_models(self, models):
        models = [str(x).strip() for x in models if x]
        # Try to acquire a model lock, if we can't, wait a while as some other thread
        # must have these resources locked
        while not self.generator.lock_models(models):
            time.sleep(0.1)
        logger.debug(f"Locked models {','.join(models)}")

    def unlock_models(self, models):
        models = [x.strip() for x in models if x]
        self.generator.unlock_models(models)
        logger.debug(f"Unlocked models {','.join(models)}")

    def basic_inference(self, payload, rawpng=False):
        # AIHorde hacks to payload
        payload = self._apply_aihorde_compatibility_hacks(payload)
        # Check payload types/values and normalise it's format
        payload = self._validate_data_structure(payload)
        # Resize the source image and mask to actual final width/height requested
        ImageUtils.resize_sources_to_request(payload)
        # Determine the correct pipeline to use
        pipeline = self._get_appropriate_pipeline(payload)
        # Final adjustments to the pipeline
        pipeline_data = self.generator.get_pipeline_data(pipeline)
        payload = self._final_pipeline_adjustments(payload, pipeline_data)
        models: list[str] = []
        # Run the pipeline
        try:
            # Add prefix to loras to avoid name collisions with other models
            models = [f"lora-{x['name']}" for x in payload.get("loras", []) if x]
            # main model
            models.append(payload.get("model_loader.model_name"))  # type: ignore # FIXME?
            # controlnet model
            models.append(payload.get("controlnet_model_loader.control_net_name"))  # type: ignore # FIXME?
            # Acquire a lock on all these models
            self.lock_models(models)
            # Call the inference pipeline
            # logger.info(payload)
            images = self.generator.run_image_pipeline(pipeline_data, payload)
        finally:
            self.unlock_models(models)
        return self._process_results(images, rawpng)

    def image_upscale(self, payload, rawpng=False) -> Image.Image | None:
        # AIHorde hacks to payload
        payload = self._apply_aihorde_compatibility_hacks(payload)
        # Remember if we were passed width and height, we wouldn't normally be passed width and height
        # because the upscale models upscale to a fixed multiple of image size. However, if we *are*
        # passed a width and height we rescale the upscale output image to this size.
        width = payload.get("width")
        height = payload.get("width")
        # Check payload types/values and normalise it's format
        payload = self._validate_data_structure(payload)
        # Final adjustments to the pipeline
        pipeline_name = "image_upscale"
        pipeline_data = self.generator.get_pipeline_data(pipeline_name)
        payload = self._final_pipeline_adjustments(payload, pipeline_data)
        # Run the pipeline
        try:
            self.lock_models([payload.get("model_loader.model_name")])
            images = self.generator.run_image_pipeline(pipeline_data, payload)
        finally:
            self.unlock_models([payload.get("model_loader.model_name")])
        if images is None:
            return None  # XXX Log error and/or raise Exception here
        # Allow arbitrary resizing by shrinking the image back down
        if width or height:
            return ImageUtils.shrink_image(Image.open(images[0]["imagedata"]), width, height)
        return self._process_results(images, rawpng)  # type: ignore # FIXME?

    def image_facefix(self, payload, rawpng=False) -> Image.Image | None:
        # AIHorde hacks to payload
        payload = self._apply_aihorde_compatibility_hacks(payload)
        # Check payload types/values and normalise it's format
        payload = self._validate_data_structure(payload)
        # Final adjustments to the pipeline
        pipeline_name = "image_facefix"
        pipeline_data = self.generator.get_pipeline_data(pipeline_name)
        payload = self._final_pipeline_adjustments(payload, pipeline_data)
        # Run the pipeline
        try:
            self.lock_models([payload.get("model_loader.model_name")])
            images = self.generator.run_image_pipeline(pipeline_data, payload)
        finally:
            self.unlock_models([payload.get("model_loader.model_name")])
        return self._process_results(images, rawpng)  # type: ignore # FIXME?
