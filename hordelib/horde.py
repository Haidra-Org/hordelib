# horde.py
# Main interface for the horde to this library.
import random
import sys

from loguru import logger
from PIL import Image, ImageOps, PngImagePlugin, UnidentifiedImageError

from hordelib.comfy_horde import Comfy_Horde
from hordelib.shared_model_manager import SharedModelManager
from hordelib.utils.dynamicprompt import DynamicPromptParser


class HordeLib:
    _instance = None
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

    # Horde to tex2img parameter mapping
    # XXX Items mapped to None are ignored for now
    BASIC_INFERENCE_PARAMS = {
        "sampler_name": "sampler.sampler_name",
        "cfg_scale": "sampler.cfg",
        "denoising_strength": "sampler.denoise",
        "seed": "sampler.seed",
        "height": "empty_latent_image.height",
        "width": "empty_latent_image.width",
        # "karras": Handled below
        "tiling": None,
        # "hires_fix": Handled below
        "clip_skip": "clip_skip.stop_at_clip_layer",
        # "control_type": Handled below
        "image_is_control": None,
        "return_control_map": "return_control_map",
        # "prompt": Handled below
        "ddim_steps": "sampler.steps",
        "n_iter": "empty_latent_image.batch_size",
        "model": "model_loader.model_name",
        "source_image": "image_loader.image",
        "source_mask": None,
        "source_processing": "source_processing",
        "control_strength": "controlnet_apply.strength",
        "hires_fix_denoising_strength": "upscale_sampler.denoise",
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
        "hough": "mlsd",
        # "<unused>": "MiDaS-DepthMapPreprocessor",
        # "<unused>": "MediaPipe-HandPosePreprocessor",
        # "<unused>": "MediaPipe-FaceMeshPreprocessor",
        # "<unused>": "BinaryPreprocessor",
        # "<unused>": "ColorPreprocessor",
        # "<unused>": "PiDiNetPreprocessor",
    }

    SOURCE_IMAGE_PROCESSING_OPTIONS = ["img2img", "inpainting", "outpainting"]

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

    def _check_payload(self, payload):
        # valid width
        if payload.get("width"):
            try:
                payload["width"] = int(payload.get("width"))
                if payload["width"] < 64:
                    payload["width"] = 64
            except ValueError:
                payload["width"] = 512
        # valid height
        if payload.get("height"):
            try:
                payload["height"] = int(payload.get("height"))
                if payload["height"] < 64:
                    payload["height"] = 64
            except ValueError:
                payload["height"] = 512
        # Validate denoising
        if payload.get("denoising_strength"):
            try:
                payload["denoising_strength"] = float(payload.get("denoising_strength"))
                if payload["denoising_strength"] < 0.0:
                    payload["denoising_strength"] = 0.0
                elif payload["denoising_strength"] > 1.0:
                    payload["denoising_strength"] = 1.0
            except ValueError:
                payload["denoising_strength"] = 1.0
        if payload.get("hires_fix_denoising_strength"):
            try:
                payload["hires_fix_denoising_strength"] = float(payload.get("hires_fix_denoising_strength"))
                if payload["hires_fix_denoising_strength"] < 0.0:
                    payload["hires_fix_denoising_strength"] = 0.0
                elif payload["hires_fix_denoising_strength"] > 1.0:
                    payload["hires_fix_denoising_strength"] = 1.0
            except ValueError:
                payload["hires_fix_denoising_strength"] = 1.0

    def _parameter_remap(self, payload: dict[str, str | None]) -> dict[str, str | None]:
        params = {}
        # Extract from the payload things we understand
        for key, value in payload.items():
            newkey = HordeLib.BASIC_INFERENCE_PARAMS.get(key, None)
            if newkey:
                params[newkey] = value

        # Inject model manager if needed
        if "model_loader.model_manager" not in params:
            params["model_loader.model_manager"] = SharedModelManager

        return params

    def _calculate_source_image_size(self, width, height):
        if width > 512 and height > 512:
            final_width = width
            final_height = height
            first_pass_ratio = min(final_height / 512, final_width / 512)
            width = (int(final_width / first_pass_ratio) // 64) * 64
            height = (int(final_height / first_pass_ratio) // 64) * 64
            return (width, height)
        return (width, height)

    def _parameter_remap_basic_inference(
        self,
        payload: dict[str, str | None],
    ) -> dict[str, str | None]:
        params = self._parameter_remap(payload)

        # XXX I think we need seed as an integer
        try:
            params["sampler.seed"] = int(params["sampler.seed"])
        except ValueError:
            # Now what? Pick a random one I guess?
            params["sampler.seed"] = random.randint(0, sys.maxsize)

        # Process dynamic prompts
        if new_prompt := DynamicPromptParser(params["sampler.seed"]).parse(payload.get("prompt", "")):
            payload["prompt"] = new_prompt

        # karras flag determines which scheduler we use
        if payload.get("karras", False):
            params["sampler.scheduler"] = "karras"
        else:
            params["sampler.scheduler"] = "normal"

        # We break prompt up on horde's "###"
        promptsCombined = payload.get("prompt", "")

        if promptsCombined is None:  # XXX
            raise TypeError("`None` value encountered!")

        promptsSplit = [x.strip() for x in promptsCombined.split("###")][:2]
        if len(promptsSplit) == 1:
            params["prompt.text"] = promptsSplit[0]
            params["negative_prompt.text"] = ""
        elif len(promptsSplit) == 2:
            params["prompt.text"] = promptsSplit[0]
            params["negative_prompt.text"] = promptsSplit[1]

        # Sampler remap
        sampler = HordeLib.SAMPLERS_MAP.get(params["sampler.sampler_name"], "unknown")
        if sampler == "unknown":
            logger.warning(f"Unknown sampler {params['sampler.sampler_name']} defaulting to euler")
            sampler = "euler"
        params["sampler.sampler_name"] = sampler

        # Clip skip inversion, comfy uses -1, -2, etc
        clip_skip_key = "clip_skip.stop_at_clip_layer"
        if params.get(clip_skip_key, 0) > 0:
            params[clip_skip_key] = -params[clip_skip_key]

        # If hires fix is enabled, use the same parameters as the main
        # sampler in our upscale sampler.
        if payload.get("hires_fix"):
            params["upscale_sampler.seed"] = params["sampler.seed"]
            params["upscale_sampler.scheduler"] = params["sampler.scheduler"]
            params["upscale_sampler.cfg"] = params["sampler.cfg"]
            params["upscale_sampler.steps"] = params["sampler.steps"]
            params["upscale_sampler.sampler_name"] = params["sampler.sampler_name"]
            if not params.get("upscale_sampler.denoise"):
                params["upscale_sampler.denoise"] = 0.65  # XXX 0.65 default upscaler denoising strength
            # Adjust image sizes
            width = params.get("empty_latent_image.width", 0)
            height = params.get("empty_latent_image.height", 0)
            if width > 512 and height > 512:
                newwidth, newheight = self._calculate_source_image_size(width, height)
                params["latent_upscale.width"] = width
                params["latent_upscale.height"] = height
                params["empty_latent_image.width"] = newwidth
                params["empty_latent_image.height"] = newheight
                # Finally mark that we are using hires fix
                params["hires_fix"] = True

        # ControlNet?
        if cnet := payload.get("control_type"):
            # Determine the pre-processor that was requested
            pre_processor = HordeLib.CONTROLNET_IMAGE_PREPROCESSOR_MAP.get(cnet)
            if not pre_processor:
                logger.warning("Unknown controlnet pre-processor type {cnet} defaulting to canny")
                pre_processor = "canny"

            # The controlnet type becomes a direct parameter to the pipeline
            # It is translated to its model as required my ComfyUI from the CN ModelManager
            params["controlnet_model_loader.control_net_name"] = cnet

            # For the pre-processor we dynamically reroute nodes in the pipeline later
            params["control_type"] = pre_processor

            # Remove the source_processing settings in case they conflict later
            if "source_processing" in params:
                del params["source_processing"]

        return params

    # Fix any nonsensical requests
    def _validate_BASIC_INFERENCE_PARAMS(self, payload):

        # Fix width/height not divisable by 65
        if payload["width"] % 64 != 0:
            payload["width"] = ((payload["width"] + 63) // 64) * 64

        if payload["height"] % 64 != 0:
            payload["height"] = ((payload["height"] + 63) // 64) * 64

        # Turn off hires fix if we're not generating a hires image
        if "hires_fix" in payload and (payload["width"] <= 512 or payload["height"] <= 512):
            payload["hires_fix"] = False

        # Remove source_processing if it's not valid
        img_proc = payload.get("source_processing")
        if img_proc and img_proc not in HordeLib.SOURCE_IMAGE_PROCESSING_OPTIONS:
            del payload["source_processing"]
        # Remove source image if we don't need it
        if payload.get("source_image"):
            if not img_proc or img_proc not in HordeLib.SOURCE_IMAGE_PROCESSING_OPTIONS:
                del payload["source_image"]

        # Turn off hires fix if we're painting as the dimensions are from the image
        if "hires_fix" in payload and (img_proc == "inpainting" or img_proc == "outpainting"):
            payload["hires_fix"] = False

        # Use denoising strength for both samplers if no second denoiser specified in hires fix
        # but not for txt2img where denoising will always generally be 1.0
        if payload.get("hires_fix"):
            if payload.get("source_processing") and payload.get("source_processing") != "txt2img":
                if not payload.get("hires_fix_denoising_strength"):
                    payload["hires_fix_denoising_strength"] = payload.get("denoising_strength")

        # Remap "denoising" to "controlnet strength" if that's what we actually wanted
        if payload.get("control_type"):
            if payload.get("denoising_strength"):
                if not payload.get("control_strength"):
                    payload["control_strength"] = payload["denoising_strength"]
                    del payload["denoising_strength"]
                else:
                    del payload["denoising_strength"]

    def _get_appropriate_pipeline(self, params):
        # Determine the correct pipeline based on the parameters we have
        pipeline = None

        # ControlNet
        if params.get("control_type"):
            if params.get("return_control_map", False):
                pipeline = "controlnet_annotator"
            else:
                if "hires_fix" in params:
                    del params["hires_fix"]
                    pipeline = "controlnet_hires_fix"
                else:
                    pipeline = "controlnet"

            return pipeline

        # Hires fix
        if "hires_fix" in params:
            del params["hires_fix"]
            pipeline = "stable_diffusion_hires_fix"
        else:
            pipeline = "stable_diffusion"

        # Source processing modes
        source_proc = params.get("source_processing")
        if source_proc:
            del params["source_processing"]
        if source_proc == "img2img":
            # FIXME: Probably will have a different name from BASIC_INFERENCE_PARAMS
            if params.get("source_mask"):
                pipeline = "stable_diffusion_paint"
            elif len(params.get("image_loader.image").split()) == 4:
                pipeline = "stable_diffusion_paint"
        elif source_proc == "inpainting":
            pipeline = "stable_diffusion_paint"
        elif source_proc == "outpainting":
            pipeline = "stable_diffusion_paint"

        return pipeline

    def _add_image_alpha_channel(self, source_image, alpha_image):
        # Convert images to RGBA mode
        source_image = source_image.convert("RGBA")
        # Convert alpha image to greyscale
        alpha_image = alpha_image.convert("L")
        # Create a new alpha channel from the second image
        alpha_image = ImageOps.invert(alpha_image)
        alpha_data = alpha_image.split()[0]
        source_image.putalpha(alpha_data)

        # Return the resulting image
        return source_image

    def _resize_sources_to_request(self, payload):
        """Ensures the source_image and source_mask are at the size requested by the client"""
        source_image = payload.get("source_image")
        if not source_image:
            return
        try:
            newwidth = payload["width"]
            newheight = payload["height"]
            if payload.get("hires_fix") or payload.get("control_type"):
                newwidth, newheight = self._calculate_source_image_size(payload["width"], payload["height"])
            if source_image.size != (newwidth, newheight):
                payload["source_image"] = source_image.resize(
                    (newwidth, newheight),
                    Image.Resampling.LANCZOS,
                )
        except (UnidentifiedImageError, AttributeError):
            logger.warning("Source image could not be parsed. Falling back to text2img")
            del payload["source_image"]
            del payload["source_processing"]
            return

        source_mask = payload.get("source_mask")
        if not source_mask:
            return
        try:
            if source_mask.size != (payload["width"], payload["height"]):
                payload["source_mask"] = source_mask.resize(
                    (payload["width"], payload["height"]),
                )
        except (UnidentifiedImageError, AttributeError):
            logger.warning(
                "Source mask could not be parsed. Falling back to img2img without mask",
            )
            del payload["source_mask"]

        if payload.get("source_mask"):
            payload["source_image"] = self._add_image_alpha_channel(payload["source_image"], payload["source_mask"])

    def _shrink_image(self, image, width, height, preserve_aspect=False):
        # Check if the provided image is an instance of the PIL.Image.Image class
        if not isinstance(image, Image.Image):
            logger.warning("Bad image passed to shrink_image")
            return

        # If both width and height are not specified, return
        if width is None and height is None:
            logger.warning("Bad image size passed to shrink_image")
            return

        # Only shrink
        if width >= image.width or height >= image.height:
            return image

        # Calculate new dimensions
        if preserve_aspect:
            aspect_ratio = float(image.width) / float(image.height)

            if width is not None:
                height = int(width / aspect_ratio)
            else:
                width = int(height * aspect_ratio)

        # Resize the image
        resized_image = image.resize((width, height), Image.LANCZOS)

        return resized_image

    def _copy_image_metadata(self, src_image, dest_image):
        metadata = src_image.info
        pnginfo = PngImagePlugin.PngInfo()
        for k, v in metadata.items():
            if k not in ("dpi", "gamma", "transparency", "aspect"):
                pnginfo.add_text(k, v)
        dest_image.info["pnginfo"] = pnginfo
        return dest_image

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

    def basic_inference(self, payload: dict[str, str | None], rawpng=False) -> Image.Image | None:
        # Check payload types
        self._check_payload(payload)
        # Validate our payload parameters
        self._validate_BASIC_INFERENCE_PARAMS(payload)
        self._resize_sources_to_request(payload)
        # Determine our parameters
        params = self._parameter_remap_basic_inference(payload)
        # Determine the correct pipeline
        pipeline = self._get_appropriate_pipeline(params)
        # Run the pipeline
        images = self.generator.run_image_pipeline(pipeline, params)
        return self._process_results(images, rawpng)

    def image_upscale(self, payload: dict[str, str | None], rawpng=False) -> Image.Image | None:
        # Check payload types
        self._check_payload(payload)
        # Determine our parameters
        params = self._parameter_remap(payload)
        # Determine the correct pipeline
        pipeline = "image_upscale"
        # Run the pipeline
        images = self.generator.run_image_pipeline(pipeline, params)
        if images is None:
            return None  # XXX Log error and/or raise Exception here
        # Allow arbitrary resizing
        width = payload.get("width")
        height = payload.get("height")
        if width or height:
            return self._shrink_image(Image.open(images[0]["imagedata"]), width, height)
        return self._process_results(images, rawpng)

    def image_facefix(self, payload: dict[str, str | None], rawpng=False) -> Image.Image | None:
        # Check payload types
        self._check_payload(payload)
        # Determine our parameters
        params = self._parameter_remap(payload)
        # Determine the correct pipeline
        pipeline = "image_facefix"
        # Run the pipeline
        images = self.generator.run_image_pipeline(pipeline, params)
        return self._process_results(images, rawpng)
