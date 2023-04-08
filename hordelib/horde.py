# horde.py
# Main interface for the horde to this library.
import contextlib

from PIL import Image

from hordelib.comfy_horde import Comfy_Horde
from hordelib.shared_model_manager import SharedModelManager


class HordeLib:
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
        "plms": "<not supported>",
    }

    # Horde to tex2img parameter mapping
    # XXX Items mapped to None are ignored for now
    TEXT_TO_IMAGE_PARAMS = {
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
        "control_type": None,
        "image_is_control": None,
        "return_control_map": None,
        # "prompt": Handled below
        "ddim_steps": "sampler.steps",
        "n_iter": "empty_latent_image.batch_size",
        "model": "model_loader.model_name",
        "source_image": "image_loader.image",
        "source_mask": None,
        "source_processing": "source_processing",
    }

    SOURCE_IMAGE_PROCESSING_OPTIONS = ["img2img", "inpainting", "outpainting"]

    def _parameter_remap(self, payload: dict[str, str | None]) -> dict[str, str | None]:
        params = {}
        # Extract from the payload things we understand
        for key, value in payload.items():
            newkey = HordeLib.TEXT_TO_IMAGE_PARAMS.get(key, None)
            if newkey:
                params[newkey] = value

        # Inject model manager if needed
        if "model_loader.model_manager" not in params:
            params["model_loader.model_manager"] = SharedModelManager

        return params

    def _parameter_remap_text_to_image(
        self,
        payload: dict[str, str | None],
    ) -> dict[str, str | None]:
        params = self._parameter_remap(payload)

        # XXX I think we need seed as an integer
        with contextlib.suppress(ValueError):
            params["sampler.seed"] = int(params["sampler.seed"])

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
        sampler = HordeLib.SAMPLERS_MAP.get(params["sampler.sampler_name"], "euler")
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
            params[
                "upscale_sampler.denoise"
            ] = 0.6  # XXX is this ok for latent upscale denoise?
            # Adjust image sizes
            width = params.get("empty_latent_image.width", 0)
            height = params.get("empty_latent_image.height", 0)
            if width > 512 and height > 512:
                final_width = width
                final_height = height
                params["latent_upscale.width"] = final_width
                params["latent_upscale.height"] = final_height
                first_pass_ratio = min(final_height / 512, final_width / 512)
                width = (int(final_width / first_pass_ratio) // 64) * 64
                height = (int(final_height / first_pass_ratio) // 64) * 64
                params["empty_latent_image.width"] = width
                params["empty_latent_image.height"] = height
                # Finally mark that we are using hires fix
                params["hires_fix"] = True

        return params

    # Fix any nonsensical requests
    def _validate_text_to_image_params(self, payload):
        # Turn off hires fix if we're not generating a hires image
        if "hires_fix" in payload and (
            payload["width"] <= 512 or payload["height"] <= 512
        ):
            payload["hires_fix"] = False

        # Remove source_processing if it's not valid
        if (
            payload.get("source_processing")
            and payload.get("source_processing")
            not in HordeLib.SOURCE_IMAGE_PROCESSING_OPTIONS
        ):
            del payload["source_processing"]

        # Remove source image if we don't need it
        if payload.get("source_image"):
            if (
                "source_processing" not in payload
                or payload["source_processing"]
                not in HordeLib.SOURCE_IMAGE_PROCESSING_OPTIONS
            ):
                del payload["source_image"]

    def _get_appropriate_pipeline(self, params):
        # Determine the correct pipeline based on the parameters we have

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
            pass  # doesn't impact pipeline
        elif source_proc == "inpainting":
            pipeline = "stable_diffusion_paint"
        elif source_proc == "outpainting":
            pipeline = "stable_diffusion_paint"

        return pipeline

    def text_to_image(self, payload: dict[str, str | None]) -> Image.Image | None:
        generator = Comfy_Horde()
        # Validate our payload parameters
        params = self._validate_text_to_image_params(payload)
        # Determine our parameters
        params = self._parameter_remap_text_to_image(payload)
        # Determine the correct pipeline
        pipeline = self._get_appropriate_pipeline(params)
        # Run the pipeline
        images = generator.run_image_pipeline(pipeline, params)
        if images is None:
            return None  # XXX Log error and/or raise Exception here
        # XXX Assumes the horde only asks for and wants 1 image
        return Image.open(images[0]["imagedata"])

    def image_upscale(self, payload: dict[str, str | None]) -> Image.Image | None:
        generator = Comfy_Horde()
        # Determine our parameters
        params = self._parameter_remap(payload)
        # Determine the correct pipeline
        pipeline = "image_upscale"
        # Run the pipeline
        images = generator.run_image_pipeline(pipeline, params)
        if images is None:
            return None  # XXX Log error and/or raise Exception here
        # XXX Assumes the horde only asks for and wants 1 image
        return Image.open(images[0]["imagedata"])
