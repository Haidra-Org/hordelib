# horde.py
# Main interface for the horde to this library.
from hordelib.comfy import Comfy
from PIL import Image
import contextlib


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
    # FIXME Items mapped to None are ignored for now
    TEXT_TO_IMAGE_PARAMS = {
        "sampler_name": "sampler.sampler_name",
        "cfg_scale": "sampler.cfg",
        "denoising_strength": "sampler.denoise",
        "seed": "sampler.seed",
        "height": "empty_latent_image.height",
        "width": "empty_latent_image.width",
        # "karras": false,
        "tiling": None,
        "hires_fix": None,
        "clip_skip": None,
        "control_type": None,
        "image_is_control": None,
        "return_control_map": None,
        # "prompt": "string",
        "ddim_steps": "sampler.steps",
        "n_iter": "empty_latent_image.batch_size",
        "model": "model_loader.ckpt_name",
    }

    def __init__(self):
        pass

    def _parameter_remap(self, payload):
        params = {}
        # Extract from the payload things we understand
        for key, value in payload.items():
            newkey = HordeLib.TEXT_TO_IMAGE_PARAMS.get(key, None)
            if newkey:
                params[newkey] = value

        # XXX I think we need seed as an integer
        with contextlib.suppress(ValueError):
            params["sampler.seed"] = int(params["sampler.seed"])

        # karras flag determines which scheduler we use
        if payload.get("karras", False):
            params["sampler.scheduler"] = "karras"
        else:
            params["sampler.scheduler"] = "normal"

        # We break prompt up on horde's "###"
        prompts = [x.strip() for x in payload.get("prompt", "").split("###")][:2]
        if len(prompts) == 1:
            params["prompt.text"] = prompts[0]
            params["negative_prompt.text"] = ""
        elif len(prompts) == 2:
            params["prompt.text"] = prompts[0]
            params["negative_prompt.text"] = prompts[1]

        # Sampler remap
        sampler = HordeLib.SAMPLERS_MAP.get(params["sampler.sampler_name"], "euler")
        params["sampler.sampler_name"] = sampler

        return params

    def text_to_image(self, payload):

        generator = Comfy()
        images = generator.run_image_pipeline(
            "stable_diffusion", self._parameter_remap(payload)
        )
        # XXX Assumes the horde only asks for and wants 1 image
        image = Image.open(images[0]["imagedata"])
        return image
