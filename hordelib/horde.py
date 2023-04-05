# horde.py
# Main interface for the horde to this library.
import contextlib

from PIL import Image

from hordelib.comfy_horde import Comfy_Horde
from hordelib.model_manager.hyper import ModelManager


class SharedModelManager:
    _instance = None
    manager: ModelManager | None = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def loadModelManagers(
        cls,
        # aitemplate: bool = False,
        blip: bool = False,
        clip: bool = False,
        codeformer: bool = False,
        compvis: bool = False,
        controlnet: bool = False,
        diffusers: bool = False,
        # esrgan: bool = False,
        # gfpgan: bool = False,
        safety_checker: bool = False,
    ):
        if cls.manager is None:
            cls.manager = ModelManager()

        args_passed = locals().copy()
        args_passed.pop("cls")

        cls.manager.init_model_managers(**args_passed)


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
        # "karras": false,
        "tiling": None,
        "hires_fix": None,
        "clip_skip": "clip_skip.stop_at_clip_layer",
        "control_type": None,
        "image_is_control": None,
        "return_control_map": None,
        # "prompt": "string",
        "ddim_steps": "sampler.steps",
        "n_iter": "empty_latent_image.batch_size",
        "model": "model_loader.ckpt_name",
    }

    def __init__(self) -> None:
        pass

    def _parameter_remap(self, payload: dict[str, str | None]) -> dict[str, str | None]:
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

        return params

    def text_to_image(self, payload: dict[str, str | None]) -> Image.Image | None:
        generator = Comfy_Horde()
        images = generator.run_image_pipeline(
            "stable_diffusion", self._parameter_remap(payload)
        )
        if images is None:
            return None  # XXX Log error and/or raise Exception here
        # XXX Assumes the horde only asks for and wants 1 image
        return Image.open(images[0]["imagedata"])
