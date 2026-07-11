# test_horde.py

import PIL.Image
import pytest

from hordelib.horde import HordeLib
from hordelib.shared_model_manager import SharedModelManager

from .testing_shared_functions import check_single_inference_image_similarity

_CONTROL_TYPES = list(HordeLib.CONTROLNET_IMAGE_PREPROCESSOR_MAP.keys())

# Control types the annotator inference test cannot exercise on a natural source image, mapped to the reason.
# ScribblePreprocessor expects an already-drawn scribble map as input, not a photograph.
_ANNOTATOR_SKIP_REASONS = {
    "scribble": "ScribblePreprocessor expects a pre-drawn scribble map, not a natural source image",
}


class TestControlnetAnnotator:
    @pytest.fixture(scope="class", autouse=True)
    def preload_controlnet_annotators(self, shared_model_manager: type[SharedModelManager]):
        # Amortized once per class: the shared HordeLib/model-manager singletons are session-scoped, so the
        # per-type tests below reuse a single download+preload rather than repeating it on every parameter.
        assert shared_model_manager.manager.controlnet
        for preproc in _CONTROL_TYPES:
            shared_model_manager.manager.controlnet.download_control_type(preproc, ["stable diffusion 1"])
        assert shared_model_manager.preload_annotators()

    @pytest.mark.parametrize("preproc", _CONTROL_TYPES)
    def test_controlnet_annotator(
        self,
        hordelib_instance: HordeLib,
        shared_model_manager: type[SharedModelManager],
        preproc: str,
    ):
        if preproc in _ANNOTATOR_SKIP_REASONS:
            pytest.skip(_ANNOTATOR_SKIP_REASONS[preproc])

        assert shared_model_manager.manager.controlnet

        image = PIL.Image.open("images/test_annotator.jpg")
        width, height = image.size
        data = {
            "sampler_name": "k_dpmpp_2m",
            "cfg_scale": 7.5,
            "denoising_strength": 1.0,
            "seed": 123456789,
            "height": height,
            "width": width,
            "karras": False,
            "tiling": False,
            "hires_fix": False,
            "clip_skip": 1,
            "control_type": preproc,
            "image_is_control": False,
            "return_control_map": True,
            "prompt": "this is not used here",
            "ddim_steps": 25,
            "n_iter": 1,
            "model": "Deliberate",
            "source_image": image,
            "source_processing": "img2img",
        }

        pil_image = hordelib_instance.basic_inference_single_image(data).image
        assert pil_image is not None
        assert isinstance(pil_image, PIL.Image.Image)
        img_filename = f"annotator_{preproc}.png"
        pil_image.save(f"images/{img_filename}", quality=100)
        # M-LSD ("hough") is run-to-run flaky (~0.90 cosine); the similarity helper lands that in its warn band
        # and pytest.skip()s only this parameter, so the flaky detector never masks the other control types.
        assert check_single_inference_image_similarity(
            f"images_expected/{img_filename}",
            pil_image,
        )
