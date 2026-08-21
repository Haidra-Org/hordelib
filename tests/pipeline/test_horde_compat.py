"""The hires-fix guard for masked img2img. No GPU required.

The masked img2img graph is single-pass, so a payload that reaches materialization with both a
mask and hires fix has already had its source image shrunk to the first-pass resolution, and the
render returns at that size instead of the requested one. The guard turns hires fix off for
masked img2img before any resizing happens; controlnet payloads keep it because the controlnet
hires graph has a real second pass.
"""

import PIL.Image

from hordelib.pipeline.horde_compat import disable_hires_fix_for_masked_img2img, resize_sources_to_request
from hordelib.pipeline.payload import ImageGenPayload

BASE = {
    "seed": 1,
    "prompt": "a guard test prompt",
    "width": 1024,
    "height": 1024,
}


def _payload(**overrides) -> ImageGenPayload:
    return ImageGenPayload.from_horde_dict({**BASE, **overrides})


def _rgb(size: tuple[int, int] = (1024, 1024)) -> PIL.Image.Image:
    return PIL.Image.new("RGB", size)


def _rgba(size: tuple[int, int] = (1024, 1024)) -> PIL.Image.Image:
    return PIL.Image.new("RGBA", size)


class TestDisableHiresFixForMaskedImg2img:
    def test_explicit_mask_disables_hires_fix(self):
        payload = _payload(
            hires_fix=True,
            source_processing="img2img",
            source_image=_rgb(),
            source_mask=_rgb(),
        )
        assert disable_hires_fix_for_masked_img2img(payload).hires_fix is False

    def test_alpha_channel_mask_disables_hires_fix(self):
        payload = _payload(hires_fix=True, source_processing="img2img", source_image=_rgba())
        assert disable_hires_fix_for_masked_img2img(payload).hires_fix is False

    def test_controlnet_keeps_hires_fix(self):
        # Mask plus controlnet selects the controlnet hires graph, which has a second pass.
        payload = _payload(
            hires_fix=True,
            control_type="canny",
            source_processing="img2img",
            source_image=_rgb(),
            source_mask=_rgb(),
        )
        assert disable_hires_fix_for_masked_img2img(payload).hires_fix is True

    def test_txt2img_keeps_hires_fix(self):
        payload = _payload(hires_fix=True)
        assert disable_hires_fix_for_masked_img2img(payload).hires_fix is True

    def test_unmasked_img2img_keeps_hires_fix(self):
        payload = _payload(hires_fix=True, source_processing="img2img", source_image=_rgb())
        assert disable_hires_fix_for_masked_img2img(payload).hires_fix is True

    def test_inpainting_is_out_of_scope(self):
        # The painting rules own inpainting/outpainting; this guard only covers img2img.
        payload = _payload(hires_fix=True, source_processing="inpainting", source_image=_rgba())
        assert disable_hires_fix_for_masked_img2img(payload).hires_fix is True

    def test_disabling_returns_a_copy(self):
        payload = _payload(
            hires_fix=True,
            source_processing="img2img",
            source_image=_rgb(),
            source_mask=_rgb(),
        )
        guarded = disable_hires_fix_for_masked_img2img(payload)
        assert guarded is not payload
        assert payload.hires_fix is True


class TestGuardBeforeResize:
    """The guard must precede the resize: it keeps the source at the requested size."""

    def test_guarded_masked_img2img_source_stays_at_request_size(self):
        payload = _payload(
            hires_fix=True,
            source_processing="img2img",
            source_image=_rgb(),
            source_mask=_rgb(),
        )
        resized = resize_sources_to_request(disable_hires_fix_for_masked_img2img(payload))
        assert resized.source_image is not None
        assert resized.source_image.size == (BASE["width"], BASE["height"])
        # The mask ends up merged into the source alpha either way.
        assert len(resized.source_image.getbands()) == 4

    def test_without_the_guard_the_source_is_shrunk(self):
        payload = _payload(
            hires_fix=True,
            source_processing="img2img",
            source_image=_rgb(),
            source_mask=_rgb(),
        )
        resized = resize_sources_to_request(payload)
        assert resized.source_image is not None
        assert resized.source_image.size != (BASE["width"], BASE["height"])
