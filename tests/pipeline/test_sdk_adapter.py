"""The horde_sdk generation-parameters adapter: mapping, overrides, and fault behavior.

No GPU required; the adapter is pure. The parameters are built the way the SDK's AI-Horde
converter builds them, so these tests also pin the worker-facing end-to-end mapping.
"""

import io
import uuid

import PIL.Image
from horde_sdk.ai_horde_api.consts import METADATA_TYPE
from horde_sdk.generation_parameters.image import (
    BasicImageGenerationParameters,
    ControlnetGenerationParameters,
    CustomWorkflowGenerationParameters,
    ExtraTextEntry,
    HiresFixGenerationParameters,
    Image2ImageGenerationParameters,
    ImageGenerationParameters,
    LoRaEntry,
    RemixGenerationParameters,
    RemixImageEntry,
    TIEntry,
)
from horde_sdk.generation_parameters.image.consts import (
    LORA_TRIGGER_INJECT_CHOICE,
    TI_TRIGGER_INJECT_CHOICE,
)
from horde_sdk.generation_parameters.image.object_models import ImageGenerationComponentContainer

from hordelib.pipeline.sdk_adapter import to_image_gen_payload


def _png_bytes(size: tuple[int, int] = (8, 8), mode: str = "RGB") -> bytes:
    buffer = io.BytesIO()
    PIL.Image.new(mode, size).save(buffer, format="PNG")
    return buffer.getvalue()


def _make_params(
    *,
    components: list | None = None,
    base_overrides: dict | None = None,
    batch_size: int = 1,
    source_processing: str = "txt2img",
) -> ImageGenerationParameters:
    base_fields = {
        "model": "Deliberate",
        "prompt": "a cat in a hat",
        "negative_prompt": "ugly",
        "seed": "42",
        "width": 512,
        "height": 768,
        "steps": 25,
        "cfg_scale": 7.5,
        "sampler_name": "k_euler",
        "scheduler": "karras",
        "clip_skip": 2,
        "denoising_strength": 0.75,
        "tiling": True,
        "transparent": True,
    }
    if base_overrides:
        base_fields.update(base_overrides)

    return ImageGenerationParameters(
        result_ids=[uuid.uuid4() for _ in range(batch_size)],
        batch_size=batch_size,
        source_processing=source_processing,
        base_params=BasicImageGenerationParameters(**base_fields),
        additional_params=ImageGenerationComponentContainer(components=components or []),
    )


def test_base_parameter_mapping() -> None:
    """The flat base parameters map onto their payload counterparts."""
    payload, faults = to_image_gen_payload(_make_params(batch_size=3))

    assert not faults
    assert payload.model == "Deliberate"
    assert payload.model_name == "Deliberate"
    assert payload.prompt == "a cat in a hat"
    assert payload.negative_prompt == "ugly"
    assert payload.seed == 42
    assert payload.width == 512
    assert payload.height == 768
    assert payload.ddim_steps == 25
    assert payload.cfg_scale == 7.5
    assert payload.sampler_name == "k_euler"
    assert payload.scheduler == "karras"
    assert payload.clip_skip == 2
    assert payload.denoising_strength == 0.75
    assert payload.tiling is True
    assert payload.transparent is True
    assert payload.n_iter == 3
    # txt2img is the payload's implicit default (None), matching the dict path's clamping
    assert payload.source_processing is None
    assert payload.hires_fix is False


def test_img2img_component_decodes_images() -> None:
    """The img2img component's bytes become PIL images on the payload."""
    components = [
        Image2ImageGenerationParameters(
            source_image=_png_bytes(),
            source_mask=_png_bytes(mode="L"),
        ),
    ]
    payload, faults = to_image_gen_payload(_make_params(components=components, source_processing="img2img"))

    assert not faults
    assert isinstance(payload.source_image, PIL.Image.Image)
    assert isinstance(payload.source_mask, PIL.Image.Image)
    assert payload.source_processing == "img2img"


def test_undecodable_source_image_records_fault() -> None:
    """A corrupt source image degrades to None with a recorded fault, never raising."""
    components = [
        Image2ImageGenerationParameters(
            source_image=b"not a png",
            source_mask=None,
        ),
    ]
    payload, faults = to_image_gen_payload(_make_params(components=components, source_processing="img2img"))

    assert payload.source_image is None
    assert any(fault.type_ == METADATA_TYPE.source_image for fault in faults)


def test_remix_component_maps_extra_source_images() -> None:
    """The remix component maps its images to the payload's extra source images."""
    components = [
        RemixGenerationParameters(
            source_image=_png_bytes(),
            remix_images=[
                RemixImageEntry(image=_png_bytes(), strength=0.5),
                RemixImageEntry(image=b"corrupt", strength=1.5),
            ],
        ),
    ]
    payload, faults = to_image_gen_payload(_make_params(components=components, source_processing="remix"))

    assert isinstance(payload.source_image, PIL.Image.Image)
    assert payload.source_processing == "remix"
    # The corrupt entry is dropped with a fault referencing its index
    assert len(payload.extra_source_images) == 1
    assert payload.extra_source_images[0].strength == 0.5
    assert any(fault.type_ == METADATA_TYPE.extra_source_images and fault.ref == "1" for fault in faults)


def test_controlnet_component_with_control_map() -> None:
    """A pre-made control map sets image_is_control and lands in source_image."""
    components = [
        ControlnetGenerationParameters(
            controlnet_type="canny",
            source_image=None,
            control_map=_png_bytes(),
            return_control_map=True,
        ),
    ]
    payload, faults = to_image_gen_payload(_make_params(components=components))

    assert not faults
    assert payload.control_type == "canny"
    assert payload.image_is_control is True
    assert payload.return_control_map is True
    assert isinstance(payload.source_image, PIL.Image.Image)


def test_controlnet_component_with_source_image() -> None:
    """A source image for preprocessing keeps image_is_control off."""
    components = [
        ControlnetGenerationParameters(
            controlnet_type="canny",
            source_image=_png_bytes(),
            control_map=None,
        ),
    ]
    payload, faults = to_image_gen_payload(_make_params(components=components))

    assert not faults
    assert payload.control_type == "canny"
    assert payload.image_is_control is False
    assert payload.return_control_map is False
    assert isinstance(payload.source_image, PIL.Image.Image)


def test_hires_fix_component_sets_two_pass_overrides() -> None:
    """The hires component's explicit two-pass values become authoritative payload overrides."""
    first_pass = BasicImageGenerationParameters(
        model="Deliberate",
        prompt="a cat in a hat",
        width=512,
        height=512,
        steps=25,
        denoising_strength=0.7,
    )
    second_pass = BasicImageGenerationParameters(
        model="Deliberate",
        prompt="a cat in a hat",
        width=1024,
        height=1024,
        steps=13,
        denoising_strength=0.6,
    )
    components = [HiresFixGenerationParameters(first_pass=first_pass, second_pass=second_pass)]
    payload, faults = to_image_gen_payload(_make_params(components=components))

    assert not faults
    assert payload.hires_fix is True
    assert payload.width == 1024
    assert payload.height == 1024
    assert payload.ddim_steps == 25
    assert payload.denoising_strength == 0.7
    assert payload.hires_fix_first_pass_width == 512
    assert payload.hires_fix_first_pass_height == 512
    assert payload.hires_fix_second_pass_steps == 13
    assert payload.hires_fix_denoising_strength == 0.6


def test_lora_entries_round_trip() -> None:
    """LoRa entries map back to the payload's name/is_version/trigger encoding."""
    components = [
        LoRaEntry(
            name="GlowingRunesAI",
            remote_version_id=None,
            source="CIVITAI",
            model_strength=0.8,
            clip_strength=0.6,
            lora_triggers=["blue"],
            lora_inject_trigger_choice=LORA_TRIGGER_INJECT_CHOICE.FUZZY_POSITIVE,
        ),
        LoRaEntry(
            name=None,
            remote_version_id="76693",
            source="CIVITAI",
            model_strength=1.0,
            clip_strength=1.0,
            lora_inject_trigger_choice=LORA_TRIGGER_INJECT_CHOICE.NO_INJECT,
        ),
    ]
    payload, faults = to_image_gen_payload(_make_params(components=components))

    assert not faults
    assert len(payload.loras) == 2

    by_name_lora, by_version_lora = payload.loras
    assert by_name_lora.name == "GlowingRunesAI"
    assert by_name_lora.is_version is False
    assert by_name_lora.model == 0.8
    assert by_name_lora.clip == 0.6
    assert by_name_lora.inject_trigger == "blue"

    assert by_version_lora.name == "76693"
    assert by_version_lora.is_version is True
    assert by_version_lora.inject_trigger is None


def test_ti_entries_round_trip() -> None:
    """TI entries map their inject choice back to the payload's prompt/negprompt encoding."""
    components = [
        TIEntry(
            name="72437",
            remote_version_id=None,
            source="HORDELING",
            model_strength=0.5,
            ti_inject_trigger_choice=TI_TRIGGER_INJECT_CHOICE.NEGATIVE_PROMPT,
        ),
        TIEntry(
            name="7808",
            remote_version_id=None,
            source="HORDELING",
            model_strength=1.0,
            ti_inject_trigger_choice=TI_TRIGGER_INJECT_CHOICE.POSITIVE_PROMPT,
        ),
    ]
    payload, faults = to_image_gen_payload(_make_params(components=components))

    assert not faults
    assert len(payload.tis) == 2
    assert payload.tis[0].inject_ti == "negprompt"
    assert payload.tis[0].strength == 0.5
    assert payload.tis[1].inject_ti == "prompt"


def test_custom_workflow_carries_extra_texts() -> None:
    """The workflow name and its extra texts land on the payload."""
    components = [
        CustomWorkflowGenerationParameters(
            custom_workflow_name="qr_code",
            extra_texts=[ExtraTextEntry(text="https://aihorde.net", reference="qr_text")],
        ),
    ]
    payload, faults = to_image_gen_payload(_make_params(components=components))

    assert not faults
    assert payload.workflow == "qr_code"
    assert len(payload.extra_texts) == 1
    assert payload.extra_texts[0].text == "https://aihorde.net"
    assert payload.extra_texts[0].reference == "qr_text"


def test_qr_code_image_fed_inherits_denoise() -> None:
    """An image-fed qr_code generation inherits the first-pass denoise for its composite sampler."""
    components = [
        Image2ImageGenerationParameters(source_image=_png_bytes(), source_mask=None),
        CustomWorkflowGenerationParameters(
            custom_workflow_name="qr_code",
            extra_texts=[ExtraTextEntry(text="https://aihorde.net", reference="qr_text")],
        ),
    ]
    payload, _faults = to_image_gen_payload(
        _make_params(components=components, source_processing="img2img"),
    )

    assert payload.hires_fix_denoising_strength == payload.denoising_strength == 0.75
