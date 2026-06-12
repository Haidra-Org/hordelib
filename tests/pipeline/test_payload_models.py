"""Clamp/coerce semantics of the typed payload models. No GPU required."""

import PIL.Image
import pytest

from hordelib.pipeline.payload import ImageGenPayload


class TestClampingSemantics:
    def test_defaults(self):
        payload = ImageGenPayload()
        assert payload.sampler_name == "k_euler"
        assert payload.cfg_scale == 8.0
        assert payload.width == 512
        assert payload.ddim_steps == 30

    def test_out_of_range_clamped(self):
        payload = ImageGenPayload.from_horde_dict({"cfg_scale": 1000, "ddim_steps": 0, "clip_skip": 99})
        assert payload.cfg_scale == 100
        assert payload.ddim_steps == 1
        assert payload.clip_skip == 20

    def test_type_coercion(self):
        payload = ImageGenPayload.from_horde_dict({"cfg_scale": "7.5", "width": "640", "hires_fix": 1})
        assert payload.cfg_scale == 7.5
        assert payload.width == 640
        assert payload.hires_fix is True

    def test_uncoercible_falls_back_to_default(self):
        payload = ImageGenPayload.from_horde_dict({"cfg_scale": "not a number", "ddim_steps": object()})
        assert payload.cfg_scale == 8.0
        assert payload.ddim_steps == 30

    def test_width_rounded_up_to_64(self):
        payload = ImageGenPayload.from_horde_dict({"width": 513, "height": 700})
        assert payload.width == 576
        assert payload.height == 704

    def test_unknown_enum_value_falls_back(self):
        payload = ImageGenPayload.from_horde_dict({"sampler_name": "not_a_sampler", "scheduler": "bogus"})
        assert payload.sampler_name == "k_euler"
        assert payload.scheduler == "normal"

    def test_enum_value_lowercased(self):
        payload = ImageGenPayload.from_horde_dict({"sampler_name": "K_DPMPP_2M"})
        assert payload.sampler_name == "k_dpmpp_2m"

    def test_unknown_keys_ignored(self):
        payload = ImageGenPayload.from_horde_dict({"no_such_key": 123, "prompt": "hello"})
        assert payload.prompt == "hello"

    def test_seed_random_when_missing(self):
        a = ImageGenPayload.from_horde_dict({})
        b = ImageGenPayload.from_horde_dict({})
        # Astronomically unlikely to collide; the legacy schema had a process-lifetime constant here
        assert isinstance(a.seed, int)
        assert a.seed != b.seed

    def test_seed_coerced_from_string(self):
        payload = ImageGenPayload.from_horde_dict({"seed": "1234"})
        assert payload.seed == 1234


class TestSubEntryDropping:
    def test_invalid_loras_dropped(self):
        payload = ImageGenPayload.from_horde_dict(
            {
                "loras": [
                    {"name": "good_lora", "model": 99.0},
                    {"model": 1.0},  # no name -> dropped
                    "not a dict",  # dropped
                ],
            },
        )
        assert len(payload.loras) == 1
        assert payload.loras[0].name == "good_lora"
        assert payload.loras[0].model == 10.0  # clamped

    def test_invalid_tis_dropped(self):
        payload = ImageGenPayload.from_horde_dict({"tis": [{"strength": 2.0}, {"name": "ti1"}]})
        assert len(payload.tis) == 1
        assert payload.tis[0].name == "ti1"

    def test_extra_texts_without_text_dropped(self):
        payload = ImageGenPayload.from_horde_dict(
            {"extra_texts": [{"reference": "qr_code"}, {"text": "hello", "reference": "qr_code"}]},
        )
        assert len(payload.extra_texts) == 1

    def test_extra_source_images_without_image_dropped(self):
        image = PIL.Image.new("RGB", (8, 8))
        payload = ImageGenPayload.from_horde_dict(
            {"extra_source_images": [{"strength": 1.0}, {"image": image}]},
        )
        assert len(payload.extra_source_images) == 1

    def test_non_list_lora_payload(self):
        payload = ImageGenPayload.from_horde_dict({"loras": "garbage"})
        assert payload.loras == []


class TestDictDumpContract:
    """``model_dump`` keeps the dict shape that wire-format consumers rely on."""

    CASES = [
        {"cfg_scale": 1000.5, "width": 513, "sampler_name": "K_EULER_A"},
        {"ddim_steps": "25", "clip_skip": -5, "scheduler": "karras"},
        {"control_type": "canny", "control_strength": 99},
        {"control_type": "unknown_type", "denoising_strength": 0.0},
        {"source_processing": "IMG2IMG", "n_iter": 1000},
        {"prompt": 12345, "negative_prompt": None},
    ]

    @pytest.mark.parametrize("case", CASES)
    def test_dump_round_trips(self, case):
        result = ImageGenPayload.from_horde_dict(dict(case)).model_dump(warnings=False)
        assert isinstance(result, dict)
        # The dump must be re-validatable (faulty values were already clamped/coerced)
        revalidated = ImageGenPayload.from_horde_dict(dict(result)).model_dump(warnings=False)
        assert revalidated == result

    def test_sub_entries_are_plain_dicts(self):
        result = ImageGenPayload.from_horde_dict(
            {"loras": [{"name": "a_lora", "model": 0.5}], "tis": [{"name": "a_ti"}]},
        ).model_dump(warnings=False)
        assert result["loras"] == [
            {"name": "a_lora", "model": 0.5, "clip": 1.0, "inject_trigger": None, "is_version": None},
        ]
        assert result["tis"] == [{"name": "a_ti", "inject_ti": None, "strength": 1.0}]
