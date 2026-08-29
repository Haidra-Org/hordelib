"""Golden-graph tests for the pure patch operations. No GPU required."""

import importlib.util
import json
from pathlib import Path

import PIL.Image
import pytest
from horde_model_reference.meta_consts import KNOWN_IMAGE_GENERATION_BASELINE

from hordelib.execution.graph_utils import fix_node_names
from hordelib.pipeline.patches import (
    SHUFFLE_PREPROCESSOR_CLASS_TYPE,
    FlowShiftNode,
    RemixImage,
    ResolvedLora,
    apply_layerdiffuse,
    configure_controlnet,
    hires_fix_first_pass_resolution,
    insert_lora_chain,
    insert_model_sampling_shift,
    insert_remix_image_chain,
    qr_layout_params,
    qr_params_from_extra_texts,
    rewire_cascade_img2img,
    rewire_img2img,
)

PIPELINES_DIR = Path(__file__).parent.parent.parent / "hordelib" / "pipelines"


def _graph(name: str) -> dict:
    raw = json.loads((PIPELINES_DIR / f"pipeline_{name}.json").read_text(encoding="utf-8"))
    return fix_node_names(raw)


def _sd_graph() -> dict:
    return _graph("stable_diffusion")


def _flux_graph() -> dict:
    return _graph("flux")


def _fake_find_spec(present: set[str]):
    """Return a ``find_spec`` replacement reporting only *present* top-level packages as importable."""

    def _find_spec(name: str, package: str | None = None) -> object | None:
        return object() if name.split(".", 1)[0] in present else None

    return _find_spec


class TestInsertLoraChain:
    def test_no_loras_is_a_noop(self):
        graph = _sd_graph()
        before = json.dumps(graph, default=str, sort_keys=True)
        insert_lora_chain(graph, [])
        assert json.dumps(graph, default=str, sort_keys=True) == before

    def test_single_lora(self):
        graph = _sd_graph()
        insert_lora_chain(graph, [ResolvedLora(filename="a.safetensors", strength_model=0.8, strength_clip=0.7)])

        assert graph["lora_0"]["class_type"] == "HordeLoraLoader"
        assert graph["lora_0"]["inputs"]["model"] == ["model_loader", 0]
        assert graph["lora_0"]["inputs"]["clip"] == ["model_loader", 1]
        assert graph["lora_0"]["inputs"]["strength_model"] == 0.8
        assert graph["sampler"]["inputs"]["model"][0] == "lora_0"
        assert graph["clip_skip"]["inputs"]["clip"][0] == "lora_0"

    def test_three_lora_chain(self):
        graph = _sd_graph()
        loras = [ResolvedLora(filename=f"l{i}.safetensors", strength_model=1.0, strength_clip=1.0) for i in range(3)]
        insert_lora_chain(graph, loras)

        assert graph["lora_0"]["inputs"]["model"] == ["model_loader", 0]
        assert graph["lora_1"]["inputs"]["model"] == ["lora_0", 0]
        assert graph["lora_2"]["inputs"]["model"] == ["lora_1", 0]
        assert graph["lora_1"]["inputs"]["clip"] == ["lora_0", 1]
        # Only the last lora feeds the consumers
        assert graph["sampler"]["inputs"]["model"][0] == "lora_2"
        assert graph["clip_skip"]["inputs"]["clip"][0] == "lora_2"

    def test_missing_upscale_sampler_is_skipped(self):
        graph = _sd_graph()
        assert "upscale_sampler" not in graph
        insert_lora_chain(graph, [ResolvedLora(filename="a", strength_model=1, strength_clip=1)])
        assert "upscale_sampler" not in graph  # no phantom node created

    def test_split_component_loaders_patch_model_and_clip(self):
        graph = _graph("qwen")
        insert_lora_chain(
            graph,
            [ResolvedLora(filename="a.safetensors", strength_model=0.8, strength_clip=0.7)],
        )

        assert graph["lora_0"]["inputs"]["model"] == ["model_loader", 0]
        assert graph["lora_0"]["inputs"]["clip"] == ["clip_loader", 0]
        assert graph["prompt"]["inputs"]["clip"] == ["lora_0", 1]
        assert graph["negative_prompt"]["inputs"]["clip"] == ["lora_0", 1]
        assert graph["sampler"]["inputs"]["model"] == ["lora_0", 0]

    def test_flux_targets(self):
        graph = _flux_graph()
        insert_lora_chain(
            graph,
            [ResolvedLora(filename="a.safetensors", strength_model=1.0, strength_clip=1.0)],
            flux=True,
        )
        assert graph["cfg_guider"]["inputs"]["model"][0] == "lora_0"
        assert graph["basic_scheduler"]["inputs"]["model"][0] == "lora_0"
        # The non-flux targets must be untouched
        assert "sampler" not in graph or graph["sampler"]["inputs"].get("model", [None])[0] != "lora_0"


class TestInsertModelSamplingShift:
    def test_no_shift_is_a_noop(self):
        graph = _flux_graph()
        before = json.dumps(graph, default=str, sort_keys=True)
        insert_model_sampling_shift(graph, None, node=FlowShiftNode.FLUX)
        assert json.dumps(graph, default=str, sort_keys=True) == before

    def test_flux_gets_the_node_between_the_model_and_its_consumers(self):
        graph = _flux_graph()
        model_source = graph["cfg_guider"]["inputs"]["model"][0]

        insert_model_sampling_shift(graph, 2.5, node=FlowShiftNode.FLUX)

        assert graph["model_sampling_shift"]["class_type"] == "ModelSamplingFlux"
        assert graph["model_sampling_shift"]["inputs"]["model"] == [model_source, 0]
        assert graph["cfg_guider"]["inputs"]["model"][0] == "model_sampling_shift"
        assert graph["basic_scheduler"]["inputs"]["model"][0] == "model_sampling_shift"

    def test_flux_shift_is_resolution_independent(self):
        # Equal base and max shift make the node's area interpolation constant, so the requested
        # value is the shift the model runs with rather than one end of a range.
        graph = _flux_graph()
        insert_model_sampling_shift(graph, 2.5, node=FlowShiftNode.FLUX)
        inputs = graph["model_sampling_shift"]["inputs"]
        assert inputs["max_shift"] == 2.5
        assert inputs["base_shift"] == 2.5

    def test_the_shift_sits_on_top_of_the_lora_chain(self):
        graph = _flux_graph()
        insert_lora_chain(
            graph,
            [ResolvedLora(filename="a.safetensors", strength_model=1.0, strength_clip=1.0)],
            flux=True,
        )
        insert_model_sampling_shift(graph, 3.0, node=FlowShiftNode.FLUX)

        assert graph["model_sampling_shift"]["inputs"]["model"] == ["lora_0", 0]
        assert graph["cfg_guider"]["inputs"]["model"][0] == "model_sampling_shift"

    def test_aura_flow_gets_the_node_between_the_model_and_the_sampler(self):
        graph = _graph("qwen")
        model_source = graph["sampler"]["inputs"]["model"][0]

        insert_model_sampling_shift(graph, 4.0, node=FlowShiftNode.AURA_FLOW)

        assert graph["model_sampling_aura_flow"]["class_type"] == "ModelSamplingAuraFlow"
        assert graph["model_sampling_aura_flow"]["inputs"] == {"model": [model_source, 0], "shift": 4.0}
        assert graph["sampler"]["inputs"]["model"][0] == "model_sampling_aura_flow"
        assert "model_sampling_shift" not in graph

    def test_the_aura_flow_shift_sits_on_top_of_the_lora_chain(self):
        graph = _graph("qwen")
        insert_lora_chain(graph, [ResolvedLora(filename="a.safetensors", strength_model=1.0, strength_clip=1.0)])

        insert_model_sampling_shift(graph, 3.0, node=FlowShiftNode.AURA_FLOW)

        assert graph["model_sampling_aura_flow"]["inputs"]["model"] == ["lora_0", 0]
        assert graph["sampler"]["inputs"]["model"][0] == "model_sampling_aura_flow"

    def test_a_model_without_a_shift_is_left_alone(self):
        graph = _sd_graph()
        before = json.dumps(graph, default=str, sort_keys=True)
        insert_model_sampling_shift(graph, 2.0)
        assert json.dumps(graph, default=str, sort_keys=True) == before


class TestRewireImg2Img:
    def test_sd_sampler_fed_from_vae_encode(self):
        graph = _sd_graph()
        assert graph["sampler"]["inputs"]["latent_image"][0] != "vae_encode"
        rewire_img2img(graph)
        assert graph["sampler"]["inputs"]["latent_image"][0] == "vae_encode"

    def test_flux_sampler_fed_from_vae_encode(self):
        graph = _flux_graph()
        rewire_img2img(graph, flux=True)
        assert graph["sampler_custom_advanced"]["inputs"]["latent_image"][0] == "vae_encode"

    def test_missing_nodes_are_a_noop(self):
        graph = _graph("controlnet")  # has no vae_encode
        before = json.dumps(graph, default=str, sort_keys=True)
        rewire_img2img(graph)
        assert json.dumps(graph, default=str, sort_keys=True) == before

    def test_cascade_both_samplers_fed_from_stage_c_encode(self):
        graph = _graph("stable_cascade")
        rewire_cascade_img2img(graph)
        assert graph["sampler_stage_c"]["inputs"]["latent_image"][0] == "stablecascade_stagec_vaeencode"
        assert graph["sampler_stage_b"]["inputs"]["latent_image"][0] == "stablecascade_stagec_vaeencode"


class TestInsertRemixImageChain:
    def test_no_extras_is_a_noop(self):
        graph = _graph("stable_cascade_remix")
        before = json.dumps(graph, default=str, sort_keys=True)
        insert_remix_image_chain(graph, [])
        assert json.dumps(graph, default=str, sort_keys=True) == before

    def test_two_extra_images_chain(self):
        graph = _graph("stable_cascade_remix")
        images = [RemixImage(image=PIL.Image.new("RGB", (8, 8)), strength=s) for s in (0.5, 0.25)]
        insert_remix_image_chain(graph, images)

        # Extras start at index 1; index 0 is the primary source image
        assert graph["sc_image_loader_1"]["class_type"] == "HordeImageLoader"
        assert graph["clip_vision_encode_1"]["inputs"]["image"] == ["sc_image_loader_1", 0]
        assert graph["unclip_conditioning_1"]["inputs"]["conditioning"] == ["unclip_conditioning_0", 0]
        assert graph["unclip_conditioning_2"]["inputs"]["conditioning"] == ["unclip_conditioning_1", 0]
        assert graph["unclip_conditioning_1"]["inputs"]["strength"] == 0.5
        assert graph["unclip_conditioning_2"]["inputs"]["strength"] == 0.25
        # Only the last conditioning feeds the stage-C sampler
        assert graph["sampler_stage_c"]["inputs"]["positive"][0] == "unclip_conditioning_2"


class TestConfigureControlnet:
    def test_canny_params(self):
        graph = _graph("controlnet")
        params = configure_controlnet(
            graph,
            control_type="canny",
            image_is_control=False,
            return_control_map=False,
            width=768,
            height=512,
        )
        assert params["controlnet_model_loader.control_net_name"] == "diff_control_sd15_canny_fp16.safetensors"
        assert params["preprocessor.preprocessor"] == "CannyEdgePreprocessor"
        assert params["preprocessor.resolution"] == 512

    def test_canny_stays_on_the_all_in_one_preprocessor_node(self):
        graph = _graph("controlnet")
        configure_controlnet(
            graph,
            control_type="canny",
            image_is_control=False,
            return_control_map=False,
            width=512,
            height=512,
            seed=99,
        )
        assert graph["preprocessor"]["class_type"] == "AIO_Preprocessor"
        assert "seed" not in graph["preprocessor"]["inputs"]

    def test_shuffle_becomes_a_seeded_detector_node(self):
        # Shuffle is the one randomised detector: on the all-in-one node it would take the detector's
        # unseeded default and give a different picture for the same request every time.
        graph = _graph("controlnet")
        params = configure_controlnet(
            graph,
            control_type="shuffle",
            image_is_control=False,
            return_control_map=False,
            width=512,
            height=512,
            seed=123456789,
        )
        assert graph["preprocessor"]["class_type"] == SHUFFLE_PREPROCESSOR_CLASS_TYPE
        assert graph["preprocessor"]["inputs"]["seed"] == 123456790
        assert "preprocessor" not in graph["preprocessor"]["inputs"]
        assert "preprocessor.preprocessor" not in params
        assert params["controlnet_model_loader.control_net_name"] == "control_v11e_sd15_shuffle_fp16.safetensors"
        assert params["preprocessor.resolution"] == 512

    def test_shuffle_seed_zero_is_still_seeded(self):
        # The detector reads a seed of 0 as "no seed", so a request that asked for 0 must be shifted.
        graph = _graph("controlnet")
        configure_controlnet(
            graph,
            control_type="shuffle",
            image_is_control=False,
            return_control_map=False,
            width=512,
            height=512,
            seed=0,
        )
        assert graph["preprocessor"]["inputs"]["seed"] == 1

    def test_shuffle_with_premade_map_stays_on_the_all_in_one_node(self):
        graph = _graph("controlnet")
        params = configure_controlnet(
            graph,
            control_type="shuffle",
            image_is_control=True,
            return_control_map=False,
            width=512,
            height=512,
            seed=7,
        )
        assert graph["preprocessor"]["class_type"] == "AIO_Preprocessor"
        assert params["preprocessor.preprocessor"] == "none"

    def test_color_params(self):
        graph = _graph("controlnet")
        params = configure_controlnet(
            graph,
            control_type="color",
            image_is_control=False,
            return_control_map=False,
            width=512,
            height=512,
        )
        assert params["preprocessor.preprocessor"] == "ColorPreprocessor"
        assert params["controlnet_model_loader.control_net_name"] == "t2iadapter_color_sd14v1.pth"

    def test_image_is_control_passes_through(self):
        graph = _graph("controlnet")
        params = configure_controlnet(
            graph,
            control_type="canny",
            image_is_control=True,
            return_control_map=False,
            width=512,
            height=512,
        )
        assert params["preprocessor.preprocessor"] == "none"

    def test_return_control_map_rewires_output(self):
        graph = _graph("controlnet")
        configure_controlnet(
            graph,
            control_type="canny",
            image_is_control=False,
            return_control_map=True,
            width=512,
            height=512,
        )
        assert graph["output_image"]["inputs"]["images"][0] == "preprocessor"

    def test_openpose_runs_without_onnxruntime(self, monkeypatch: pytest.MonkeyPatch):
        # The exposed openpose preprocessor is the classic torch-only detector chain (DWPose is not a
        # horde control_type), so it configures on a lean base install with nothing importable.
        monkeypatch.setattr(importlib.util, "find_spec", _fake_find_spec(present=set()))
        graph = _graph("controlnet")
        params = configure_controlnet(
            graph,
            control_type="openpose",
            image_is_control=False,
            return_control_map=False,
            width=512,
            height=512,
        )
        assert params["preprocessor.preprocessor"] == "OpenposePreprocessor"

    def test_openpose_with_premade_map_passes_through(self, monkeypatch: pytest.MonkeyPatch):
        # image_is_control means the preprocessor is "none" and never runs.
        monkeypatch.setattr(importlib.util, "find_spec", _fake_find_spec(present=set()))
        graph = _graph("controlnet")
        params = configure_controlnet(
            graph,
            control_type="openpose",
            image_is_control=True,
            return_control_map=False,
            width=512,
            height=512,
        )
        assert params["preprocessor.preprocessor"] == "none"

    def test_openpose_configures_with_packages_present(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(importlib.util, "find_spec", _fake_find_spec(present={"onnxruntime"}))
        graph = _graph("controlnet")
        params = configure_controlnet(
            graph,
            control_type="openpose",
            image_is_control=False,
            return_control_map=False,
            width=512,
            height=512,
        )
        assert params["preprocessor.preprocessor"] == "OpenposePreprocessor"

    def test_canny_does_not_raise_when_onnxruntime_absent(self, monkeypatch: pytest.MonkeyPatch):
        # Canny is pure-cv2; it has no gated dependency and must run on a lean base install.
        monkeypatch.setattr(importlib.util, "find_spec", _fake_find_spec(present=set()))
        graph = _graph("controlnet")
        params = configure_controlnet(
            graph,
            control_type="canny",
            image_is_control=False,
            return_control_map=False,
            width=512,
            height=512,
        )
        assert params["preprocessor.preprocessor"] == "CannyEdgePreprocessor"


class TestApplyLayerdiffuse:
    def test_sd1_rewires_and_configures(self):
        graph = _sd_graph()
        params = apply_layerdiffuse(
            graph,
            baseline=KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_1,
            hires_fix=False,
        )
        assert params["model_loader.will_load_loras"] is True
        assert params["layer_diffuse_apply.config"] == "SD15, Attention Injection, attn_sharing"
        assert params["layer_diffuse_decode_rgba.sd_version"] == "SD15"
        assert graph["sampler"]["inputs"]["model"][0] == "layer_diffuse_apply"
        assert graph["layer_diffuse_apply"]["inputs"]["model"][0] == "model_loader"
        assert graph["output_image"]["inputs"]["images"][0] == "layer_diffuse_decode_rgba"
        assert graph["layer_diffuse_decode_rgba"]["inputs"]["images"][0] == "vae_decode"

    def test_sdxl_config(self):
        graph = _sd_graph()
        params = apply_layerdiffuse(
            graph,
            baseline=KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_xl,
            hires_fix=False,
        )
        assert params["layer_diffuse_apply.config"] == "SDXL, Conv Injection"
        assert params["layer_diffuse_decode_rgba.sd_version"] == "SDXL"

    def test_unsupported_baseline_leaves_graph_untouched(self):
        graph = _sd_graph()
        before = json.dumps(graph, default=str, sort_keys=True)
        params = apply_layerdiffuse(
            graph,
            baseline=KNOWN_IMAGE_GENERATION_BASELINE.stable_cascade,
            hires_fix=False,
        )
        # A transparent gen still implies lora loading, even when not rewired
        assert params == {"model_loader.will_load_loras": True}
        assert json.dumps(graph, default=str, sort_keys=True) == before


class TestHiresFixFirstPassResolution:
    def test_target_is_requested_resolution(self):
        params = hires_fix_first_pass_resolution(KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_1, 1024, 1024)
        assert params["latent_upscale.width"] == 1024
        assert params["latent_upscale.height"] == 1024
        # SD1 first pass shrinks to the 512-min strategy
        assert params["empty_latent_image.width"] == 512
        assert params["empty_latent_image.height"] == 512

    def test_sdxl_uses_resolution_buckets(self):
        params = hires_fix_first_pass_resolution(KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_xl, 2048, 2048)
        assert (params["empty_latent_image.width"], params["empty_latent_image.height"]) == (1024, 1024)

    def test_unknown_baseline_falls_back_to_1024(self):
        params = hires_fix_first_pass_resolution(None, 2048, 2048)
        assert (params["empty_latent_image.width"], params["empty_latent_image.height"]) == (1024, 1024)


class TestQrParams:
    def test_defaults(self):
        params = qr_params_from_extra_texts([], prompt="a prompt", width=512, height=768)
        assert params["qr_code_split.text"] == "https://haidra.net"
        assert params["qr_code_split.max_image_size"] == 768
        assert params["qr_code_split.protocol"] == "None"
        assert params["function_layer_prompt.text"] == "a prompt"

    def test_extra_texts_parsed(self):
        params = qr_params_from_extra_texts(
            [
                {"text": "https://example.com", "reference": "qr_code"},
                {"text": "https", "reference": "protocol"},
                {"text": "circle", "reference": "module_drawer"},
                {"text": "-5", "reference": "x_offset"},
                {"text": "200", "reference": "y_offset"},
                {"text": "3", "reference": "qr_border"},
            ],
            prompt="p",
            width=512,
            height=512,
        )
        assert params["qr_code_split.text"] == "https://example.com"
        assert params["qr_code_split.protocol"] == "Https"
        assert params["qr_code_split.module_drawer"] == "Circle"
        assert params["qr_flattened_composite.x"] == 10  # negative offsets clamp to 10
        assert params["qr_flattened_composite.y"] == 200
        assert params["qr_code_split.border"] == 3

    def test_layout_centers_on_64_multiple(self):
        params = qr_layout_params(width=1024, height=1024, qr_size=400)
        # Centered would be 312; rounded down to a multiple of 64 is 256
        assert params["qr_flattened_composite.x"] == 256
        assert params["mask_composite.x"] == 256
        assert params["module_layer_composite.y"] == params["qr_flattened_composite.y"]

    def test_layout_clamps_offsets_that_overflow(self):
        params = qr_layout_params(width=512, height=512, qr_size=400, x_offset=400, y_offset=50)
        assert params["qr_flattened_composite.x"] == (512 - 400) - 10
        assert params["qr_flattened_composite.y"] == 50
