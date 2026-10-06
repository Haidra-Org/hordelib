"""GPU-free tests for the post-processing pipeline family."""

import pytest
from PIL import Image

from hordelib.pipeline.context import PostProcessingContext
from hordelib.pipeline.families.post_processing import (
    IMAGE_FACEFIX_DEFINITION,
    IMAGE_UPSCALE_DEFINITION,
    POST_PROCESSING_REGISTRY,
    compose_post_processing_chain,
)
from hordelib.pipeline.payload_pp import (
    FacefixPayload,
    PostProcessorKind,
    StripBackgroundPayload,
    UpscalePayload,
    classify_post_processor,
    order_post_processing,
    post_processing_payload_from_horde_dict,
)


@pytest.fixture
def source_image() -> Image.Image:
    return Image.new("RGB", (64, 64), (128, 64, 32))


class TestClassification:
    @pytest.mark.parametrize(
        "name",
        ["RealESRGAN_x4plus", "RealESRGAN_x2plus", "RealESRGAN_x4plus_anime_6B", "NMKD_Siax", "4x_AnimeSharp"],
    )
    def test_upscalers(self, name: str) -> None:
        assert classify_post_processor(name) is PostProcessorKind.upscaler

    @pytest.mark.parametrize("name", ["GFPGAN", "CodeFormers"])
    def test_facefixers(self, name: str) -> None:
        assert classify_post_processor(name) is PostProcessorKind.facefixer

    def test_strip_background(self) -> None:
        assert classify_post_processor("strip_background") is PostProcessorKind.strip_background

    def test_unknown(self) -> None:
        assert classify_post_processor("not_a_post_processor") is None


class TestOrder:
    def test_facefixers_before_upscalers_before_strip_background(self) -> None:
        names = ["strip_background", "RealESRGAN_x4plus", "GFPGAN"]
        assert order_post_processing(names) == ["GFPGAN", "RealESRGAN_x4plus", "strip_background"]

    def test_groups_keep_caller_order(self) -> None:
        names = ["NMKD_Siax", "CodeFormers", "RealESRGAN_x2plus", "GFPGAN"]
        assert order_post_processing(names) == ["CodeFormers", "GFPGAN", "NMKD_Siax", "RealESRGAN_x2plus"]

    def test_unrecognized_names_are_kept_last(self) -> None:
        names = ["not_a_post_processor", "strip_background", "GFPGAN"]
        assert order_post_processing(names) == ["GFPGAN", "strip_background", "not_a_post_processor"]

    def test_input_is_not_mutated(self) -> None:
        names = ["RealESRGAN_x4plus", "GFPGAN"]
        order_post_processing(names)
        assert names == ["RealESRGAN_x4plus", "GFPGAN"]


class TestPayloadFromDict:
    def test_upscale_with_rescale(self, source_image: Image.Image) -> None:
        payload = post_processing_payload_from_horde_dict(
            {"model": "NMKD_Siax", "source_image": source_image, "width": 320, "height": 256},
        )
        assert isinstance(payload, UpscalePayload)
        assert payload.rescale_width == 320
        assert payload.rescale_height == 256

    def test_facefix_strength_maps_to_blend_not_fidelity(self, source_image: Image.Image) -> None:
        # facefixer_strength is the blend of the restored image over the input; CodeFormer's
        # fidelity is a different knob and is left at its default.
        payload = post_processing_payload_from_horde_dict(
            {"model": "CodeFormers", "source_image": source_image, "facefixer_strength": 0.9},
        )
        assert isinstance(payload, FacefixPayload)
        assert payload.strength == 0.9
        assert payload.fidelity == 0.5

    def test_facefix_strength_defaults_to_full_restoration(self, source_image: Image.Image) -> None:
        payload = post_processing_payload_from_horde_dict(
            {"model": "GFPGAN", "source_image": source_image},
        )
        assert isinstance(payload, FacefixPayload)
        assert payload.strength == 1.0

    def test_strip_background(self, source_image: Image.Image) -> None:
        payload = post_processing_payload_from_horde_dict(
            {"model": "strip_background", "source_image": source_image},
        )
        assert isinstance(payload, StripBackgroundPayload)

    def test_unknown_model_rejected(self, source_image: Image.Image) -> None:
        with pytest.raises(ValueError, match="Unknown post-processor"):
            post_processing_payload_from_horde_dict({"model": "bogus", "source_image": source_image})

    def test_missing_source_image_rejected(self) -> None:
        with pytest.raises(ValueError, match="source_image"):
            post_processing_payload_from_horde_dict({"model": "GFPGAN"})


class TestFidelityClamping:
    def test_clamps_out_of_range(self, source_image: Image.Image) -> None:
        assert FacefixPayload(model="CodeFormers", source_image=source_image, fidelity=2.0).fidelity == 1.0
        assert FacefixPayload(model="CodeFormers", source_image=source_image, fidelity=-1).fidelity == 0.0

    def test_coerces_garbage_to_default(self, source_image: Image.Image) -> None:
        assert (
            FacefixPayload(model="CodeFormers", source_image=source_image, fidelity="bogus").fidelity == 0.5  # type: ignore[arg-type]
        )


class TestStrengthClamping:
    def test_clamps_out_of_range(self, source_image: Image.Image) -> None:
        assert FacefixPayload(model="GFPGAN", source_image=source_image, strength=2.0).strength == 1.0
        assert FacefixPayload(model="GFPGAN", source_image=source_image, strength=-1).strength == 0.0

    def test_coerces_garbage_to_full_restoration(self, source_image: Image.Image) -> None:
        assert FacefixPayload(model="GFPGAN", source_image=source_image, strength="bogus").strength == 1.0  # type: ignore[arg-type]
        assert FacefixPayload(model="GFPGAN", source_image=source_image, strength=None).strength == 1.0  # type: ignore[arg-type]


class TestRegistrySelection:
    def test_upscale_selected(self, source_image: Image.Image) -> None:
        context = PostProcessingContext(model_name="NMKD_Siax", model_file="NMKD_Siax.pth")
        template = POST_PROCESSING_REGISTRY.select(
            UpscalePayload(model="NMKD_Siax", source_image=source_image),
            context,
        )
        assert template is IMAGE_UPSCALE_DEFINITION

    def test_facefix_selected(self, source_image: Image.Image) -> None:
        context = PostProcessingContext(model_name="GFPGAN", model_file="GFPGANv1.4.pth")
        template = POST_PROCESSING_REGISTRY.select(
            FacefixPayload(model="GFPGAN", source_image=source_image),
            context,
        )
        assert template is IMAGE_FACEFIX_DEFINITION


class TestMaterialization:
    def test_upscale_graph(self, source_image: Image.Image) -> None:
        payload = UpscalePayload(model="NMKD_Siax", source_image=source_image)
        context = PostProcessingContext(model_name="NMKD_Siax", model_file="NMKD_Siax.pth")
        graph = IMAGE_UPSCALE_DEFINITION.materialize(payload, context).to_api_dict()

        model_loader = next(n for n in graph.values() if n["_meta"]["title"] == "model_loader")
        assert model_loader["inputs"]["model_name"] == "NMKD_Siax.pth"
        # to_api_dict deep-copies, so the bound image is an equal copy rather than the same object
        image_loader = next(n for n in graph.values() if n["_meta"]["title"] == "image_loader")
        bound_image = image_loader["inputs"]["image"]
        assert isinstance(bound_image, Image.Image)
        assert bound_image.tobytes() == source_image.tobytes()
        # LoadImage must have been swapped for the PIL-accepting Horde node at load time
        assert image_loader["class_type"] == "HordeImageLoader"

    def test_facefix_graph_binds_fidelity(self, source_image: Image.Image) -> None:
        payload = FacefixPayload(model="CodeFormers", source_image=source_image, fidelity=0.7)
        context = PostProcessingContext(model_name="CodeFormers", model_file="codeformer.pth")
        graph = IMAGE_FACEFIX_DEFINITION.materialize(payload, context).to_api_dict()

        model_loader = next(n for n in graph.values() if n["_meta"]["title"] == "model_loader")
        assert model_loader["inputs"]["model_name"] == "codeformer.pth"
        restore = next(n for n in graph.values() if n["_meta"]["title"] == "face_restore_with_model")
        assert restore["inputs"]["codeformer_fidelity"] == 0.7

    def test_facefix_graph_binds_strength_to_blend(self, source_image: Image.Image) -> None:
        payload = FacefixPayload(model="GFPGAN", source_image=source_image, strength=0.25)
        context = PostProcessingContext(model_name="GFPGAN", model_file="GFPGANv1.4.pth")
        graph = IMAGE_FACEFIX_DEFINITION.materialize(payload, context).to_api_dict()

        blend = next(n for n in graph.values() if n["_meta"]["title"] == "facefix_blend")
        assert blend["inputs"]["blend_factor"] == 0.25
        assert blend["inputs"]["blend_mode"] == "normal"

    def test_default_strength_is_full_restoration(self, source_image: Image.Image) -> None:
        # An unset facefixer_strength must leave the blend a pass-through of the restored image.
        payload = FacefixPayload(model="GFPGAN", source_image=source_image)
        context = PostProcessingContext(model_name="GFPGAN", model_file="GFPGANv1.4.pth")
        graph = IMAGE_FACEFIX_DEFINITION.materialize(payload, context).to_api_dict()
        blend = next(n for n in graph.values() if n["_meta"]["title"] == "facefix_blend")
        assert blend["inputs"]["blend_factor"] == 1.0

    def test_default_fidelity_matches_legacy_graph_value(self, source_image: Image.Image) -> None:
        # The packaged graph hardcodes 0.5; the payload default must reproduce it exactly so
        # ported facefix output stays image-identical.
        payload = FacefixPayload(model="GFPGAN", source_image=source_image)
        context = PostProcessingContext(model_name="GFPGAN", model_file="GFPGANv1.4.pth")
        graph = IMAGE_FACEFIX_DEFINITION.materialize(payload, context).to_api_dict()
        restore = next(n for n in graph.values() if n["_meta"]["title"] == "face_restore_with_model")
        assert restore["inputs"]["codeformer_fidelity"] == 0.5


def _upscale(source_image: Image.Image, model_file: str) -> tuple[UpscalePayload, PostProcessingContext]:
    return (
        UpscalePayload(model="NMKD_Siax", source_image=source_image),
        PostProcessingContext(model_name="NMKD_Siax", model_file=model_file),
    )


def _facefix(source_image: Image.Image) -> tuple[FacefixPayload, PostProcessingContext]:
    return (
        FacefixPayload(model="CodeFormers", source_image=source_image, fidelity=0.7, strength=0.25),
        PostProcessingContext(model_name="CodeFormers", model_file="codeformer.pth"),
    )


def _of_class(graph: dict, class_type: str) -> list[str]:
    return [title for title, node in graph.items() if node["class_type"] == class_type]


def _links(graph: dict) -> dict[str, str]:
    """Every connected input as ``title.input`` -> source title."""
    return {
        f"{title}.{name}": value[0]
        for title, node in graph.items()
        for name, value in node["inputs"].items()
        if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str)
    }


def _cross_stage_links(graph: dict, index: int) -> dict[str, str]:
    prefix = f"stage{index}_"
    return {
        target: source
        for target, source in _links(graph).items()
        if target.startswith(prefix) and not source.startswith(prefix)
    }


def _assert_chain_shape(graph: dict, stage_count: int) -> None:
    assert len(_of_class(graph, "HordeImageLoader")) == 1
    assert len(_of_class(graph, "HordeImageOutput")) == 1
    assert all(source in graph for source in _links(graph).values())
    assert all(node["_meta"]["title"] == title for title, node in graph.items())
    assert _of_class(graph, "HordeImageLoader") == ["stage0_image_loader"]
    assert _of_class(graph, "HordeImageOutput") == [f"stage{stage_count - 1}_output_image"]
    assert not _cross_stage_links(graph, 0)
    for index in range(1, stage_count):
        # A later stage reads only the previous stage's result, never a model loader or an older stage.
        sources = set(_cross_stage_links(graph, index).values())
        assert len(sources) == 1
        (source,) = sources
        assert source.startswith(f"stage{index - 1}_")
        assert source != f"stage{index - 1}_model_loader"


class TestComposedChain:
    def test_empty_chain_rejected(self) -> None:
        with pytest.raises(ValueError):
            compose_post_processing_chain([])

    def test_single_upscale_is_the_materialized_definition(self, source_image: Image.Image) -> None:
        payload, context = _upscale(source_image, "NMKD_Siax.pth")
        chain = compose_post_processing_chain([(payload, context)])
        expected = IMAGE_UPSCALE_DEFINITION.materialize(payload, context).to_api_dict()
        # PIL images compare by mode, size and pixels, so the bound image takes part in the equality.
        assert chain.graph.to_api_dict() == expected
        assert [output.node for output in chain.outputs] == ["output_image"]

    def test_facefix_then_upscale(self, source_image: Image.Image) -> None:
        chain = compose_post_processing_chain([_facefix(source_image), _upscale(source_image, "4x_Siax.pth")])
        graph = chain.graph.to_api_dict()
        _assert_chain_shape(graph, 2)
        assert [output.node for output in chain.outputs] == ["stage1_output_image"]
        assert graph["stage0_model_loader"]["inputs"]["model_name"] == "codeformer.pth"
        assert graph["stage1_model_loader"]["inputs"]["model_name"] == "4x_Siax.pth"
        assert graph["stage0_face_restore_with_model"]["inputs"]["codeformer_fidelity"] == 0.7
        assert graph["stage0_facefix_blend"]["inputs"]["blend_factor"] == 0.25
        assert graph["stage0_face_restore_with_model"]["inputs"]["image"][0] == "stage0_image_loader"
        assert graph["stage0_facefix_blend"]["inputs"]["image1"][0] == "stage0_image_loader"
        assert set(_cross_stage_links(graph, 1).values()) == {"stage0_facefix_blend"}

    def test_two_upscalers(self, source_image: Image.Image) -> None:
        chain = compose_post_processing_chain(
            [_upscale(source_image, "first.pth"), _upscale(source_image, "second.pth")],
        )
        graph = chain.graph.to_api_dict()
        _assert_chain_shape(graph, 2)
        assert graph["stage0_model_loader"]["inputs"]["model_name"] == "first.pth"
        assert graph["stage1_model_loader"]["inputs"]["model_name"] == "second.pth"

    def test_upscale_then_facefix_rewires_both_loader_consumers(self, source_image: Image.Image) -> None:
        chain = compose_post_processing_chain([_upscale(source_image, "4x.pth"), _facefix(source_image)])
        graph = chain.graph.to_api_dict()
        _assert_chain_shape(graph, 2)
        assert set(_cross_stage_links(graph, 1)) == {
            "stage1_face_restore_with_model.image",
            "stage1_facefix_blend.image1",
        }

    def test_facefix_upscale_upscale(self, source_image: Image.Image) -> None:
        chain = compose_post_processing_chain(
            [_facefix(source_image), _upscale(source_image, "a.pth"), _upscale(source_image, "b.pth")],
        )
        graph = chain.graph.to_api_dict()
        _assert_chain_shape(graph, 3)
        assert [graph[f"stage{i}_model_loader"]["inputs"]["model_name"] for i in range(3)] == [
            "codeformer.pth",
            "a.pth",
            "b.pth",
        ]
        assert graph["stage0_facefix_blend"]["inputs"]["blend_factor"] == 0.25
