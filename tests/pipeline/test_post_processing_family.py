"""GPU-free tests for the post-processing pipeline family."""

import pytest
from PIL import Image

from hordelib.pipeline.context import PostProcessingContext
from hordelib.pipeline.families.post_processing import (
    IMAGE_FACEFIX_DEFINITION,
    IMAGE_UPSCALE_DEFINITION,
    POST_PROCESSING_REGISTRY,
)
from hordelib.pipeline.payload_pp import (
    FacefixPayload,
    PostProcessorKind,
    StripBackgroundPayload,
    UpscalePayload,
    classify_post_processor,
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


class TestPayloadFromDict:
    def test_upscale_with_rescale(self, source_image: Image.Image) -> None:
        payload = post_processing_payload_from_horde_dict(
            {"model": "NMKD_Siax", "source_image": source_image, "width": 320, "height": 256},
        )
        assert isinstance(payload, UpscalePayload)
        assert payload.rescale_width == 320
        assert payload.rescale_height == 256

    def test_facefix_ignores_facefixer_strength(self, source_image: Image.Image) -> None:
        # The legacy path never wired facefixer_strength to the graph; the dict surface
        # preserves that so existing callers keep getting identical images.
        payload = post_processing_payload_from_horde_dict(
            {"model": "CodeFormers", "source_image": source_image, "facefixer_strength": 0.9},
        )
        assert isinstance(payload, FacefixPayload)
        assert payload.fidelity == 0.5

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

    def test_default_fidelity_matches_legacy_graph_value(self, source_image: Image.Image) -> None:
        # The packaged graph hardcodes 0.5; the payload default must reproduce it exactly so
        # ported facefix output stays image-identical.
        payload = FacefixPayload(model="GFPGAN", source_image=source_image)
        context = PostProcessingContext(model_name="GFPGAN", model_file="GFPGANv1.4.pth")
        graph = IMAGE_FACEFIX_DEFINITION.materialize(payload, context).to_api_dict()
        restore = next(n for n in graph.values() if n["_meta"]["title"] == "face_restore_with_model")
        assert restore["inputs"]["codeformer_fidelity"] == 0.5
