"""The ImagePipeline vocabulary and explicit pipeline resolution. No GPU required."""

from horde_model_reference.meta_consts import KNOWN_IMAGE_GENERATION_BASELINE

from hordelib.horde import HordeLib
from hordelib.pipeline.context import ModelContext
from hordelib.pipeline.families.image_gen import IMAGE_PIPELINES
from hordelib.pipeline.identifiers import AUTO_PIPELINE, AutoPipeline, ImagePipeline
from hordelib.pipeline.payload import ImageGenPayload

SD1 = KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_1


def test_enum_matches_registered_pipelines() -> None:
    """Every registered pipeline has an enum member and vice versa (mirrors the import-time audit)."""
    registered_names = {definition.name for definition in IMAGE_PIPELINES}
    enumerated_names = {member.value for member in ImagePipeline}
    assert registered_names == enumerated_names


def test_auto_sentinel_is_not_an_image_pipeline() -> None:
    """The AUTO sentinel is a deliberate, visibly different choice from naming a pipeline."""
    assert isinstance(AUTO_PIPELINE, AutoPipeline)
    assert not isinstance(AUTO_PIPELINE, ImagePipeline)
    assert AUTO_PIPELINE.value not in {member.value for member in ImagePipeline}


def test_auto_resolution_matches_registry_selection() -> None:
    """AUTO resolves to the same definition the registry selects."""
    payload = ImageGenPayload.from_horde_dict({"hires_fix": True})
    context = ModelContext(horde_model_name="some model", baseline=SD1)

    definition = HordeLib._resolve_pipeline_definition(payload, context, AUTO_PIPELINE)

    assert definition.name == ImagePipeline.STABLE_DIFFUSION_HIRES_FIX.value


def test_explicit_pipeline_is_honored() -> None:
    """Naming a pipeline bypasses registry selection entirely."""
    payload = ImageGenPayload.from_horde_dict({"hires_fix": True})
    context = ModelContext(horde_model_name="some model", baseline=SD1)

    definition = HordeLib._resolve_pipeline_definition(payload, context, ImagePipeline.STABLE_DIFFUSION_HIRES_FIX)

    assert definition.name == ImagePipeline.STABLE_DIFFUSION_HIRES_FIX.value


def test_explicit_pipeline_mismatch_warns_but_proceeds(caplog) -> None:
    """An explicit choice whose selector would not match is trusted, with a logged warning."""
    from loguru import logger

    handler_id = logger.add(caplog.handler, level="WARNING", format="{message}")
    try:
        # A plain payload has no controlnet feature, so the controlnet selector cannot match.
        payload = ImageGenPayload.from_horde_dict({})
        context = ModelContext(horde_model_name="some model", baseline=SD1)

        definition = HordeLib._resolve_pipeline_definition(payload, context, ImagePipeline.CONTROLNET)
    finally:
        logger.remove(handler_id)

    assert definition.name == ImagePipeline.CONTROLNET.value
    assert any("Explicit pipeline does not match" in record.message for record in caplog.records)


def test_every_enum_member_resolves_explicitly() -> None:
    """Every ImagePipeline member resolves to its registered definition when named explicitly."""
    payload = ImageGenPayload.from_horde_dict({})
    context = ModelContext(horde_model_name="some model", baseline=SD1)

    for member in ImagePipeline:
        definition = HordeLib._resolve_pipeline_definition(payload, context, member)
        assert definition.name == member.value
