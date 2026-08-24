"""Audit: hordelib's control-type tables must cover exactly what the API accepts. No GPU required.

A control type the horde API accepts but hordelib has no preprocessor or model for is not rejected:
the payload clamps it away and the request renders as plain image-to-image, so the requester pays for
a picture that promised guidance and silently got none. Keeping the tables in lockstep with
``KNOWN_IMAGE_CONTROLNETS`` turns that into a test failure at the moment a control type is added on
either side.
"""

from horde_sdk.generation_parameters.image.consts import KNOWN_IMAGE_CONTROLNETS

from hordelib.pipeline.constants import (
    CONTROLNET_IMAGE_PREPROCESSOR_MAP,
    CONTROLNET_MODEL_MAP,
)


def test_preprocessor_map_covers_exactly_the_known_control_types() -> None:
    assert set(CONTROLNET_IMAGE_PREPROCESSOR_MAP) == set(KNOWN_IMAGE_CONTROLNETS)


def test_model_map_covers_exactly_the_known_control_types() -> None:
    assert set(CONTROLNET_MODEL_MAP) == set(KNOWN_IMAGE_CONTROLNETS)


def test_every_control_type_has_both_a_preprocessor_and_a_model() -> None:
    """A control type with only half a pair configures a graph that cannot guide the generation."""
    assert set(CONTROLNET_IMAGE_PREPROCESSOR_MAP) <= set(CONTROLNET_MODEL_MAP)
