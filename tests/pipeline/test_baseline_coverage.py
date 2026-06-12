"""CI tripwire: every known image-generation baseline must be consciously classified.

When horde_model_reference adds a new baseline enum value, this test fails until someone
either gives the baseline a pipeline family (and records the expected template here) or
explicitly lists it as unsupported / riding the generic SD path. See
``docs/adding-a-baseline.md`` for the full checklist.
"""

import pytest
from horde_model_reference.meta_consts import KNOWN_IMAGE_GENERATION_BASELINE

from hordelib.pipeline.context import ModelContext
from hordelib.pipeline.families.image import DEFAULT_REGISTRY
from hordelib.pipeline.payload import ImageGenPayload

GENERIC_SD_TEMPLATE = "stable_diffusion"

BASELINE_EXPECTED_TEMPLATE: dict[KNOWN_IMAGE_GENERATION_BASELINE, str] = {
    # Baselines served by the generic SD family (no dedicated template needed).
    KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_1: GENERIC_SD_TEMPLATE,
    KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_2_768: GENERIC_SD_TEMPLATE,
    KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_2_512: GENERIC_SD_TEMPLATE,
    KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_xl: GENERIC_SD_TEMPLATE,
    # Baselines with dedicated families.
    KNOWN_IMAGE_GENERATION_BASELINE.stable_cascade: "stable_cascade",
    KNOWN_IMAGE_GENERATION_BASELINE.flux_1: "flux",
    KNOWN_IMAGE_GENERATION_BASELINE.flux_schnell: "flux",
    KNOWN_IMAGE_GENERATION_BASELINE.flux_dev: "flux",
    KNOWN_IMAGE_GENERATION_BASELINE.qwen_image: "qwen",
}
"""The template a canonical txt2img payload must select for each supported baseline."""

UNSUPPORTED_BASELINES: set[KNOWN_IMAGE_GENERATION_BASELINE] = {
    KNOWN_IMAGE_GENERATION_BASELINE.infer,
    # z_image_turbo appeared in the model reference 2026-06; no hordelib family exists yet.
    KNOWN_IMAGE_GENERATION_BASELINE.z_image_turbo,
}
"""Baselines consciously not (yet) given a pipeline family.

A baseline here falls through to the generic SD template at selection time, which is
almost certainly wrong for a genuinely new architecture — supporting it means a new
``families/<name>.py`` per docs/adding-a-baseline.md, then moving it to the dict above.
"""


def _canonical_payload() -> ImageGenPayload:
    return ImageGenPayload.from_horde_dict(
        {
            "prompt": "a test prompt",
            "width": 512,
            "height": 512,
            "ddim_steps": 10,
            "seed": "1",
            "model": "test model",
        },
    )


def test_every_baseline_is_classified() -> None:
    """Adding a baseline to horde_model_reference must fail CI until it is classified here."""
    classified = set(BASELINE_EXPECTED_TEMPLATE) | UNSUPPORTED_BASELINES
    all_baselines = set(KNOWN_IMAGE_GENERATION_BASELINE)

    unclassified = all_baselines - classified
    assert not unclassified, (
        f"New baseline(s) {sorted(b.value for b in unclassified)} are not classified. "
        "Either add a pipeline family (docs/adding-a-baseline.md) and record its template in "
        "BASELINE_EXPECTED_TEMPLATE, or consciously add the baseline to UNSUPPORTED_BASELINES."
    )

    stale = classified - all_baselines
    assert not stale, f"Classified baseline(s) {stale} no longer exist in horde_model_reference."

    overlap = set(BASELINE_EXPECTED_TEMPLATE) & UNSUPPORTED_BASELINES
    assert not overlap, f"Baseline(s) {overlap} are both supported and unsupported."


@pytest.mark.parametrize(
    "baseline",
    sorted(BASELINE_EXPECTED_TEMPLATE, key=lambda b: b.value),
    ids=lambda b: b.value,
)
def test_supported_baseline_selects_expected_template(baseline: KNOWN_IMAGE_GENERATION_BASELINE) -> None:
    """A canonical txt2img payload for each supported baseline selects its family's template."""
    context = ModelContext(horde_model_name="test model", baseline=baseline)
    template = DEFAULT_REGISTRY.select(_canonical_payload(), context)

    assert template is not None
    assert template.name == BASELINE_EXPECTED_TEMPLATE[baseline]
