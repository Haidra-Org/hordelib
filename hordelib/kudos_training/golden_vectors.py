"""The golden-vector fixture and the generator that produces it.

The fixture is the drift protection on the whole pricing path. Its encoder section pins every
payload's exact float32 vector under the shipped manifest, and its pricing section pins the ledger
arithmetic that turns a prediction into a price. hordelib's CI evaluates both, and the same file is
copied into the AI-Horde tree so its CI evaluates the server-side path against the same numbers:
any port, refactor or dependency bump that changes a price fails a test instead of a payout.

Every expected number in the fixture is computed here rather than typed, so regenerating after a
deliberate manifest or ledger revision is a command rather than an editing exercise:

    python -m hordelib.kudos_training.golden_vectors tests/kudos_golden_vectors_v22.json

The pricing section's predicted seconds are stated inputs, not model outputs. Pinning the
composition against a fixed prediction keeps the fixture meaningful across retraining, which is
exactly when the ledger most needs to be shown unchanged.
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from hordelib.kudos_training.ledger import (
    KudosPolicyLedger,
    PayloadFeatures,
    PredictedSeconds,
    PriceBreakdown,
    PricingBasis,
    compose_user_price,
    default_ledger,
)
from hordelib.kudos_training.manifest import KudosFeatureManifest, default_manifest

FIXTURE_BASIS: Final[PricingBasis] = PricingBasis(basis_seconds=10.0)
"""The seconds-to-kudos anchor the pricing section composes under.

A round ten seconds for the basis job is a fixture convention rather than a measurement: the
composition is proportional in the basis, so a stated anchor keeps the pinned prices readable and
independent of whichever model revision is current.
"""


@dataclass(frozen=True)
class EncoderCase:
    """Represents one payload whose exact encoding the fixture pins."""

    name: str
    """Case identifier, referenced by the coverage assertions in the encoder suite."""

    note: str
    """What the case is here to catch."""

    payload: dict[str, Any]
    """The payload as a caller would submit it."""


@dataclass(frozen=True)
class PricingCase:
    """Represents one composition whose exact price breakdown the fixture pins."""

    name: str
    """Case identifier."""

    note: str
    """Which line item or interaction the case is here to catch."""

    predicted_seconds: PredictedSeconds
    """Stated per-resource durations standing in for a model prediction."""

    payload_features: PayloadFeatures
    """The request-side fields the ledger reads."""


ENCODER_CASES: Final[tuple[EncoderCase, ...]] = (
    EncoderCase(
        name="basis_payload",
        note="The manifest's own basis job, which must encode with nothing collapsed or dropped.",
        payload={
            "width": 512,
            "height": 512,
            "steps": 50,
            "cfg_scale": 7.5,
            "denoising_strength": 1.0,
            "control_strength": 1.0,
            "scheduler": "karras",
            "hires_fix": False,
            "source_image": False,
            "source_mask": False,
            "source_processing": "txt2img",
            "sampler_name": "k_euler",
            "control_type": "None",
            "baseline": "stable_diffusion_1",
            "post_processing": [],
            "n_images": 1,
            "loras_count": 0,
            "tis_count": 0,
            "queue_depth_at_dispatch": 0,
        },
    ),
    EncoderCase(
        name="v21_payload_example",
        note=(
            "The v21 payload shape verbatim (hordelib/train.py:87-102): ddim_steps rather than steps, a "
            "karras boolean the v22 manifest does not read, and no scheduler, baseline, batch or count "
            "fields."
        ),
        payload={
            "height": 576,
            "width": 1024,
            "ddim_steps": 35,
            "cfg_scale": 9.0,
            "denoising_strength": 0.75,
            "control_strength": 1.0,
            "karras": True,
            "hires_fix": False,
            "source_image": False,
            "source_mask": False,
            "source_processing": "txt2img",
            "sampler_name": "k_dpm_2_a",
            "control_type": "canny",
            "post_processing": ["RealESRGAN_x4plus", "CodeFormers"],
        },
    ),
    EncoderCase(
        name="unknown_sampler_and_post_processor",
        note=(
            "An unrecognised sampler collapses onto k_euler and is reported; an unrecognised "
            "post-processor is dropped and counted rather than collapsed."
        ),
        payload={
            "height": 1024,
            "width": 1024,
            "steps": 20,
            "cfg_scale": 5.0,
            "sampler_name": "k_brand_new_solver",
            "scheduler": "a_new_schedule",
            "baseline": "some_future_family",
            "post_processing": ["RealESRGAN_x2plus", "not_a_post_processor"],
            "n_images": 1,
        },
    ),
    EncoderCase(
        name="remix_collapses_to_img2img",
        note="source_processing=remix is aliased onto img2img, so no collapse is reported for it.",
        payload={
            "height": 1024,
            "width": 1024,
            "steps": 25,
            "cfg_scale": 4.0,
            "denoising_strength": 0.6,
            "source_image": True,
            "source_processing": "remix",
            "sampler_name": "k_dpmpp_2m",
            "scheduler": "karras",
            "baseline": "stable_diffusion_xl",
            "n_images": 2,
        },
    ),
    EncoderCase(
        name="missing_optional_fields",
        note=(
            "Only the two required dimensions are supplied; every other feature takes its manifest "
            "default, including control_strength through its denoising_strength fallback."
        ),
        payload={"height": 512, "width": 512},
    ),
    EncoderCase(
        name="karras_false_encodes_normal",
        note=(
            "A live payload names no scheduler and carries karras=false, which resolves to the normal "
            "schedule rather than to the feature's karras default."
        ),
        payload={
            "height": 512,
            "width": 512,
            "steps": 20,
            "cfg_scale": 7.0,
            "karras": False,
            "sampler_name": "k_euler",
            "baseline": "stable_diffusion_1",
        },
    ),
    EncoderCase(
        name="named_scheduler_overrides_karras",
        note="A named scheduler wins over the legacy boolean, even when the two disagree.",
        payload={
            "height": 512,
            "width": 512,
            "steps": 20,
            "cfg_scale": 7.0,
            "karras": True,
            "scheduler": "sgm_uniform",
            "sampler_name": "k_euler",
            "baseline": "stable_diffusion_1",
        },
    ),
    EncoderCase(
        name="five_loras_batch_eight",
        note="The heaviest routine shape: a full lora stack, a batch of eight, and a chained upscale.",
        payload={
            "height": 1024,
            "width": 1536,
            "steps": 30,
            "cfg_scale": 7.0,
            "denoising_strength": 1.0,
            "hires_fix": True,
            "sampler_name": "k_euler_a",
            "scheduler": "simple",
            "baseline": "stable_diffusion_xl",
            "source_processing": "txt2img",
            "control_type": "None",
            "post_processing": ["RealESRGAN_x4plus", "GFPGAN"],
            "n_images": 8,
            "loras_count": 5,
            "tis_count": 2,
            "queue_depth_at_dispatch": 3,
        },
    ),
    EncoderCase(
        name="unified_control_type_shuffle",
        note=(
            "A control type the unified preprocessor set added: it holds its own slot rather than "
            "collapsing onto None, which is what priced it as an uncontrolled job before."
        ),
        payload={
            "height": 768,
            "width": 768,
            "steps": 25,
            "cfg_scale": 6.0,
            "denoising_strength": 0.8,
            "control_strength": 0.9,
            "source_image": True,
            "source_processing": "img2img",
            "sampler_name": "k_dpmpp_2m",
            "scheduler": "karras",
            "baseline": "stable_diffusion_xl",
            "control_type": "shuffle",
            "n_images": 1,
        },
    ),
)
"""The payloads the encoder section pins, in fixture order."""


PRICING_CASES: Final[tuple[PricingCase, ...]] = (
    PricingCase(
        name="basis_job_at_par",
        note="The basis job on a par baseline is worth the basis kudos exactly, by construction.",
        predicted_seconds=PredictedSeconds(sampler_window=10.0),
        payload_features=PayloadFeatures(baseline="stable_diffusion_1"),
    ),
    PricingCase(
        name="sdxl_capability_premium",
        note="The ported 2x for stable_diffusion_xl multiplies the whole measured subtotal.",
        predicted_seconds=PredictedSeconds(sampler_window=18.0),
        payload_features=PayloadFeatures(baseline="stable_diffusion_xl"),
    ),
    PricingCase(
        name="stable_cascade_capability_premium",
        note="The ported 4x for stable_cascade, the heaviest residual that is not marked provisional.",
        predicted_seconds=PredictedSeconds(sampler_window=22.5),
        payload_features=PayloadFeatures(baseline="stable_cascade"),
    ),
    PricingCase(
        name="flux_provisional_residual",
        note="A provisional heavy-model residual prices exactly like a settled one; the flag is review-only.",
        predicted_seconds=PredictedSeconds(sampler_window=40.0),
        payload_features=PayloadFeatures(baseline="flux_1"),
    ),
    PricingCase(
        name="qwen_heaviest_residual",
        note="The largest ported multiplier the catalog carries.",
        predicted_seconds=PredictedSeconds(sampler_window=55.0),
        payload_features=PayloadFeatures(baseline="qwen_image"),
    ),
    PricingCase(
        name="lora_and_ti_adders",
        note=(
            "The lora adder scales with the count while the textual-inversion adder is paid once, and "
            "both sit inside the premium rather than beside it."
        ),
        predicted_seconds=PredictedSeconds(sampler_window=12.0),
        payload_features=PayloadFeatures(baseline="stable_diffusion_1", loras_count=3, tis_count=2),
    ),
    PricingCase(
        name="post_processing_lane_weight",
        note="A post-processing-lane second is priced at the lane weight, not at the sampler weight.",
        predicted_seconds=PredictedSeconds(sampler_window=9.0, pp_lane=4.0),
        payload_features=PayloadFeatures(baseline="stable_diffusion_1"),
    ),
    PricingCase(
        name="unmeasured_model_surcharges_nothing",
        note=(
            "A model with no measured churn entry adds no surcharge, so an unpopulated map cannot "
            "quietly charge a guess."
        ),
        predicted_seconds=PredictedSeconds(sampler_window=15.0),
        payload_features=PayloadFeatures(
            baseline="stable_diffusion_xl",
            model_name="a model the corpus has not measured",
            loras_count=1,
        ),
    ),
)
"""The compositions the pricing section pins, in fixture order."""


@dataclass(frozen=True)
class GoldenDocument:
    """Represents the fixture document, so a caller can inspect it before it is written."""

    content: dict[str, Any]
    """The document as it is serialized."""

    encoder_case_count: int
    """How many encoder cases the document carries."""

    pricing_case_count: int
    """How many pricing cases the document carries."""


def build_golden_document(
    manifest: KudosFeatureManifest | None = None,
    ledger: KudosPolicyLedger | None = None,
    basis: PricingBasis = FIXTURE_BASIS,
) -> GoldenDocument:
    """Build the golden-vector fixture from the checked-in cases.

    Args:
        manifest: Feature manifest to encode against. Defaults to the shipped revision.
        ledger: Policy ledger to compose under. Defaults to the shipped revision.
        basis: Seconds-to-kudos anchor for the pricing section.

    Returns:
        The document and its case counts.
    """
    active_manifest = manifest if manifest is not None else default_manifest()
    active_ledger = ledger if ledger is not None else default_ledger()

    encoder_cases = []
    for case in ENCODER_CASES:
        result = active_manifest.encode(case.payload)
        encoder_cases.append(
            {
                "name": case.name,
                "note": case.note,
                "payload": case.payload,
                "expected_vector": [float(slot) for slot in result.vector],
                "expected_collapsed": result.collapsed,
                "expected_dropped_unknown": result.dropped_unknown,
            },
        )

    pricing_cases = []
    for pricing_case in PRICING_CASES:
        breakdown = compose_user_price(
            pricing_case.predicted_seconds,
            pricing_case.payload_features,
            active_ledger,
            basis,
        )
        pricing_cases.append(
            {
                "name": pricing_case.name,
                "note": pricing_case.note,
                "predicted_seconds": {
                    "sampler_window": pricing_case.predicted_seconds.sampler_window,
                    "pp_lane": pricing_case.predicted_seconds.pp_lane,
                },
                "payload_features": {
                    "baseline": pricing_case.payload_features.baseline,
                    "model_name": pricing_case.payload_features.model_name,
                    "loras_count": pricing_case.payload_features.loras_count,
                    "tis_count": pricing_case.payload_features.tis_count,
                },
                "expected_user_price": _breakdown_to_document(breakdown),
            },
        )

    content = {
        "manifest_version": active_manifest.manifest_version,
        "sampler_semantics": active_manifest.sampler_semantics.model_dump(mode="json"),
        "vector_length": active_manifest.vector_length(),
        "slot_names": list(active_manifest.slot_names()),
        "cases": encoder_cases,
        "pricing": {
            "ledger_version": active_ledger.ledger_version,
            "reference_machine": active_ledger.reference_machine,
            "basis": {"basis_seconds": basis.basis_seconds, "basis_kudos": basis.basis_kudos},
            "note": (
                "Predicted seconds are stated inputs rather than model outputs, so this section pins "
                "the ledger arithmetic that composes a price from a prediction and survives "
                "retraining. The horde-funded served-variety term is absent because it depends on "
                "assignment history the server holds and is tested there."
            ),
            "cases": pricing_cases,
        },
    }
    return GoldenDocument(
        content=content,
        encoder_case_count=len(encoder_cases),
        pricing_case_count=len(pricing_cases),
    )


def write_golden_document(
    path: str | Path,
    manifest: KudosFeatureManifest | None = None,
    ledger: KudosPolicyLedger | None = None,
    basis: PricingBasis = FIXTURE_BASIS,
) -> GoldenDocument:
    """Build the fixture and write it to *path*.

    Args:
        path: Destination fixture file, overwritten in place.
        manifest: Feature manifest to encode against. Defaults to the shipped revision.
        ledger: Policy ledger to compose under. Defaults to the shipped revision.
        basis: Seconds-to-kudos anchor for the pricing section.

    Returns:
        The document that was written.
    """
    document = build_golden_document(manifest=manifest, ledger=ledger, basis=basis)
    Path(path).write_text(json.dumps(document.content, indent=2) + "\n", encoding="utf-8")
    return document


def _breakdown_to_document(breakdown: PriceBreakdown) -> dict[str, float]:
    """Convert a composed price into the fixture's per-line-item mapping."""
    return {
        "sampler_seconds_kudos": breakdown.sampler_seconds_kudos,
        "pp_lane_seconds_kudos": breakdown.pp_lane_seconds_kudos,
        "amortized_model_surcharge_kudos": breakdown.amortized_model_surcharge_kudos,
        "lora_wait_kudos": breakdown.lora_wait_kudos,
        "ti_kudos": breakdown.ti_kudos,
        "measured_subtotal_kudos": breakdown.measured_subtotal_kudos,
        "capability_premium": breakdown.capability_premium,
        "quality_premium": breakdown.quality_premium,
        "total_kudos": breakdown.total_kudos,
    }


def main(argv: list[str] | None = None) -> int:
    """Regenerate the golden-vector fixture at the path given on the command line.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    arguments = sys.argv[1:] if argv is None else argv
    if len(arguments) != 1:
        print("usage: python -m hordelib.kudos_training.golden_vectors <fixture.json>", file=sys.stderr)
        return 2

    destination = Path(arguments[0])
    document = write_golden_document(destination)
    print(
        f"wrote {destination}: {document.encoder_case_count} encoder cases, "
        f"{document.pricing_case_count} pricing cases",
    )
    return 0


__all__ = [
    "ENCODER_CASES",
    "FIXTURE_BASIS",
    "PRICING_CASES",
    "EncoderCase",
    "GoldenDocument",
    "PricingCase",
    "build_golden_document",
    "main",
    "write_golden_document",
]


if __name__ == "__main__":
    sys.exit(main())
