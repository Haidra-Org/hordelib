"""The policy ledger must load strictly and compose the same price forever.

The ledger carries every cost the trained model does not predict, so a typo that silently defaults,
a baseline the file forgot, or a rearranged formula all change what jobs pay without changing a
single weight. The loader tests pin the strictness; the composition tests pin each line item's
contribution; the fixture test pins the whole composition against the golden vectors, which the
AI-Horde tree evaluates against the same file.

CPU-only: nothing here loads a model or touches a GPU.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from hordelib.kudos_training.golden_vectors import FIXTURE_BASIS, build_golden_document
from hordelib.kudos_training.ledger import (
    DEFAULT_BASIS_KUDOS,
    DEFAULT_LEDGER_PATH,
    KudosPolicyLedger,
    LedgerSchemaError,
    LedgerUnit,
    PayloadFeatures,
    PredictedSeconds,
    PricingBasis,
    UnknownBaselineError,
    compose_user_price,
    compose_worker_reward,
    default_ledger,
    load_ledger,
)

GOLDEN_VECTORS_PATH = Path(__file__).parent / "kudos_golden_vectors_v22.json"

PORTED_BASELINE_MULTIPLIERS = {
    "infer": 1.0,
    "stable_diffusion_1": 1.0,
    "stable_diffusion_2_512": 1.0,
    "stable_diffusion_2_768": 1.0,
    "stable_diffusion_xl": 2.0,
    "stable_cascade": 4.0,
    "flux_1": 8.0,
    "flux_dev": 8.0,
    "flux_schnell": 8.0,
    "z_image_turbo": 8.0,
    "krea2_turbo": 8.0,
    "anima": 8.0,
    "qwen_image": 12.0,
}
"""What the AI-Horde server charges per baseline today, from the served baseline catalog.

The ledger's premiums must reproduce these exactly: v1 is a port, so any difference is a repricing
that was never decided.
"""

PROVISIONAL_BASELINES = frozenset(
    {"flux_1", "flux_dev", "flux_schnell", "z_image_turbo", "krea2_turbo", "anima", "qwen_image"},
)
"""Heavy families whose residual premium awaits measurement on hardware that can hold them."""


@pytest.fixture(scope="module")
def ledger() -> KudosPolicyLedger:
    return default_ledger()


@pytest.fixture(scope="module")
def basis() -> PricingBasis:
    return PricingBasis(basis_seconds=10.0)


@pytest.fixture()
def ledger_document() -> dict[str, Any]:
    return json.loads(DEFAULT_LEDGER_PATH.read_text(encoding="utf-8"))


def _write(tmp_path: Path, document: dict[str, Any]) -> Path:
    path = tmp_path / "ledger.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def test_shipped_ledger_identity(ledger: KudosPolicyLedger) -> None:
    assert ledger.ledger_version == "v1"
    assert ledger.manifest_version == "v22"
    assert ledger.reference_machine == "tazlin-4070tis"


def test_shipped_ledger_ports_the_server_baseline_multipliers(ledger: KudosPolicyLedger) -> None:
    """Every family's capability times quality premium must equal what the server charges today."""
    assert set(ledger.priced_baselines()) == set(PORTED_BASELINE_MULTIPLIERS)
    for baseline, ported in PORTED_BASELINE_MULTIPLIERS.items():
        capability = ledger.capability_premium_for(baseline).value
        quality = ledger.quality_premium_for(baseline).value
        measured = ledger.measured_time_component_for(baseline).value
        assert capability * quality * measured == pytest.approx(ported), baseline


def test_measured_time_component_starts_at_par(ledger: KudosPolicyLedger) -> None:
    """v1 attributes no share of any ported multiplier to measured time, so the residual is the whole."""
    for baseline in ledger.priced_baselines():
        assert ledger.measured_time_component_for(baseline).value == 1.0, baseline


def test_heavy_family_residuals_are_marked_provisional(ledger: KudosPolicyLedger) -> None:
    for baseline in ledger.priced_baselines():
        capability = ledger.capability_premium_for(baseline)
        assert capability.provisional is (baseline in PROVISIONAL_BASELINES), baseline
        if capability.provisional:
            assert "4090" in capability.provenance, baseline


def test_every_line_item_carries_provenance(ledger: KudosPolicyLedger) -> None:
    items = [
        ledger.user_price.resource_weights.sampler_second,
        ledger.user_price.resource_weights.pp_lane_second,
        ledger.user_price.lora_wait_kudos,
        ledger.user_price.ti_kudos,
        ledger.worker_reward.served_variety_kudos_per_assigned_swap,
        *ledger.user_price.capability_premium.values(),
        *ledger.user_price.quality_premium.values(),
        *ledger.user_price.measured_time_component.values(),
    ]
    for item in items:
        assert item.provenance.strip()


def test_ported_adders_and_weights(ledger: KudosPolicyLedger) -> None:
    assert ledger.user_price.lora_wait_kudos.value == 3.0
    assert ledger.user_price.lora_wait_kudos.unit is LedgerUnit.KUDOS_PER_LORA
    assert ledger.user_price.lora_wait_kudos.provisional is True
    assert ledger.user_price.ti_kudos.value == 1.0
    assert ledger.user_price.ti_kudos.unit is LedgerUnit.KUDOS_PER_JOB
    assert ledger.user_price.ti_kudos.provisional is True
    assert ledger.user_price.resource_weights.sampler_second.value == 1.0
    assert ledger.user_price.resource_weights.pp_lane_second.value == 0.35


def test_amortized_surcharge_starts_empty_with_its_shape_documented(ledger: KudosPolicyLedger) -> None:
    assert ledger.user_price.amortized_model_surcharge == {}
    assert "amortized_model_surcharge" in ledger.notes
    assert ledger.amortized_model_surcharge_seconds("any model at all") == 0.0
    assert ledger.amortized_model_surcharge_seconds(None) == 0.0


def test_served_variety_is_a_worker_reward_placeholder(ledger: KudosPolicyLedger) -> None:
    """The variety term is horde-funded, so it must not sit among the items the user pays."""
    item = ledger.worker_reward.served_variety_kudos_per_assigned_swap
    assert item.value == 0.0
    assert item.unit is LedgerUnit.KUDOS_PER_SWAP
    assert item.provisional is True


def test_loader_rejects_a_missing_item(tmp_path: Path, ledger_document: dict[str, Any]) -> None:
    del ledger_document["user_price"]["ti_kudos"]
    with pytest.raises(LedgerSchemaError, match=r"user_price is missing \['ti_kudos'\]"):
        load_ledger(_write(tmp_path, ledger_document))


def test_loader_rejects_an_unrecognised_key(tmp_path: Path, ledger_document: dict[str, Any]) -> None:
    """A typo must fail the load rather than take a default, which would price a policy nobody set."""
    ledger_document["user_price"]["lora_wait_kudo"] = ledger_document["user_price"]["lora_wait_kudos"]
    with pytest.raises(LedgerSchemaError, match="unrecognised keys"):
        load_ledger(_write(tmp_path, ledger_document))


def test_loader_rejects_a_wrong_unit(tmp_path: Path, ledger_document: dict[str, Any]) -> None:
    ledger_document["user_price"]["ti_kudos"]["unit"] = LedgerUnit.MULTIPLIER.value
    with pytest.raises(LedgerSchemaError, match="unit must be 'kudos_per_job'"):
        load_ledger(_write(tmp_path, ledger_document))


def test_loader_rejects_a_non_numeric_value(tmp_path: Path, ledger_document: dict[str, Any]) -> None:
    ledger_document["user_price"]["lora_wait_kudos"]["value"] = "three"
    with pytest.raises(LedgerSchemaError, match="value must be a number"):
        load_ledger(_write(tmp_path, ledger_document))


def test_loader_rejects_empty_provenance(tmp_path: Path, ledger_document: dict[str, Any]) -> None:
    ledger_document["user_price"]["ti_kudos"]["provenance"] = "   "
    with pytest.raises(LedgerSchemaError, match="must be a non-empty string"):
        load_ledger(_write(tmp_path, ledger_document))


def test_loader_rejects_baseline_maps_that_disagree(tmp_path: Path, ledger_document: dict[str, Any]) -> None:
    """A family priced in one map and absent from another would apply half a policy."""
    del ledger_document["user_price"]["quality_premium"]["flux_1"]
    with pytest.raises(LedgerSchemaError, match=r"price different baselines.*flux_1"):
        load_ledger(_write(tmp_path, ledger_document))


def test_loader_rejects_malformed_json(tmp_path: Path) -> None:
    path = tmp_path / "ledger.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(LedgerSchemaError, match="is not valid JSON"):
        load_ledger(path)


def test_unknown_baseline_raises_rather_than_pricing_at_par(
    ledger: KudosPolicyLedger,
    basis: PricingBasis,
) -> None:
    """A newly served family must be a ledger edit, not a plausible-looking under-charge."""
    with pytest.raises(UnknownBaselineError, match="a_family_the_ledger_has_never_seen"):
        compose_user_price(
            PredictedSeconds(sampler_window=10.0),
            PayloadFeatures(baseline="a_family_the_ledger_has_never_seen"),
            ledger,
            basis,
        )
    assert ledger.knows_baseline("a_family_the_ledger_has_never_seen") is False
    assert ledger.knows_baseline(None) is False
    assert ledger.knows_baseline("stable_diffusion_1") is True


def test_basis_converts_seconds_to_kudos(ledger: KudosPolicyLedger) -> None:
    """The basis job on a par family is worth the basis kudos, whatever the machine's speed."""
    for basis_seconds in (4.0, 10.0, 31.5):
        basis = PricingBasis(basis_seconds=basis_seconds)
        breakdown = compose_user_price(
            PredictedSeconds(sampler_window=basis_seconds),
            PayloadFeatures(baseline="stable_diffusion_1"),
            ledger,
            basis,
        )
        assert breakdown.total_kudos == pytest.approx(DEFAULT_BASIS_KUDOS)


@pytest.mark.parametrize("basis_seconds", [0.0, -1.0])
def test_basis_rejects_a_non_positive_anchor(basis_seconds: float) -> None:
    with pytest.raises(ValueError, match="basis_seconds must be positive"):
        PricingBasis(basis_seconds=basis_seconds)


def test_sampler_seconds_are_the_numeraire(ledger: KudosPolicyLedger, basis: PricingBasis) -> None:
    breakdown = compose_user_price(
        PredictedSeconds(sampler_window=20.0),
        PayloadFeatures(baseline="stable_diffusion_1"),
        ledger,
        basis,
    )
    assert breakdown.sampler_seconds_kudos == pytest.approx(20.0 * basis.kudos_per_second)
    assert breakdown.pp_lane_seconds_kudos == 0.0


def test_post_processing_seconds_are_discounted_by_their_lane_weight(
    ledger: KudosPolicyLedger,
    basis: PricingBasis,
) -> None:
    breakdown = compose_user_price(
        PredictedSeconds(sampler_window=9.0, pp_lane=4.0),
        PayloadFeatures(baseline="stable_diffusion_1"),
        ledger,
        basis,
    )
    weight = ledger.user_price.resource_weights.pp_lane_second.value
    assert breakdown.pp_lane_seconds_kudos == pytest.approx(4.0 * weight * basis.kudos_per_second)
    assert breakdown.pp_lane_seconds_kudos < breakdown.sampler_seconds_kudos


def test_lora_adder_scales_with_the_count_and_the_ti_adder_does_not(
    ledger: KudosPolicyLedger,
    basis: PricingBasis,
) -> None:
    def price(loras_count: int, tis_count: int) -> Any:
        return compose_user_price(
            PredictedSeconds(sampler_window=10.0),
            PayloadFeatures(baseline="stable_diffusion_1", loras_count=loras_count, tis_count=tis_count),
            ledger,
            basis,
        )

    assert price(0, 0).lora_wait_kudos == 0.0
    assert price(3, 0).lora_wait_kudos == pytest.approx(9.0)
    assert price(0, 0).ti_kudos == 0.0
    assert price(0, 1).ti_kudos == pytest.approx(1.0)
    assert price(0, 4).ti_kudos == pytest.approx(1.0)


def test_adders_sit_inside_the_baseline_premium(ledger: KudosPolicyLedger, basis: PricingBasis) -> None:
    """The server multiplies a price that already carries its adders, and the ledger must match."""
    breakdown = compose_user_price(
        PredictedSeconds(sampler_window=10.0),
        PayloadFeatures(baseline="stable_diffusion_xl", loras_count=2, tis_count=1),
        ledger,
        basis,
    )
    subtotal = DEFAULT_BASIS_KUDOS + 6.0 + 1.0
    assert breakdown.measured_subtotal_kudos == pytest.approx(subtotal)
    assert breakdown.total_kudos == pytest.approx(subtotal * 2.0)


def test_amortized_surcharge_is_priced_at_the_sampler_weight(
    tmp_path: Path,
    ledger_document: dict[str, Any],
    basis: PricingBasis,
) -> None:
    """A populated surcharge must add its seconds to the price, so an empty map is the only reason it is zero."""
    ledger_document["user_price"]["amortized_model_surcharge"]["A Heavy Checkpoint"] = {
        "value": 6.0,
        "unit": LedgerUnit.SECONDS_PER_JOB.value,
        "provenance": "Synthetic entry used to pin how a populated surcharge is applied.",
        "provisional": True,
    }
    surcharged = load_ledger(_write(tmp_path, ledger_document))

    breakdown = compose_user_price(
        PredictedSeconds(sampler_window=10.0),
        PayloadFeatures(baseline="stable_diffusion_1", model_name="A Heavy Checkpoint"),
        surcharged,
        basis,
    )
    assert breakdown.amortized_model_surcharge_kudos == pytest.approx(6.0 * basis.kudos_per_second)
    assert breakdown.total_kudos == pytest.approx(16.0 * basis.kudos_per_second)


def test_provisional_flags_do_not_change_the_arithmetic(
    tmp_path: Path,
    ledger_document: dict[str, Any],
    ledger: KudosPolicyLedger,
    basis: PricingBasis,
) -> None:
    """The flag is a review signal; flipping every one of them must reprice nothing."""
    for item_map in ("measured_time_component", "capability_premium", "quality_premium"):
        for entry in ledger_document["user_price"][item_map].values():
            entry["provisional"] = not entry["provisional"]
    for item_name in ("lora_wait_kudos", "ti_kudos"):
        item = ledger_document["user_price"][item_name]
        item["provisional"] = not item["provisional"]
    flipped = load_ledger(_write(tmp_path, ledger_document))

    features = PayloadFeatures(baseline="flux_1", loras_count=2, tis_count=1)
    predicted = PredictedSeconds(sampler_window=30.0, pp_lane=5.0)
    assert compose_user_price(predicted, features, flipped, basis) == compose_user_price(
        predicted,
        features,
        ledger,
        basis,
    )


def test_worker_reward_adds_the_horde_funded_variety_term(
    ledger: KudosPolicyLedger,
    basis: PricingBasis,
) -> None:
    user_price = compose_user_price(
        PredictedSeconds(sampler_window=10.0),
        PayloadFeatures(baseline="stable_diffusion_1"),
        ledger,
        basis,
    )
    reward = compose_worker_reward(user_price, 4, ledger)
    rate = ledger.worker_reward.served_variety_kudos_per_assigned_swap.value
    assert reward.user_price_kudos == pytest.approx(user_price.total_kudos)
    assert reward.served_variety_kudos == pytest.approx(rate * 4)
    assert reward.total_kudos == pytest.approx(user_price.total_kudos + rate * 4)


def test_worker_reward_rejects_a_negative_swap_count(ledger: KudosPolicyLedger, basis: PricingBasis) -> None:
    user_price = compose_user_price(
        PredictedSeconds(sampler_window=10.0),
        PayloadFeatures(baseline="stable_diffusion_1"),
        ledger,
        basis,
    )
    with pytest.raises(ValueError, match="assigned_swaps must not be negative"):
        compose_worker_reward(user_price, -1, ledger)


def test_golden_pricing_section_matches_the_shipped_ledger(ledger: KudosPolicyLedger) -> None:
    golden = json.loads(GOLDEN_VECTORS_PATH.read_text(encoding="utf-8"))["pricing"]
    assert golden["ledger_version"] == ledger.ledger_version
    assert golden["reference_machine"] == ledger.reference_machine
    assert golden["basis"] == {
        "basis_seconds": FIXTURE_BASIS.basis_seconds,
        "basis_kudos": FIXTURE_BASIS.basis_kudos,
    }


def test_golden_pricing_cases_compose_to_their_recorded_breakdown(ledger: KudosPolicyLedger) -> None:
    """Every fixture case must reprice to the number the AI-Horde tree evaluates against."""
    golden = json.loads(GOLDEN_VECTORS_PATH.read_text(encoding="utf-8"))["pricing"]
    basis = PricingBasis(
        basis_seconds=golden["basis"]["basis_seconds"],
        basis_kudos=golden["basis"]["basis_kudos"],
    )
    assert golden["cases"], "the fixture carries no pricing cases"

    for case in golden["cases"]:
        seconds = case["predicted_seconds"]
        features = case["payload_features"]
        breakdown = compose_user_price(
            PredictedSeconds(sampler_window=seconds["sampler_window"], pp_lane=seconds["pp_lane"]),
            PayloadFeatures(
                baseline=features["baseline"],
                model_name=features["model_name"],
                loras_count=features["loras_count"],
                tis_count=features["tis_count"],
            ),
            ledger,
            basis,
        )
        for line_item, expected in case["expected_user_price"].items():
            assert getattr(breakdown, line_item) == pytest.approx(expected, rel=1e-12), f"{case['name']}: {line_item}"


def test_golden_fixture_is_what_the_generator_produces() -> None:
    """The checked-in fixture must be the generator's output, so no number in it was typed by hand."""
    on_disk = json.loads(GOLDEN_VECTORS_PATH.read_text(encoding="utf-8"))
    assert on_disk == build_golden_document().content
