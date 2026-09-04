"""The versioned policy ledger and the price composition it drives.

The trained model predicts measurable resource-seconds and nothing else. Every cost that is not a
measurable per-job duration is a named line item in this ledger: the per-baseline capability and
quality premiums, the per-feature premiums a shape-changing request substitutes for them, the
amortized model load-and-eviction surcharge, the lora and textual-inversion adders, the relative
weight of each resource a job occupies, and the horde-funded served-variety reward. A line item
stays visible, is adjustable without retraining, and is auditable when someone asks why a family
pays what it pays.

Two compositions are published here and they are deliberately different:
:func:`compose_user_price` is what the requesting user is charged, and
:func:`compose_worker_reward` adds the horde-funded items that steer the fleet rather than recover a
job's cost.

The ledger document is parsed by pydantic, which the AI-Horde server already carries, and the
composition itself is plain arithmetic over frozen dataclasses with no third-party dependency in the
request path: the server prices every request through the same functions, so the port stays a copy
rather than a hand translation, which is where pricing drift enters.
"""

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Final, Self

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, ValidationError, model_validator

LEDGER_FILENAME: Final[str] = "kudos_policy_ledger_v1.json"
"""Filename of the ledger revision this package ships."""

DEFAULT_LEDGER_PATH: Final[Path] = Path(__file__).parent / LEDGER_FILENAME
"""Location of the shipped ledger. The AI-Horde tree carries a byte-identical copy."""

DEFAULT_BASIS_KUDOS: Final[float] = 11.0
"""Kudos the basis job is worth: the horde's basis of 10 plus the image path's adjustment of 1."""


class LedgerUnit(StrEnum):
    """What a line item's value means, so a reader never has to infer it from the item's name."""

    MULTIPLIER = "multiplier"
    """A dimensionless factor applied to a composed price."""

    RELATIVE_WEIGHT = "relative_weight"
    """A resource's worth per second, expressed against a sampler-second."""

    KUDOS_PER_LORA = "kudos_per_lora"
    """Kudos added once for each lora the request names."""

    KUDOS_PER_JOB = "kudos_per_job"
    """Kudos added once to a job that meets the item's condition."""

    KUDOS_PER_SWAP = "kudos_per_swap"
    """Kudos paid per model change the server assigned to a worker."""

    SECONDS_PER_JOB = "seconds_per_job"
    """Expected seconds a job carries, priced through a resource weight like any measured second."""


class PricedFeature(StrEnum):
    """A request feature that changes the shape of the render and carries its own premium.

    These are the features the served baseline catalog prices apart from the baseline itself, and a
    premium here replaces the baseline's capability premium rather than multiplying it.
    """

    QR_CODE = "qr_code"
    """The QR-code workflow, selected by ``workflow == "qr_code"``."""

    HIRES_FIX = "hires_fix"
    """The second high-resolution pass."""


class LedgerError(Exception):
    """Base class for every failure raised while loading or applying a policy ledger."""


class LedgerSchemaError(LedgerError):
    """Raised when a ledger file does not match the schema this module implements.

    A schema failure raised by pydantic is wrapped rather than propagated so that callers keep one
    exception type to catch, and the underlying :class:`pydantic.ValidationError` stays reachable
    through :attr:`validation_error` and through the exception chain.
    """

    def __init__(self, message: str, *, validation_error: ValidationError | None = None) -> None:
        self.validation_error = validation_error
        super().__init__(message)


class UnknownBaselineError(LedgerError):
    """Raised when a price is asked for a baseline the ledger carries no premium for.

    Falling back to par would silently under-charge a family the catalog has just started serving,
    and the failure mode is invisible: the price is plausible and simply wrong. Raising makes a
    newly served baseline an explicit ledger edit.
    """

    def __init__(self, baseline: str | None, priced_baselines: tuple[str, ...]) -> None:
        self.baseline = baseline
        self.priced_baselines = priced_baselines
        super().__init__(
            f"ledger carries no premium for baseline {baseline!r}; priced baselines are {list(priced_baselines)}",
        )


def _non_blank(value: str) -> str:
    """Require a string to carry something a reviewer can read."""
    if not value.strip():
        raise ValueError("must be a non-empty string")
    return value


NonBlankString = Annotated[str, AfterValidator(_non_blank)]
"""A string that a whitespace-only value cannot satisfy."""

_LEDGER_MODEL_CONFIG: Final[ConfigDict] = ConfigDict(extra="forbid", frozen=True, strict=True)
"""Shared model configuration: an unrecognised key is a load failure and never a silent default.

Strictness is what makes a typo visible. A string where a number belongs, or a key nobody declared,
would otherwise take a default and price a policy no reviewer ever set.
"""


class LineItem(BaseModel):
    """Represents one named, auditable entry of the policy ledger."""

    model_config = _LEDGER_MODEL_CONFIG

    value: float
    """The number applied, in the item's own unit."""

    unit: Annotated[LedgerUnit, Field(strict=False)]
    """What :attr:`value` means and therefore how composition applies it."""

    provenance: NonBlankString
    """Where the number came from, in one sentence."""

    provisional: bool
    """Whether the number is a placeholder awaiting measurement.

    A provisional item is applied exactly like a settled one; the flag is a review signal, never an
    arithmetic one, so a pending measurement can never change a price by accident.
    """


def _expects_unit(unit: LedgerUnit) -> Callable[[LineItem], LineItem]:
    """Build a validator requiring a line item to carry the unit its position implies."""

    def check(item: LineItem) -> LineItem:
        if item.unit is not unit:
            raise ValueError(f"unit must be {unit.value!r}, got {item.unit.value!r}")
        return item

    return check


MultiplierItem = Annotated[LineItem, AfterValidator(_expects_unit(LedgerUnit.MULTIPLIER))]
"""A line item that must be a dimensionless multiplier."""

RelativeWeightItem = Annotated[LineItem, AfterValidator(_expects_unit(LedgerUnit.RELATIVE_WEIGHT))]
"""A line item that must be a per-second weight."""

SecondsPerJobItem = Annotated[LineItem, AfterValidator(_expects_unit(LedgerUnit.SECONDS_PER_JOB))]
"""A line item that must be an expected per-job duration."""

KudosPerLoraItem = Annotated[LineItem, AfterValidator(_expects_unit(LedgerUnit.KUDOS_PER_LORA))]
"""A line item that must be a per-lora adder."""

KudosPerJobItem = Annotated[LineItem, AfterValidator(_expects_unit(LedgerUnit.KUDOS_PER_JOB))]
"""A line item that must be a once-per-job adder."""

KudosPerSwapItem = Annotated[LineItem, AfterValidator(_expects_unit(LedgerUnit.KUDOS_PER_SWAP))]
"""A line item that must be a per-swap payment."""


class ResourceWeights(BaseModel):
    """Represents what one second of each occupied resource is worth against a sampler-second."""

    model_config = _LEDGER_MODEL_CONFIG

    sampler_second: RelativeWeightItem
    """The numeraire: the serialized bottleneck every other resource is priced against."""

    pp_lane_second: RelativeWeightItem
    """A post-processing-lane second, which runs beside the sampler on a disaggregated worker."""


class UserPriceItems(BaseModel):
    """Represents the ledger items the requesting user pays for."""

    model_config = _LEDGER_MODEL_CONFIG

    resource_weights: ResourceWeights
    """Relative worth of a second on each resource a job occupies."""

    measured_time_component: dict[str, MultiplierItem]
    """Per baseline, the share of the ported multiplier the v22 prediction already covers.

    Recorded rather than applied: the model carries this component in its predicted seconds, so
    composition would double-charge it. It is par while a baseline's cells are absent from the
    corpus, and rises as the residual premium beside it falls.
    """

    capability_premium: dict[str, MultiplierItem]
    """Per baseline, what the fleet is paid beyond measured time for being able to serve it."""

    quality_premium: dict[str, MultiplierItem]
    """Per baseline, the deliberate over-reward for output the operators judge worth more."""

    feature_premium: dict[str, dict[str, MultiplierItem]]
    """Per feature, per baseline, the multiplier that replaces the baseline's capability premium.

    Keyed by a :class:`PricedFeature` value and then by baseline. A feature the catalog prices for
    one family only carries that family alone, and a baseline absent from a feature's map keeps its
    own capability premium.
    """

    amortized_model_surcharge: dict[str, SecondsPerJobItem]
    """Per model, expected load-and-eviction seconds, priced at the sampler-second weight."""

    lora_wait_kudos: KudosPerLoraItem
    """Kudos per lora, standing in for the dispatch wait a cache miss costs."""

    ti_kudos: KudosPerJobItem
    """Kudos added once to a job that names any textual inversion."""

    @model_validator(mode="after")
    def _check_baseline_maps_agree(self) -> Self:
        """Require every per-baseline map to cover exactly the same baselines.

        A baseline present in one map and absent from another would price through a partial policy:
        the premium a reviewer thinks they set would be applied without the split that explains it.
        """
        maps = {
            "measured_time_component": self.measured_time_component,
            "capability_premium": self.capability_premium,
            "quality_premium": self.quality_premium,
        }
        reference_name, reference_map = next(iter(maps.items()))
        for name, entries in maps.items():
            if set(entries) != set(reference_map):
                difference = sorted(set(entries).symmetric_difference(reference_map))
                raise ValueError(
                    f"user_price.{name} and user_price.{reference_name} price different baselines; "
                    f"the difference is {difference}",
                )
        if not reference_map:
            raise ValueError(f"user_price.{reference_name} prices no baselines")
        return self

    @model_validator(mode="after")
    def _check_feature_premiums_are_priceable(self) -> Self:
        """Require every feature premium to name a feature and a baseline composition can apply.

        A premium keyed by a feature nobody reads, or by a family the ledger does not otherwise
        price, would never reach a price and would look settled while doing nothing.
        """
        known_features = sorted(feature.value for feature in PricedFeature)
        for feature_name, entries in self.feature_premium.items():
            if feature_name not in known_features:
                raise ValueError(
                    f"user_price.feature_premium carries unrecognised feature {feature_name!r}; "
                    f"known features are {known_features}",
                )
            unpriced = sorted(set(entries) - set(self.capability_premium))
            if unpriced:
                raise ValueError(
                    f"user_price.feature_premium.{feature_name} prices baselines the ledger carries "
                    f"no capability premium for: {unpriced}",
                )
        return self


class WorkerRewardItems(BaseModel):
    """Represents the horde-funded items paid to a worker beyond the user's price."""

    model_config = _LEDGER_MODEL_CONFIG

    served_variety_kudos_per_assigned_swap: KudosPerSwapItem
    """Kudos per model change the server assigned, covering load and displaced-model re-reads."""


class KudosPolicyLedger(BaseModel):
    """Represents one revision of the pricing policy that composes over the trained model."""

    model_config = _LEDGER_MODEL_CONFIG

    ledger_version: NonBlankString
    """Revision identifier, recorded in every model card and every exported artifact."""

    manifest_version: NonBlankString
    """Feature manifest revision this ledger's measured-time split was authored against."""

    reference_machine: NonBlankString
    """Machine id from ``machines.toml`` whose seconds the composed prices are expressed in."""

    notes: dict[str, NonBlankString]
    """Item name to a sentence describing what a populated entry means, for items that start empty."""

    user_price: UserPriceItems
    """Items the requesting user pays."""

    worker_reward: WorkerRewardItems
    """Items the horde funds on top of the user's price."""

    def priced_baselines(self) -> tuple[str, ...]:
        """Return the baselines this ledger carries premiums for, in file order."""
        return tuple(self.user_price.capability_premium)

    def capability_premium_for(self, baseline: str | None) -> LineItem:
        """Return the capability premium for *baseline*.

        Args:
            baseline: The baseline the generation runs on.

        Returns:
            The line item.

        Raises:
            UnknownBaselineError: If the ledger carries no premium for *baseline*.
        """
        return self._premium_for(self.user_price.capability_premium, baseline)

    def quality_premium_for(self, baseline: str | None) -> LineItem:
        """Return the quality premium for *baseline*.

        Args:
            baseline: The baseline the generation runs on.

        Returns:
            The line item.

        Raises:
            UnknownBaselineError: If the ledger carries no premium for *baseline*.
        """
        return self._premium_for(self.user_price.quality_premium, baseline)

    def measured_time_component_for(self, baseline: str | None) -> LineItem:
        """Return the recorded measured-time share of *baseline*'s ported multiplier.

        Args:
            baseline: The baseline the generation runs on.

        Returns:
            The line item.

        Raises:
            UnknownBaselineError: If the ledger carries no entry for *baseline*.
        """
        return self._premium_for(self.user_price.measured_time_component, baseline)

    def feature_premium_for(self, feature: PricedFeature | str, baseline: str | None) -> LineItem | None:
        """Return the premium *feature* substitutes for *baseline*'s capability premium.

        Args:
            feature: The request feature being priced.
            baseline: The baseline the generation runs on.

        Returns:
            The line item, or None where the catalog prices this feature at the baseline's own rate.
        """
        if baseline is None:
            return None
        return self.user_price.feature_premium.get(str(feature), {}).get(baseline)

    def knows_baseline(self, baseline: str | None) -> bool:
        """Return whether this ledger can price *baseline*."""
        return baseline is not None and baseline in self.user_price.capability_premium

    def amortized_model_surcharge_seconds(self, model_name: str | None) -> float:
        """Return the expected load-and-eviction seconds carried by *model_name*.

        An unlisted model surcharges nothing, which is the correct reading of an unpopulated map:
        the corpus has not measured that model's churn, so charging a guess would price luck.

        Args:
            model_name: The model the job ran, or None where the caller has no model.

        Returns:
            Expected seconds per job, zero when the model has no measured entry.
        """
        if model_name is None:
            return 0.0
        surcharge = self.user_price.amortized_model_surcharge.get(model_name)
        return surcharge.value if surcharge is not None else 0.0

    def _premium_for(self, items: Mapping[str, LineItem], baseline: str | None) -> LineItem:
        """Return *baseline*'s entry from *items*, raising when the ledger does not carry one."""
        if baseline is not None:
            item = items.get(baseline)
            if item is not None:
                return item
        raise UnknownBaselineError(baseline, self.priced_baselines())


@dataclass(frozen=True)
class PricingBasis:
    """Represents the anchor that turns predicted seconds into kudos.

    The horde's price scale is defined by one reference job rather than by an absolute rate, so a
    faster fleet does not deflate every price: the basis job's predicted seconds on the reference
    machine are worth :attr:`basis_kudos`, and every other job is priced in proportion.
    """

    basis_seconds: float
    """Predicted seconds for the manifest's basis payload on the reference machine."""

    basis_kudos: float = DEFAULT_BASIS_KUDOS
    """Kudos the basis job is worth."""

    def __post_init__(self) -> None:
        if self.basis_seconds <= 0:
            raise ValueError(f"basis_seconds must be positive, got {self.basis_seconds}")
        if self.basis_kudos <= 0:
            raise ValueError(f"basis_kudos must be positive, got {self.basis_kudos}")

    @property
    def kudos_per_second(self) -> float:
        """Return the kudos one resource-second of the basis job is worth."""
        return self.basis_kudos / self.basis_seconds


@dataclass(frozen=True)
class PredictedSeconds:
    """Represents the per-resource durations a job is predicted to occupy.

    Only the sampler window is predicted today; the lane fields are here because the price
    composition already prices them, so populating them is a training change and not a policy one.
    """

    sampler_window: float
    """Occupancy of the serialized sampler resource, the model's primary target."""

    pp_lane: float = 0.0
    """Occupancy of the post-processing lane, zero until the worker reports per-stage durations."""


@dataclass(frozen=True)
class PayloadFeatures:
    """Represents the request-side fields the ledger reads, as distinct from the model's features."""

    baseline: str
    """Baseline of the model the job runs on, keying the per-baseline premiums."""

    model_name: str | None = None
    """Model the job runs, keying the amortized load-and-eviction surcharge."""

    loras_count: int = 0
    """Number of loras the request names."""

    tis_count: int = 0
    """Number of textual inversions the request names."""

    hires_fix: bool = False
    """Whether the request asked for the second high-resolution pass."""

    workflow: str | None = None
    """Named workflow the request selected; ``qr_code`` is the one the catalog prices apart."""

    def priced_feature(self) -> PricedFeature | None:
        """Return the shape-changing feature this request is priced under, if any.

        The QR-code workflow wins over the high-resolution pass, matching the order the server
        resolves them in.
        """
        if self.workflow == PricedFeature.QR_CODE.value:
            return PricedFeature.QR_CODE
        if self.hires_fix:
            return PricedFeature.HIRES_FIX
        return None


@dataclass(frozen=True)
class PriceBreakdown:
    """Represents one composed user price with every line item's contribution kept separate."""

    sampler_seconds_kudos: float
    """Predicted sampler-window seconds at the sampler-second weight."""

    pp_lane_seconds_kudos: float
    """Predicted post-processing-lane seconds at the lane weight."""

    amortized_model_surcharge_kudos: float
    """The model's expected load-and-eviction seconds at the sampler-second weight."""

    lora_wait_kudos: float
    """The lora adder times the lora count."""

    ti_kudos: float
    """The textual-inversion adder, applied once when the request names any."""

    measured_subtotal_kudos: float
    """Sum of the items above: what the job costs before the per-baseline premiums."""

    capability_premium: float
    """The baseline's capability multiplier, whether or not a feature premium replaced it."""

    feature_premium: float | None
    """The multiplier a shape-changing feature substituted for the capability premium, if any."""

    quality_premium: float
    """The quality multiplier applied to the subtotal."""

    total_kudos: float
    """What the requesting user pays."""


@dataclass(frozen=True)
class WorkerRewardBreakdown:
    """Represents what a worker earns for a job: the user's price plus the horde-funded items."""

    user_price_kudos: float
    """The composed user price this reward is built on."""

    served_variety_kudos: float
    """The horde-funded payment for the model changes the server assigned."""

    total_kudos: float
    """What the worker is paid."""


def compose_user_price(
    predicted_seconds: PredictedSeconds,
    payload_features: PayloadFeatures,
    ledger: KudosPolicyLedger,
    basis: PricingBasis,
) -> PriceBreakdown:
    """Compose what a requesting user pays for one job.

    Measured resource-seconds are weighted by their resource's scarcity and converted to kudos
    through the basis; the data-derived surcharges are added in the same currency; the per-baseline
    premiums multiply the whole subtotal, as the server's baseline multiplier does today.

    A request that changes the shape of the render pays the catalog's premium for that feature
    *instead of* the baseline's own, which is how the server resolves the two.

    The recorded measured-time component is not applied here: it names the share of the ported
    multiplier the model's predicted seconds already carry, so applying it would charge that share
    twice.

    Args:
        predicted_seconds: Per-resource durations the model predicts for the job.
        payload_features: The request-side fields the ledger reads.
        ledger: The policy ledger revision to price under.
        basis: The seconds-to-kudos anchor for the reference machine.

    Returns:
        Every line item's contribution and the total.

    Raises:
        UnknownBaselineError: If the ledger carries no premium for the payload's baseline.
    """
    weights = ledger.user_price.resource_weights
    kudos_per_second = basis.kudos_per_second

    sampler_seconds_kudos = predicted_seconds.sampler_window * weights.sampler_second.value * kudos_per_second
    pp_lane_seconds_kudos = predicted_seconds.pp_lane * weights.pp_lane_second.value * kudos_per_second
    # A model load occupies the card itself, so it is priced at the sampler-second weight rather
    # than at a lane's.
    surcharge_seconds = ledger.amortized_model_surcharge_seconds(payload_features.model_name)
    amortized_model_surcharge_kudos = surcharge_seconds * weights.sampler_second.value * kudos_per_second

    lora_wait_kudos = ledger.user_price.lora_wait_kudos.value * payload_features.loras_count
    ti_kudos = ledger.user_price.ti_kudos.value if payload_features.tis_count > 0 else 0.0

    measured_subtotal_kudos = (
        sampler_seconds_kudos + pp_lane_seconds_kudos + amortized_model_surcharge_kudos + lora_wait_kudos + ti_kudos
    )

    capability_premium = ledger.capability_premium_for(payload_features.baseline).value
    quality_premium = ledger.quality_premium_for(payload_features.baseline).value

    feature = payload_features.priced_feature()
    feature_item = ledger.feature_premium_for(feature, payload_features.baseline) if feature is not None else None
    feature_premium = feature_item.value if feature_item is not None else None
    applied_premium = feature_premium if feature_premium is not None else capability_premium

    return PriceBreakdown(
        sampler_seconds_kudos=sampler_seconds_kudos,
        pp_lane_seconds_kudos=pp_lane_seconds_kudos,
        amortized_model_surcharge_kudos=amortized_model_surcharge_kudos,
        lora_wait_kudos=lora_wait_kudos,
        ti_kudos=ti_kudos,
        measured_subtotal_kudos=measured_subtotal_kudos,
        capability_premium=capability_premium,
        feature_premium=feature_premium,
        quality_premium=quality_premium,
        total_kudos=measured_subtotal_kudos * applied_premium * quality_premium,
    )


def compose_worker_reward(
    user_price: PriceBreakdown,
    assigned_swaps: int,
    ledger: KudosPolicyLedger,
) -> WorkerRewardBreakdown:
    """Compose what a worker earns for one job.

    The served-variety term is horde-funded rather than paid by the requesting user, and it counts
    model changes the server itself assigned, so nothing in a submit payload can inflate it.

    Args:
        user_price: The composed user price for the same job.
        assigned_swaps: Model changes the server assigned this worker for this job, from its own
            assignment history.
        ledger: The policy ledger revision to price under.

    Returns:
        The user price, the horde-funded contribution, and the total.

    Raises:
        ValueError: If *assigned_swaps* is negative.
    """
    if assigned_swaps < 0:
        raise ValueError(f"assigned_swaps must not be negative, got {assigned_swaps}")

    rate = ledger.worker_reward.served_variety_kudos_per_assigned_swap.value
    served_variety_kudos = rate * assigned_swaps
    return WorkerRewardBreakdown(
        user_price_kudos=user_price.total_kudos,
        served_variety_kudos=served_variety_kudos,
        total_kudos=user_price.total_kudos + served_variety_kudos,
    )


def load_ledger(path: str | Path | None = None) -> KudosPolicyLedger:
    """Load and validate a policy ledger.

    Args:
        path: Ledger file to read. Defaults to the revision shipped with this package.

    Returns:
        The validated ledger.

    Raises:
        LedgerSchemaError: If the file is not valid JSON or does not match the ledger schema.
        FileNotFoundError: If *path* does not exist.
    """
    resolved = Path(path) if path is not None else DEFAULT_LEDGER_PATH
    try:
        document = json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError as decode_error:
        raise LedgerSchemaError(f"{resolved} is not valid JSON: {decode_error}") from decode_error
    try:
        return KudosPolicyLedger.model_validate(document)
    except ValidationError as validation_error:
        raise LedgerSchemaError(
            f"{resolved} does not match the ledger schema: {validation_error}",
            validation_error=validation_error,
        ) from validation_error


@lru_cache(maxsize=1)
def default_ledger() -> KudosPolicyLedger:
    """Return the ledger revision shipped with this package, parsed once per process."""
    return load_ledger()


__all__ = [
    "DEFAULT_BASIS_KUDOS",
    "DEFAULT_LEDGER_PATH",
    "LEDGER_FILENAME",
    "KudosPolicyLedger",
    "LedgerError",
    "LedgerSchemaError",
    "LedgerUnit",
    "LineItem",
    "PayloadFeatures",
    "PredictedSeconds",
    "PriceBreakdown",
    "PricedFeature",
    "PricingBasis",
    "ResourceWeights",
    "UnknownBaselineError",
    "UserPriceItems",
    "WorkerRewardBreakdown",
    "WorkerRewardItems",
    "compose_user_price",
    "compose_worker_reward",
    "default_ledger",
    "load_ledger",
]
