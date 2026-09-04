"""The export stage: a LightGBM reference run in, the served npz MLP and its bundle out.

The AI-Horde server prices a job with a pure-numpy forward pass over an ``npz`` of four dense
layers (``w0..w3`` / ``b0..b3``, ReLU between them, a linear output read as seconds). That contract
is fixed here: the reference model is a gradient-boosted tree that cannot be served through it, so
the served artifact is a small multi-layer perceptron distilled from the reference's predictions.

Distillation targets the reference's predicted seconds, not the observed durations, over the rows
the reference was fit on plus dense synthetic payloads jittered around them, so the served model
tracks the reference away from the corpus as well as on it. The loss is relative rather than
absolute because pricing error is multiplicative: a second of error on a four-second job is a
mispricing, and on a four-minute job it is not.

Nothing is published unless it can be shown to serve correctly. An export writes its artifact only
after the reloaded npz reproduces the reference within the acceptance thresholds on rows neither
model was fit on, agrees with the torch network it came from, and prices the basis job positively.

:func:`predict_seconds_npz` is the forward pass, mirroring the server's arithmetic, and every
consumer here (metrics, golden vectors, tests) reads the artifact through it.
"""

import hashlib
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from hordelib.kudos_training.encoding import (
    SOURCE_MODE_WITH_MASK,
    SOURCE_MODES_WITH_IMAGE,
    frame_to_matrix,
    row_to_payload,
)
from hordelib.kudos_training.golden_vectors import ENCODER_CASES
from hordelib.kudos_training.ledger import (
    DEFAULT_BASIS_KUDOS,
    KudosPolicyLedger,
    PayloadFeatures,
    PredictedSeconds,
    PriceBreakdown,
    PricingBasis,
    compose_user_price,
    default_ledger,
)
from hordelib.kudos_training.manifest import (
    CategoricalFeature,
    FloatFeature,
    KudosFeatureManifest,
    MultiHotFeature,
    default_manifest,
)
from hordelib.utils.optional_deps import require

if TYPE_CHECKING:
    import lightgbm as lgb
    import pandas as pd

LAYER_COUNT = 4
"""Dense layers in the served artifact: three ReLU hidden layers and one linear output."""

HIDDEN_LAYER_COUNT = LAYER_COUNT - 1
"""Hidden layers, whose widths the search chooses."""

SERVED_DECIMALS = 2
"""Decimal places the server rounds a predicted duration to before pricing from it."""

_REFERENCE_MODEL_FILENAME = "model.txt"
"""The persisted LightGBM booster a train-stage run directory carries."""

_SPLITS_FILENAME = "splits.json"
"""The per-job split assignment a train-stage run directory carries."""

_EXPORT_DIRECTORY_NAME = "export"
"""Bundle directory, created inside the run directory the artifact was distilled from."""

_METADATA_FILENAME = "export.json"
_GOLDEN_VECTORS_FILENAME = "model_golden_vectors.json"
_HPO_STORAGE_FILENAME = "hpo.sqlite3"
_STUDY_NAME = "kudos-mlp-distillation"

_DEFAULT_SEED = 22
"""Seed every stochastic step of an export derives from, recorded in the bundle."""

_GOLDEN_SOURCE_ENCODER_CASE = "encoder_case"
"""Golden case drawn from the checked-in encoder payloads."""

_GOLDEN_SOURCE_VOCABULARY_SWEEP = "vocabulary_sweep"
"""Golden case generated to visit a vocabulary entry."""

_HELD_OUT_SPLIT_PREFERENCE = ("test", "validation")
"""Split names the acceptance gate scores on, in order of preference."""

_DISTILLATION_SPLITS = ("train", "validation")
"""Splits the reference was fit on, and therefore the region distillation is anchored in."""

_ENCODED_REACH = 2.0
"""Broad resampling reaches this multiple of a feature's normalization constant.

A slot encodes to 1.0 at its divisor, so the envelope is the encoded unit box doubled: wide enough
that the served model is exercised well past the corpus, narrow enough that it is not spending
capacity on payloads no request can carry.
"""

_DERIVED_SOURCE_FLAGS = ("source_image", "source_mask")
"""Payload flags implied by ``source_processing`` rather than sampled independently."""

_INTEGER_PAYLOAD_FEATURES = frozenset(
    {"height", "width", "trajectory_steps", "n_images", "loras_count", "tis_count", "queue_depth_at_dispatch"},
)
"""Float slots whose payload units are whole numbers, so a sampled payload stays one a request could carry."""

_DIMENSION_FEATURES = frozenset({"height", "width"})
"""Float slots quantized to the pixel grid a request may ask for."""

_DIMENSION_QUANTUM = 64.0
"""Pixel grid the horde's image dimensions fall on."""

_POST_PROCESSING_JITTER_MAXIMUM = 2
"""Most post-processors a jittered payload chains, matching what a request realistically asks for."""


class ExportError(Exception):
    """Base class for every failure raised while exporting a served artifact."""


class AcceptanceGateError(ExportError):
    """Raised when the distilled artifact does not reproduce the reference closely enough."""

    def __init__(self, median_ape: float, p90_ape: float, config: "ExportConfig", rows: int) -> None:
        self.median_ape = median_ape
        self.p90_ape = p90_ape
        self.rows = rows
        super().__init__(
            f"distilled model missed the acceptance gate on {rows} held-out rows: "
            f"median APE {median_ape:.4f} (limit {config.median_ape_threshold:.4f}), "
            f"p90 APE {p90_ape:.4f} (limit {config.p90_ape_threshold:.4f})",
        )


@dataclass(frozen=True)
class ExportConfig:
    """Search space, synthetic-sampling shape, and acceptance thresholds for one export."""

    seed: int = _DEFAULT_SEED
    """Seed for the search sampler, the synthetic payloads, and every network initialization."""

    trials: int = 50
    """Hyperparameter trials the search runs."""

    synthetic_samples_per_row: int = 8
    """Jittered payloads drawn around each row the reference was fit on."""

    continuous_jitter_fraction: float = 0.25
    """Local jitter width, as a fraction of a feature's normalization constant."""

    continuous_resample_probability: float = 0.25
    """Chance a continuous feature is redrawn across its whole envelope rather than jittered."""

    categorical_resample_probability: float = 0.35
    """Chance a categorical, multi-hot or boolean feature is redrawn from its vocabulary."""

    hidden_width_choices: tuple[int, ...] = (32, 64, 128, 256)
    """Widths the search may choose for each hidden layer."""

    learning_rate_range: tuple[float, float] = (1e-4, 1e-2)
    """Bounds the search draws the learning rate from, log-uniformly."""

    weight_decay_range: tuple[float, float] = (1e-8, 1e-3)
    """Bounds the search draws the weight decay from, log-uniformly."""

    min_epochs: int = 200
    """Fewest epochs the search may choose."""

    max_epochs: int = 2000
    """Most epochs the search may choose, before early stopping cuts a fit short."""

    early_stopping_patience: int = 100
    """Epochs without a validation improvement before a fit stops."""

    batch_size: int = 512
    """Rows per optimizer step."""

    validation_fraction: float = 0.2
    """Share of the observed rows held back to early-stop and score a trial on."""

    median_ape_threshold: float = 0.03
    """Largest median distilled-versus-reference error the gate admits."""

    p90_ape_threshold: float = 0.10
    """Largest 90th-percentile distilled-versus-reference error the gate admits."""

    def __post_init__(self) -> None:
        if self.trials < 1:
            raise ValueError("trials must be at least 1")
        if not self.hidden_width_choices:
            raise ValueError("hidden_width_choices must offer at least one width")
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must lie strictly between 0 and 1")
        if self.min_epochs < 1 or self.max_epochs < self.min_epochs:
            raise ValueError("epoch bounds must be positive with max_epochs at or above min_epochs")
        for name, bounds in (
            ("learning_rate_range", self.learning_rate_range),
            ("weight_decay_range", self.weight_decay_range),
        ):
            if not 0 < bounds[0] <= bounds[1]:
                raise ValueError(f"{name} must be positive with its lower bound at or below its upper bound")


@dataclass(frozen=True)
class ExportResult:
    """One completed export and where its bundle landed."""

    export_dir: Path
    model_path: Path
    metadata_path: Path
    golden_vectors_path: Path
    median_ape: float
    """Median distilled-versus-reference error on the held-out rows."""

    p90_ape: float
    """90th-percentile distilled-versus-reference error on the held-out rows."""

    held_out_rows: int
    basis_seconds: float
    """The artifact's own prediction for the manifest's basis job, which anchors every price."""

    best_params: dict[str, Any]
    """The search's winning hyperparameters."""


@dataclass(frozen=True)
class _Hyperparameters:
    """One point in the search space."""

    hidden_widths: tuple[int, ...]
    learning_rate: float
    weight_decay: float
    epochs: int


@dataclass(frozen=True)
class _SearchResult:
    """The winning trial of one hyperparameter search."""

    hyperparameters: _Hyperparameters
    """The point in the space the final fit runs at."""

    params: dict[str, Any]
    """The parameters as the study recorded them, carried into the bundle verbatim."""

    seed: int
    """Seed the winning trial fitted at, reused so the final fit reproduces that trial."""

    validation_median_ape: float
    """The winning trial's score."""


@dataclass(frozen=True)
class _FittedNetwork:
    """A trained network already reduced to the served weight layout."""

    weights: dict[str, np.ndarray]
    """``w0..w3`` / ``b0..b3``, float32, with the target scale folded into the output layer."""

    validation_median_ape: float
    """Median relative error against the reference on the distillation validation split."""


def predict_seconds_npz(weights: Mapping[str, np.ndarray], vector: np.ndarray) -> float:
    """Return the predicted seconds for one encoded payload, as the server computes them.

    Args:
        weights: The artifact's arrays, keyed ``w0..w3`` and ``b0..b3``.
        vector: One manifest-encoded feature vector.

    Returns:
        Predicted seconds. Unrounded: callers that pin or price a number round it themselves.
    """
    return float(_forward(weights, np.asarray(vector, dtype=np.float32).reshape(1, -1))[0])


def load_npz_weights(model_path: Path) -> dict[str, np.ndarray]:
    """Load a served artifact's arrays with numpy alone, the way the server loads it."""
    with np.load(model_path) as loaded:
        return {key: loaded[key].astype(np.float32) for key in _weight_keys()}


def export(
    run_dir: Path,
    clean_path: Path,
    *,
    manifest: KudosFeatureManifest | None = None,
    ledger: KudosPolicyLedger | None = None,
    config: ExportConfig | None = None,
) -> ExportResult:
    """Distil a training run's reference model into the served npz MLP and bundle it.

    Args:
        run_dir: A train-stage run directory (``model.txt`` + ``splits.json``).
        clean_path: The cleaned snapshot the run trained from.
        manifest: Feature manifest the run encoded against. Defaults to the shipped revision.
        ledger: Policy ledger the golden vectors' prices compose under. Defaults to the shipped
            revision.
        config: Search space and acceptance thresholds. Defaults to :class:`ExportConfig` defaults.

    Returns:
        Where the bundle landed and the acceptance numbers it passed on.

    Raises:
        ExportError: If the run directory is not exportable, if the held-out rows are missing, or
            if the artifact does not reproduce the network it was folded from.
        AcceptanceGateError: If the distilled model misses the acceptance thresholds.
    """
    require("pandas", extra="kudos-training", feature="kudos-train export")
    require("lightgbm", extra="kudos-training", feature="kudos-train export")
    require("optuna", extra="kudos-training", feature="kudos-train export")
    import lightgbm as lgb
    import pandas as pd

    active_manifest = manifest if manifest is not None else default_manifest()
    active_ledger = ledger if ledger is not None else default_ledger()
    active_config = config if config is not None else ExportConfig()

    booster = _load_reference_booster(run_dir)
    frame = pd.read_parquet(clean_path)
    splits = json.loads((run_dir / _SPLITS_FILENAME).read_text(encoding="utf-8"))
    split_labels = frame["job_id"].map(splits).to_numpy()

    observed_features = frame_to_matrix(frame, active_manifest)
    observed_seconds = _reference_seconds(booster, observed_features)

    distillation_mask = np.isin(split_labels, _DISTILLATION_SPLITS)
    if not distillation_mask.any():
        raise ExportError(f"{clean_path.name} carries no rows the reference was fit on; nothing to distil from")
    held_out_mask, held_out_split = _held_out_rows(split_labels)

    synthetic_features = _synthetic_features(
        frame[distillation_mask],
        manifest=active_manifest,
        config=active_config,
    )
    synthetic_seconds = _reference_seconds(booster, synthetic_features)

    distillation_features = np.vstack([observed_features[distillation_mask], synthetic_features])
    distillation_seconds = np.concatenate([observed_seconds[distillation_mask], synthetic_seconds])
    fit_mask = _fit_mask(int(distillation_mask.sum()), len(distillation_features), config=active_config)

    export_dir = run_dir / _EXPORT_DIRECTORY_NAME
    export_dir.mkdir(parents=True, exist_ok=True)

    search = _search_hyperparameters(
        features=distillation_features,
        targets=distillation_seconds,
        fit_mask=fit_mask,
        config=active_config,
        storage_path=export_dir / _HPO_STORAGE_FILENAME,
    )
    fitted = _fit_distilled_network(
        features=distillation_features,
        targets=distillation_seconds,
        fit_mask=fit_mask,
        hyperparameters=search.hyperparameters,
        seed=search.seed,
        config=active_config,
    )

    snapshot_hash = _snapshot_hash(clean_path)
    model_path = export_dir / f"kudos-{active_manifest.manifest_version}-{snapshot_hash}.npz"
    staging_path = model_path.with_name(f"{model_path.stem}.staging.npz")
    # Spread through a loosely typed mapping: a checker otherwise reads an array keyed "b0" or
    # "w0" as a candidate for savez's own keyword arguments.
    arrays: dict[str, Any] = dict(fitted.weights)
    np.savez(staging_path, **arrays)
    served_weights = load_npz_weights(staging_path)

    try:
        _check_round_trip(served_weights, fitted.weights, observed_features[held_out_mask])
        basis_seconds = _basis_seconds(served_weights, active_manifest)
        median_ape, p90_ape = _acceptance_metrics(
            served_weights,
            features=observed_features[held_out_mask],
            reference_seconds=observed_seconds[held_out_mask],
        )
        if median_ape > active_config.median_ape_threshold or p90_ape > active_config.p90_ape_threshold:
            raise AcceptanceGateError(median_ape, p90_ape, active_config, int(held_out_mask.sum()))
    except ExportError:
        staging_path.unlink(missing_ok=True)
        raise

    staging_path.replace(model_path)

    golden_document = build_model_golden_document(
        served_weights,
        model_filename=model_path.name,
        manifest=active_manifest,
        ledger=active_ledger,
        seed=active_config.seed,
    )
    golden_vectors_path = export_dir / _GOLDEN_VECTORS_FILENAME
    golden_vectors_path.write_text(json.dumps(golden_document, indent=2) + "\n", encoding="utf-8")

    metadata = {
        "manifest_version": active_manifest.manifest_version,
        "ledger_version": active_ledger.ledger_version,
        "reference_run": run_dir.name,
        "clean_snapshot": clean_path.name,
        "snapshot_hash": snapshot_hash,
        "model_file": model_path.name,
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "seeds": {"export": active_config.seed, "final_fit": search.seed},
        "library_versions": _library_versions(lgb, pd),
        "architecture": {
            "input_length": active_manifest.vector_length(),
            "hidden_widths": list(search.hyperparameters.hidden_widths),
            "layer_shapes": [list(fitted.weights[f"w{index}"].shape) for index in range(LAYER_COUNT)],
        },
        "distillation": {
            "observed_rows": int(distillation_mask.sum()),
            "synthetic_rows": int(len(synthetic_features)),
            "fit_rows": int(fit_mask.sum()),
            "validation_rows": int((~fit_mask).sum()),
            "synthetic_samples_per_row": active_config.synthetic_samples_per_row,
        },
        "hpo": {
            "trials": active_config.trials,
            "best_params": search.params,
            "best_trial_validation_median_ape": search.validation_median_ape,
            "final_validation_median_ape": fitted.validation_median_ape,
            "storage": _HPO_STORAGE_FILENAME,
        },
        "acceptance": {
            "held_out_split": held_out_split,
            "rows": int(held_out_mask.sum()),
            "median_ape": median_ape,
            "p90_ape": p90_ape,
            "median_ape_threshold": active_config.median_ape_threshold,
            "p90_ape_threshold": active_config.p90_ape_threshold,
        },
        "basis": {"basis_seconds": basis_seconds, "basis_kudos": DEFAULT_BASIS_KUDOS},
    }
    metadata_path = export_dir / _METADATA_FILENAME
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    return ExportResult(
        export_dir=export_dir,
        model_path=model_path,
        metadata_path=metadata_path,
        golden_vectors_path=golden_vectors_path,
        median_ape=median_ape,
        p90_ape=p90_ape,
        held_out_rows=int(held_out_mask.sum()),
        basis_seconds=basis_seconds,
        best_params=search.params,
    )


def build_model_golden_document(
    weights: Mapping[str, np.ndarray],
    *,
    model_filename: str,
    manifest: KudosFeatureManifest | None = None,
    ledger: KudosPolicyLedger | None = None,
    seed: int = _DEFAULT_SEED,
) -> dict[str, Any]:
    """Build the served artifact's golden-vector document.

    The cases are the checked-in encoder payloads followed by a sweep that visits every entry of
    every vocabulary, so a vocabulary slot cannot silently stop mattering to a price. Each case
    carries the artifact's predicted seconds at serving precision and the price those seconds
    compose to, anchored on the artifact's own basis prediction.

    Args:
        weights: The served artifact's arrays.
        model_filename: Name of the npz the numbers were read from.
        manifest: Feature manifest to encode against. Defaults to the shipped revision.
        ledger: Policy ledger to price under. Defaults to the shipped revision.
        seed: Seed for the sweep's continuous features.

    Returns:
        The document, as it is serialized.
    """
    active_manifest = manifest if manifest is not None else default_manifest()
    active_ledger = ledger if ledger is not None else default_ledger()

    basis = PricingBasis(basis_seconds=_basis_seconds(weights, active_manifest), basis_kudos=DEFAULT_BASIS_KUDOS)
    cases = [
        _golden_case(
            case.name,
            _GOLDEN_SOURCE_ENCODER_CASE,
            case.payload,
            weights=weights,
            manifest=active_manifest,
            ledger=active_ledger,
            basis=basis,
        )
        for case in ENCODER_CASES
    ]
    cases.extend(
        _golden_case(
            f"sweep_{index:03d}",
            _GOLDEN_SOURCE_VOCABULARY_SWEEP,
            payload,
            weights=weights,
            manifest=active_manifest,
            ledger=active_ledger,
            basis=basis,
        )
        for index, payload in enumerate(_vocabulary_sweep_payloads(active_manifest, seed=seed))
    )

    return {
        "manifest_version": active_manifest.manifest_version,
        "ledger_version": active_ledger.ledger_version,
        "model_file": model_filename,
        "basis": {"basis_seconds": basis.basis_seconds, "basis_kudos": basis.basis_kudos},
        "served_decimals": SERVED_DECIMALS,
        "note": (
            "Predicted seconds are the artifact's own forward pass at serving precision, and each "
            "price composes from that rounded figure the way the server prices from it. A case "
            "whose baseline the ledger carries no premium for is left unpriced rather than priced "
            "at par."
        ),
        "cases": cases,
    }


def _forward(weights: Mapping[str, np.ndarray], matrix: np.ndarray) -> np.ndarray:
    """Run the served forward pass over a matrix of encoded payloads, returning seconds per row."""
    values = np.asarray(matrix, dtype=np.float32)
    for index in range(LAYER_COUNT - 1):
        values = np.maximum(values @ weights[f"w{index}"].T + weights[f"b{index}"], 0)
    output_index = LAYER_COUNT - 1
    return np.asarray(
        values @ weights[f"w{output_index}"].T + weights[f"b{output_index}"],
        dtype=np.float32,
    ).reshape(-1)


def _weight_keys() -> tuple[str, ...]:
    """Return the artifact's array keys, in the order the server reads them."""
    return tuple(f"{prefix}{index}" for index in range(LAYER_COUNT) for prefix in ("w", "b"))


def _load_reference_booster(run_dir: Path) -> "lgb.Booster":
    """Load the run's persisted LightGBM reference model.

    Raises:
        ExportError: If the run directory carries no reference model to distil from.
    """
    import lightgbm as lgb

    model_path = run_dir / _REFERENCE_MODEL_FILENAME
    if not model_path.exists():
        raise ExportError(f"{run_dir} carries no {_REFERENCE_MODEL_FILENAME}; export needs a trained reference model")
    return lgb.Booster(model_file=str(model_path))


def _reference_seconds(booster: "lgb.Booster", features: np.ndarray) -> np.ndarray:
    """Return the reference model's predicted seconds for every encoded row."""
    if not len(features):
        return np.zeros(0, dtype=np.float64)
    return np.exp(np.asarray(booster.predict(features), dtype=np.float64))


def _held_out_rows(split_labels: np.ndarray) -> tuple[np.ndarray, str]:
    """Return the mask of rows the acceptance gate scores on, and which split they are.

    Raises:
        ExportError: If the run's splits leave no rows outside the reference's fit.
    """
    for split_name in _HELD_OUT_SPLIT_PREFERENCE:
        mask = split_labels == split_name
        if mask.any():
            return mask, split_name
    raise ExportError(
        "the run's splits carry no held-out rows, so the acceptance gate would score the "
        f"distillation set itself; expected one of {list(_HELD_OUT_SPLIT_PREFERENCE)}",
    )


def _fit_mask(observed_count: int, total_count: int, *, config: ExportConfig) -> np.ndarray:
    """Return the mask splitting the distillation set into fit and validation rows.

    The validation slice is drawn from the observed rows alone, which lead the set. Synthetic
    payloads are there to shape the fit away from the corpus, but a trial has to be judged on the
    payload distribution the artifact is priced against: a validation split dominated by synthetic
    draws ranks the fits that track real jobs least closely the highest.

    Args:
        observed_count: Rows encoded from the snapshot, at the head of the distillation set.
        total_count: Rows in the whole distillation set.
        config: Supplies the validation fraction and the seed.

    Returns:
        True where a row is fitted on, False where it is validated on.
    """
    generator = np.random.default_rng(config.seed)
    validation_count = max(1, int(round(observed_count * config.validation_fraction)))
    mask = np.ones(total_count, dtype=bool)
    mask[generator.permutation(observed_count)[:validation_count]] = False
    return mask


def _snapshot_hash(clean_path: Path) -> str:
    """Return the short content hash naming the artifact, over the snapshot it was distilled from."""
    return hashlib.sha256(clean_path.read_bytes()).hexdigest()[:8]


def _synthetic_features(
    frame: "pd.DataFrame",
    *,
    manifest: KudosFeatureManifest,
    config: ExportConfig,
) -> np.ndarray:
    """Encode dense synthetic payloads jittered around every row the reference was fit on."""
    generator = np.random.default_rng(config.seed)
    vectors: list[np.ndarray] = []
    for row in frame.to_dict(orient="records"):
        payload = row_to_payload(row)
        for _ in range(config.synthetic_samples_per_row):
            vectors.append(manifest.to_vector(_jitter_payload(payload, manifest, generator, config)))
    if not vectors:
        return np.zeros((0, manifest.vector_length()), dtype=np.float32)
    return np.vstack(vectors)


def _jitter_payload(
    payload: Mapping[str, Any],
    manifest: KudosFeatureManifest,
    generator: np.random.Generator,
    config: ExportConfig,
) -> dict[str, Any]:
    """Return a payload drawn near *payload*, moving each feature within its manifest range."""
    jittered = dict(payload)
    for feature in manifest.features:
        if isinstance(feature, FloatFeature):
            _jitter_float(feature, jittered, generator, config)
            continue
        if generator.random() >= config.categorical_resample_probability:
            continue
        if isinstance(feature, CategoricalFeature):
            jittered[feature.payload_keys[0]] = str(generator.choice(feature.vocabulary))
        else:
            jittered[feature.payload_keys[0]] = _sample_post_processing(feature, generator)
    _apply_source_flags(jittered)
    return jittered


def _jitter_float(
    feature: FloatFeature,
    payload: dict[str, Any],
    generator: np.random.Generator,
    config: ExportConfig,
) -> None:
    """Move one float feature's payload value, leaving derived and implied slots alone."""
    if feature.derived is not None or feature.name in _DERIVED_SOURCE_FLAGS:
        return
    key = feature.payload_keys[0]
    if feature.bool_as_float:
        if generator.random() < config.categorical_resample_probability:
            payload[key] = bool(generator.random() < 0.5)
        return

    current = _finite_or_none(payload.get(key))
    broadly_resampled = generator.random() < config.continuous_resample_probability
    payload[key] = _as_payload_number(
        feature,
        _sample_float(
            feature,
            generator,
            around=None if broadly_resampled else current,
            jitter_fraction=config.continuous_jitter_fraction,
        ),
    )


def _sample_float(
    feature: FloatFeature,
    generator: np.random.Generator,
    *,
    around: float | None,
    jitter_fraction: float,
) -> float:
    """Return a payload-unit value for *feature*, jittered around a value or drawn across its range.

    The normalization constant sets the scale for both: local jitter is a fraction of it, and a
    broad draw spans the encoded envelope. An observed value outside that envelope widens it rather
    than being pulled into it, so jittering a real row never rewrites the row.

    Args:
        feature: The float feature to draw a payload-unit value for.
        generator: Source of randomness.
        around: Value to jitter around, or ``None`` to draw across the whole envelope.
        jitter_fraction: Local jitter width, as a fraction of the feature's normalization constant.

    Returns:
        A value inside the feature's clamps, integral or pixel-quantized where the payload unit is.
    """
    lower = feature.clamp_min if feature.clamp_min is not None else 0.0
    upper = feature.divisor * _ENCODED_REACH
    if feature.clamp_max is not None:
        upper = min(upper, feature.clamp_max)
    upper = max(upper, lower)
    if around is not None:
        lower = min(lower, around)
        upper = max(upper, around)

    if around is None:
        value = float(generator.uniform(lower, upper))
    else:
        value = around + float(generator.uniform(-1.0, 1.0)) * jitter_fraction * feature.divisor

    if feature.name in _DIMENSION_FEATURES:
        value = round(value / _DIMENSION_QUANTUM) * _DIMENSION_QUANTUM
    elif feature.name in _INTEGER_PAYLOAD_FEATURES:
        value = float(round(value))
    return float(min(max(value, lower), upper))


def _as_payload_number(feature: FloatFeature, value: float) -> float | int:
    """Return *value* in the units a request states the feature in.

    A payload carries whole steps and whole pixels, so a sampled payload states them as integers
    rather than as the floats the encoder would accept either way.
    """
    if feature.name in _DIMENSION_FEATURES or feature.name in _INTEGER_PAYLOAD_FEATURES:
        return int(value)
    return value


def _sample_post_processing(feature: MultiHotFeature, generator: np.random.Generator) -> list[str]:
    """Return a post-processing chain drawn from the feature's vocabulary."""
    count = int(generator.integers(0, _POST_PROCESSING_JITTER_MAXIMUM + 1))
    if not count:
        return []
    return [str(name) for name in generator.choice(feature.vocabulary, size=count, replace=False)]


def _apply_source_flags(payload: dict[str, Any]) -> None:
    """Set the source-image and source-mask flags from the payload's source processing mode.

    The two are not independent axes: a mode carries its own source image and mask, and
    :func:`hordelib.kudos_training.encoding.row_to_payload` derives them the same way for observed
    rows.
    """
    source_processing = payload.get("source_processing")
    payload["source_image"] = source_processing in SOURCE_MODES_WITH_IMAGE
    payload["source_mask"] = source_processing == SOURCE_MODE_WITH_MASK


def _finite_or_none(value: Any) -> float | None:
    """Return *value* as a finite float, or ``None`` when it is absent, NaN, or infinite."""
    if value is None or isinstance(value, bool):
        return None
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        return None
    return as_float if np.isfinite(as_float) else None


def _search_hyperparameters(
    *,
    features: np.ndarray,
    targets: np.ndarray,
    fit_mask: np.ndarray,
    config: ExportConfig,
    storage_path: Path,
) -> "_SearchResult":
    """Search hidden widths, learning rate, epochs and weight decay against the validation split.

    The study is stored in a SQLite file inside the bundle directory, so a search is inspectable
    after the fact and a re-export starts from a clean study rather than resuming a stale one. Its
    storage is disposed before returning, because a held connection keeps the next export from
    replacing the file.

    Returns:
        The winning point in the space, the parameters and seed that produced it, and its score.
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    storage_path.unlink(missing_ok=True)
    storage = optuna.storages.RDBStorage(url=f"sqlite:///{storage_path.as_posix()}")
    study = optuna.create_study(
        study_name=_STUDY_NAME,
        storage=storage,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=config.seed),
        pruner=optuna.pruners.MedianPruner(),
    )

    def objective(trial: "optuna.Trial") -> float:
        hyperparameters = _Hyperparameters(
            hidden_widths=tuple(
                int(trial.suggest_categorical(f"hidden_width_{layer}", list(config.hidden_width_choices)))
                for layer in range(HIDDEN_LAYER_COUNT)
            ),
            learning_rate=trial.suggest_float("learning_rate", *config.learning_rate_range, log=True),
            weight_decay=trial.suggest_float("weight_decay", *config.weight_decay_range, log=True),
            epochs=trial.suggest_int("epochs", config.min_epochs, config.max_epochs, log=True),
        )

        def report(epoch: int, validation_median_ape: float) -> None:
            trial.report(validation_median_ape, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned

        fitted = _fit_distilled_network(
            features=features,
            targets=targets,
            fit_mask=fit_mask,
            hyperparameters=hyperparameters,
            seed=config.seed + trial.number,
            config=config,
            on_epoch=report,
        )
        return fitted.validation_median_ape

    try:
        study.optimize(objective, n_trials=config.trials)
        best_params = dict(study.best_params)
        best_trial_number = int(study.best_trial.number)
        best_value = float(study.best_value)
    finally:
        storage.engine.dispose()

    return _SearchResult(
        hyperparameters=_Hyperparameters(
            hidden_widths=tuple(int(best_params[f"hidden_width_{layer}"]) for layer in range(HIDDEN_LAYER_COUNT)),
            learning_rate=float(best_params["learning_rate"]),
            weight_decay=float(best_params["weight_decay"]),
            epochs=int(best_params["epochs"]),
        ),
        params=best_params,
        seed=config.seed + best_trial_number,
        validation_median_ape=best_value,
    )


def _fit_distilled_network(
    *,
    features: np.ndarray,
    targets: np.ndarray,
    fit_mask: np.ndarray,
    hyperparameters: _Hyperparameters,
    seed: int,
    config: ExportConfig,
    on_epoch: Callable[[int, float], None] | None = None,
) -> _FittedNetwork:
    """Fit one network against the reference's predicted seconds.

    The loss is squared relative error, so a job's error is weighted by what the job costs rather
    than by how long it runs. Targets are divided by their median while fitting and the divisor is
    folded back into the output layer afterwards, which conditions the fit without changing what
    the served artifact computes.

    Args:
        features: Encoded payloads of the whole distillation set.
        targets: Reference predicted seconds, aligned to *features*.
        fit_mask: True where a row is fitted on, False where it is validated on.
        hyperparameters: The point in the search space to fit at.
        seed: Seed for initialization and batch shuffling.
        config: Batch size and early-stopping patience.
        on_epoch: Called with the epoch index and its validation median error, for trial pruning.

    Returns:
        The best epoch's weights in served layout, and that epoch's validation error.
    """
    import torch

    torch.manual_seed(seed)
    shuffle_generator = torch.Generator().manual_seed(seed)

    fit_indices = torch.from_numpy(np.flatnonzero(fit_mask))
    validation_indices = torch.from_numpy(np.flatnonzero(~fit_mask))
    target_scale = float(np.median(targets[fit_mask]))
    feature_tensor = torch.from_numpy(features.astype(np.float32))
    target_tensor = torch.from_numpy((targets / target_scale).astype(np.float32)).unsqueeze(1)

    network = _build_network(feature_tensor.shape[1], hyperparameters.hidden_widths)
    optimizer = torch.optim.Adam(
        network.parameters(),
        lr=hyperparameters.learning_rate,
        weight_decay=hyperparameters.weight_decay,
    )

    best_median_ape = float("inf")
    best_state: dict[str, Any] = {}
    epochs_without_improvement = 0
    for epoch in range(hyperparameters.epochs):
        network.train()
        order = torch.randperm(len(fit_indices), generator=shuffle_generator)
        for start in range(0, len(fit_indices), config.batch_size):
            batch = fit_indices[order[start : start + config.batch_size]]
            batch_targets = target_tensor[batch]
            predicted = network(feature_tensor[batch])
            loss = torch.mean(((predicted - batch_targets) / batch_targets) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        network.eval()
        with torch.no_grad():
            validation_predicted = network(feature_tensor[validation_indices])
        validation_median_ape = float(
            np.median(
                np.abs(
                    (validation_predicted - target_tensor[validation_indices]).numpy()
                    / target_tensor[validation_indices].numpy(),
                ),
            ),
        )
        if on_epoch is not None:
            on_epoch(epoch, validation_median_ape)

        if validation_median_ape < best_median_ape:
            best_median_ape = validation_median_ape
            best_state = {name: tensor.detach().clone() for name, tensor in network.state_dict().items()}
            epochs_without_improvement = 0
            continue
        epochs_without_improvement += 1
        if epochs_without_improvement >= config.early_stopping_patience:
            break

    network.load_state_dict(best_state)
    return _FittedNetwork(
        weights=_served_weights(network, target_scale=target_scale),
        validation_median_ape=best_median_ape,
    )


def _build_network(input_length: int, hidden_widths: Sequence[int]) -> Any:
    """Create the served architecture: dense layers with ReLU between and a linear output."""
    import torch

    layers: list[Any] = []
    width_in = input_length
    for width_out in hidden_widths:
        layers.append(torch.nn.Linear(width_in, width_out))
        layers.append(torch.nn.ReLU())
        width_in = width_out
    layers.append(torch.nn.Linear(width_in, 1))
    return torch.nn.Sequential(*layers)


def _served_weights(network: Any, *, target_scale: float) -> dict[str, np.ndarray]:
    """Reduce a fitted network to the served array layout, folding the target scale into the output.

    The output layer is linear, so scaling its weight and bias is exactly the scaling the fit
    divided out; the served artifact therefore predicts seconds with no post-processing at all.
    """
    import torch

    linear_layers = [module for module in network if isinstance(module, torch.nn.Linear)]
    if len(linear_layers) != LAYER_COUNT:
        raise ExportError(f"served contract needs exactly {LAYER_COUNT} dense layers, fitted {len(linear_layers)}")

    weights: dict[str, np.ndarray] = {}
    for index, layer in enumerate(linear_layers):
        scale = target_scale if index == LAYER_COUNT - 1 else 1.0
        weights[f"w{index}"] = (layer.weight.detach().numpy() * scale).astype(np.float32)
        weights[f"b{index}"] = (layer.bias.detach().numpy() * scale).astype(np.float32)
    return weights


def _check_round_trip(
    served_weights: Mapping[str, np.ndarray],
    fitted_weights: Mapping[str, np.ndarray],
    features: np.ndarray,
) -> None:
    """Verify the reloaded artifact computes what the arrays it was written from compute.

    Raises:
        ExportError: If the npz does not reproduce the arrays or their predictions.
    """
    missing = [key for key in _weight_keys() if key not in served_weights]
    if missing:
        raise ExportError(f"exported artifact is missing arrays {missing}")
    for key in _weight_keys():
        if not np.array_equal(served_weights[key], fitted_weights[key]):
            raise ExportError(f"exported array {key} did not survive the npz round trip")
    if len(features) and not np.allclose(
        _forward(served_weights, features),
        _forward(fitted_weights, features),
        rtol=1e-5,
        atol=1e-6,
    ):
        raise ExportError("the reloaded artifact does not reproduce the predictions it was written from")


def _basis_seconds(weights: Mapping[str, np.ndarray], manifest: KudosFeatureManifest) -> float:
    """Return the artifact's predicted seconds for the manifest's basis job.

    Raises:
        ExportError: If the basis job does not price positively, which would leave the whole scale
            undefined.
    """
    basis_seconds = round(predict_seconds_npz(weights, manifest.to_vector(manifest.basis_payload)), SERVED_DECIMALS)
    if basis_seconds <= 0:
        raise ExportError(f"the artifact predicts {basis_seconds} seconds for the basis job, which prices nothing")
    return basis_seconds


def _acceptance_metrics(
    weights: Mapping[str, np.ndarray],
    *,
    features: np.ndarray,
    reference_seconds: np.ndarray,
) -> tuple[float, float]:
    """Return the median and 90th-percentile distilled-versus-reference error on held-out rows."""
    distilled_seconds = _forward(weights, features).astype(np.float64)
    ape = np.abs(distilled_seconds - reference_seconds) / reference_seconds
    return float(np.median(ape)), float(np.quantile(ape, 0.90))


def _library_versions(lightgbm_module: Any, pandas_module: Any) -> dict[str, str]:
    """Return the versions of every library whose behavior the artifact's numbers depend on."""
    import optuna
    import torch

    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "torch": torch.__version__,
        "lightgbm": lightgbm_module.__version__,
        "optuna": optuna.__version__,
        "pandas": pandas_module.__version__,
    }


def _vocabulary_sweep_payloads(manifest: KudosFeatureManifest, *, seed: int) -> list[dict[str, Any]]:
    """Build payloads that visit every entry of every vocabulary at least once.

    The sweep runs as long as the widest vocabulary, cycling each feature through its own entries,
    so the fixture's coverage grows with the manifest instead of being a fixed hand-written list.
    """
    generator = np.random.default_rng(seed)
    vocabulary_lengths = [
        len(feature.vocabulary)
        for feature in manifest.features
        if isinstance(feature, (CategoricalFeature, MultiHotFeature))
    ]
    if not vocabulary_lengths:
        return []

    payloads: list[dict[str, Any]] = []
    for index in range(max(vocabulary_lengths)):
        payload: dict[str, Any] = {}
        for feature in manifest.features:
            if isinstance(feature, FloatFeature):
                _sweep_float(feature, payload, generator)
            elif isinstance(feature, CategoricalFeature):
                payload[feature.payload_keys[0]] = feature.vocabulary[index % len(feature.vocabulary)]
            else:
                payload[feature.payload_keys[0]] = _sweep_post_processing(feature, index)
        _apply_source_flags(payload)
        payloads.append(payload)
    return payloads


def _sweep_float(feature: FloatFeature, payload: dict[str, Any], generator: np.random.Generator) -> None:
    """Draw one float feature across its envelope for a sweep payload."""
    if feature.derived is not None or feature.name in _DERIVED_SOURCE_FLAGS:
        return
    key = feature.payload_keys[0]
    if feature.bool_as_float:
        payload[key] = bool(generator.random() < 0.5)
        return
    payload[key] = _as_payload_number(feature, _sample_float(feature, generator, around=None, jitter_fraction=0.0))


def _sweep_post_processing(feature: MultiHotFeature, index: int) -> list[str]:
    """Return the post-processing chain a sweep payload at *index* carries.

    Every third payload chains two post-processors and every third carries one, so both the single
    and the chained shapes are covered while each vocabulary entry is still visited.
    """
    vocabulary_size = len(feature.vocabulary)
    if index % 3 == 1:
        return [feature.vocabulary[index % vocabulary_size]]
    if index % 3 == 2:
        return [feature.vocabulary[index % vocabulary_size], feature.vocabulary[(index + 1) % vocabulary_size]]
    return []


def _golden_case(
    name: str,
    source: str,
    payload: dict[str, Any],
    *,
    weights: Mapping[str, np.ndarray],
    manifest: KudosFeatureManifest,
    ledger: KudosPolicyLedger,
    basis: PricingBasis,
) -> dict[str, Any]:
    """Build one golden-vector case: the payload, its served seconds, and the price they compose to."""
    encoded = manifest.encode(payload)
    predicted_seconds = round(predict_seconds_npz(weights, encoded.vector), SERVED_DECIMALS)
    baseline = encoded.collapsed.get("baseline") or _resolved_baseline(manifest, payload)

    case: dict[str, Any] = {
        "name": name,
        "source": source,
        "payload": payload,
        "predicted_seconds": predicted_seconds,
        "collapsed": encoded.collapsed,
        "dropped_unknown": encoded.dropped_unknown,
        "baseline_priced": baseline,
    }
    if not ledger.knows_baseline(baseline):
        case["user_price"] = None
        return case

    breakdown = compose_user_price(
        PredictedSeconds(sampler_window=predicted_seconds),
        PayloadFeatures(
            baseline=baseline,
            loras_count=int(payload.get("loras_count") or 0),
            tis_count=int(payload.get("tis_count") or 0),
        ),
        ledger,
        basis,
    )
    case["user_price"] = _breakdown_to_case(breakdown)
    return case


def _resolved_baseline(manifest: KudosFeatureManifest, payload: Mapping[str, Any]) -> str:
    """Return the baseline the manifest encodes *payload* as, defaults included."""
    for feature in manifest.features:
        if not isinstance(feature, CategoricalFeature) or feature.name != "baseline":
            continue
        stated = payload.get(feature.payload_keys[0])
        if stated in feature.vocabulary:
            return str(stated)
        return feature.default
    raise ExportError("the manifest declares no baseline feature, so no price can be composed")


def _breakdown_to_case(breakdown: PriceBreakdown) -> dict[str, float]:
    """Convert a composed price into the golden document's per-line-item mapping."""
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


__all__ = [
    "HIDDEN_LAYER_COUNT",
    "LAYER_COUNT",
    "SERVED_DECIMALS",
    "AcceptanceGateError",
    "ExportConfig",
    "ExportError",
    "ExportResult",
    "build_model_golden_document",
    "export",
    "load_npz_weights",
    "predict_seconds_npz",
]
