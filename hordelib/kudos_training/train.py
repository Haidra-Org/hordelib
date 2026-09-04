"""The train stage: cleaned snapshot in, a LightGBM reference model run directory out.

The reference model regresses ``log(sampler_window_seconds)`` with a Huber objective and monotone
constraints on the axes where cost physically cannot decrease (trajectory steps, megapixels, batch size). It
defines the accuracy bar; a served artifact (ONNX or distilled npz MLP) is produced from it by the
export stage.

Splits are time-ordered and grouped by (machine, day) so that a model is never scored on data
interleaved with what it trained on. When the corpus spans too few (machine, day) groups for a
grouped split to mean anything, the split degrades to time-ordered rows within the corpus and the
run config records that it did.
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from hordelib.kudos_training.encoding import frame_to_matrix
from hordelib.kudos_training.manifest import KudosFeatureManifest, default_manifest
from hordelib.utils.optional_deps import require

if TYPE_CHECKING:
    import pandas as pd

_MONOTONE_INCREASING_FEATURES = ("trajectory_steps", "megapixels", "n_images")
"""Feature slots constrained to never decrease the prediction."""

_MIN_GROUPS_FOR_GROUPED_SPLIT = 5
"""Below this many (machine, day) groups, the grouped split degrades to time-ordered rows."""

_SPLIT_FRACTIONS = (0.70, 0.15, 0.15)
"""Train / validation / test fractions, applied to groups or rows depending on the split mode."""


@dataclass(frozen=True)
class TrainConfig:
    """Hyperparameters and reproducibility inputs for one training run.

    The defaults are sized for the standard-tier corpus (hundreds of rows); they are not tuned,
    because the LightGBM reference needs no search to serve as the accuracy bar.
    """

    seed: int = 22
    minimum_trajectory_levels_per_sampler: int = 2
    """Minimum distinct trajectory lengths required for every sampler present in corpus rows."""

    n_estimators: int = 600
    learning_rate: float = 0.05
    num_leaves: int = 31
    min_child_samples: int = 5
    huber_alpha: float = 0.9
    early_stopping_rounds: int = 50


@dataclass(frozen=True)
class TrainResult:
    """One completed training run."""

    run_dir: Path
    split_mode: str
    rows: dict[str, int]
    metrics: dict[str, dict[str, float]]


def train(
    clean_path: Path,
    *,
    out_dir: Path,
    manifest: KudosFeatureManifest | None = None,
    config: TrainConfig | None = None,
) -> TrainResult:
    """Train the LightGBM reference model from a cleaned snapshot.

    Args:
        clean_path: A sanitized snapshot parquet.
        out_dir: Directory run directories are created under (``runs/<timestamp>/``).
        manifest: Feature manifest to encode against. Defaults to the shipped revision.
        config: Hyperparameters and seed. Defaults to :class:`TrainConfig` defaults.

    Returns:
        The run directory and its headline metrics.

    Raises:
        ValueError: If the snapshot has no usable target values.
    """
    require("pandas", extra="kudos-training", feature="kudos-train train")
    require("lightgbm", extra="kudos-training", feature="kudos-train train")
    import lightgbm as lgb
    import pandas as pd

    active_manifest = manifest if manifest is not None else default_manifest()
    active_config = config if config is not None else TrainConfig()

    frame = pd.read_parquet(clean_path)
    if frame["sampler_window_seconds"].isna().any() or (frame["sampler_window_seconds"] <= 0).any():
        raise ValueError("cleaned snapshot still carries missing or non-positive targets; sanitize it first")
    _validate_sampler_trajectory_coverage(
        frame,
        minimum_levels=active_config.minimum_trajectory_levels_per_sampler,
    )

    split_labels, split_mode = _assign_splits(frame)
    features = frame_to_matrix(frame, active_manifest)
    log_target = np.log(frame["sampler_window_seconds"].to_numpy(dtype=np.float64))

    train_mask = split_labels == "train"
    validation_mask = split_labels == "validation"
    test_mask = split_labels == "test"

    constraints = _monotone_constraints(active_manifest)
    model = lgb.LGBMRegressor(
        objective="huber",
        alpha=active_config.huber_alpha,
        n_estimators=active_config.n_estimators,
        learning_rate=active_config.learning_rate,
        num_leaves=active_config.num_leaves,
        min_child_samples=active_config.min_child_samples,
        monotone_constraints=constraints,
        random_state=active_config.seed,
        verbose=-1,
    )
    model.fit(
        features[train_mask],
        log_target[train_mask],
        eval_set=[(features[validation_mask], log_target[validation_mask])],
        callbacks=[lgb.early_stopping(active_config.early_stopping_rounds, verbose=False)],
    )

    metrics = {
        split_name: _split_metrics(model, features[mask], log_target[mask])
        for split_name, mask in (("train", train_mask), ("validation", validation_mask), ("test", test_mask))
        if int(mask.sum())
    }

    run_dir = out_dir / time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    run_dir.mkdir(parents=True, exist_ok=False)
    model.booster_.save_model(str(run_dir / "model.txt"))

    slot_names = active_manifest.slot_names()
    importances = dict(
        sorted(
            zip(slot_names, (int(value) for value in model.feature_importances_), strict=True),
            key=lambda pair: -pair[1],
        ),
    )
    run_config: dict[str, Any] = {
        "clean_snapshot": clean_path.name,
        "manifest_version": active_manifest.manifest_version,
        "sampler_semantics": active_manifest.sampler_semantics.model_dump(mode="json"),
        "target": active_manifest.target,
        "split_mode": split_mode,
        "rows": {name: int((split_labels == name).sum()) for name in ("train", "validation", "test")},
        "monotone_constraints": dict(zip(slot_names, constraints, strict=True)),
        "config": active_config.__dict__,
        "best_iteration": int(model.best_iteration_) if model.best_iteration_ is not None else None,
    }
    (run_dir / "config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    split_by_job = dict(zip(frame["job_id"], (str(label) for label in split_labels), strict=True))
    (run_dir / "splits.json").write_text(json.dumps(split_by_job, indent=2), encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (run_dir / "feature_importance.json").write_text(json.dumps(importances, indent=2), encoding="utf-8")

    return TrainResult(
        run_dir=run_dir,
        split_mode=split_mode,
        rows={name: int((split_labels == name).sum()) for name in ("train", "validation", "test")},
        metrics=metrics,
    )


def _validate_sampler_trajectory_coverage(frame: "pd.DataFrame", *, minimum_levels: int) -> None:
    """Validate that corpus observations can identify sampler-specific trajectory effects.

    Args:
        frame: Cleaned snapshot rows about to enter training.
        minimum_levels: Required distinct requested trajectory lengths per observed sampler.

    Raises:
        ValueError: If the minimum is invalid or any corpus sampler has insufficient coverage.
    """
    if minimum_levels < 2:
        raise ValueError("minimum_trajectory_levels_per_sampler must be at least 2")

    corpus = frame[frame["source_kind"] == "corpus"]
    levels_by_sampler = corpus.groupby("sampler_name", dropna=False)["trajectory_steps"].nunique()
    insufficient = {
        str(sampler_name): int(level_count)
        for sampler_name, level_count in levels_by_sampler.items()
        if int(level_count) < minimum_levels
    }
    if insufficient:
        formatted = ", ".join(f"{sampler}: {levels}" for sampler, levels in sorted(insufficient.items()))
        raise ValueError(
            "sampler trajectory coverage is insufficient; each observed corpus sampler needs "
            f"at least {minimum_levels} distinct trajectory lengths ({formatted})",
        )


def _assign_splits(frame: "pd.DataFrame") -> tuple[np.ndarray, str]:
    """Assign every row to train/validation/test, time-ordered.

    Groups are (machine, UTC day); whole groups land in one split so a day's drift cannot leak
    across the boundary. With too few groups the assignment degrades to time-ordered rows.

    Returns:
        Per-row split labels (aligned to the frame) and the split mode used.
    """
    import pandas as pd

    day = pd.to_datetime(frame["time_popped"], unit="s", utc=True).dt.strftime("%Y-%m-%d")
    group_key = frame["machine_id"].astype(str) + "@" + day
    group_start = frame.groupby(group_key)["time_popped"].transform("min")

    labels = np.empty(len(frame), dtype=object)
    unique_groups = group_key.groupby(group_key).head(1)
    if len(set(unique_groups)) >= _MIN_GROUPS_FOR_GROUPED_SPLIT:
        ordered_groups = (
            pd.DataFrame({"group": group_key, "start": group_start}).drop_duplicates("group").sort_values("start")
        )
        boundaries = _split_boundaries(len(ordered_groups))
        assignment = {
            group: _label_for_position(position, boundaries) for position, group in enumerate(ordered_groups["group"])
        }
        labels[:] = [assignment[group] for group in group_key]
        return labels, "grouped_by_machine_day"

    order = np.argsort(frame["time_popped"].to_numpy())
    boundaries = _split_boundaries(len(frame))
    for position, frame_position in enumerate(order):
        labels[frame_position] = _label_for_position(position, boundaries)
    return labels, "time_ordered_rows"


def _split_boundaries(count: int) -> tuple[int, int]:
    """Return the index boundaries splitting *count* items into the configured fractions."""
    train_end = int(count * _SPLIT_FRACTIONS[0])
    validation_end = train_end + int(count * _SPLIT_FRACTIONS[1])
    return train_end, max(validation_end, train_end + 1)


def _label_for_position(position: int, boundaries: tuple[int, int]) -> str:
    """Return the split label for an item at *position* under *boundaries*."""
    if position < boundaries[0]:
        return "train"
    if position < boundaries[1]:
        return "validation"
    return "test"


def _monotone_constraints(manifest: KudosFeatureManifest) -> list[int]:
    """Return the per-slot monotone constraint vector (+1 increasing, 0 unconstrained)."""
    constrained = set(_MONOTONE_INCREASING_FEATURES)
    return [1 if name in constrained else 0 for name in manifest.slot_names()]


def _split_metrics(model: Any, features: np.ndarray, log_target: np.ndarray) -> dict[str, float]:
    """Compute seconds-domain error metrics for one split."""
    predicted_seconds = np.exp(np.asarray(model.predict(features), dtype=np.float64))
    actual_seconds = np.exp(log_target)
    ape = np.abs(predicted_seconds - actual_seconds) / actual_seconds
    return {
        "rows": float(len(actual_seconds)),
        "median_ape": float(np.median(ape)),
        "p90_ape": float(np.quantile(ape, 0.90)),
        "mae_seconds": float(np.mean(np.abs(predicted_seconds - actual_seconds))),
    }


__all__ = ["TrainConfig", "TrainResult", "train"]
