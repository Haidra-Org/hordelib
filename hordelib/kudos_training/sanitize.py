"""The sanitize stage: snapshot in, cleaned snapshot plus a sanitation report out.

Mechanism-aware filters run first, statistics second. Every filter is a named, individually
togglable rule that reports its drop count, so a rule silently eating the corpus is visible in the
report rather than discoverable only from a shrunken row count.

Rule numbering follows the design document (``docs/kudos-model-training.md``): rule 4 (concurrent
slot occupancy) is deliberately absent because the worker does not yet record queue depth at
dispatch; the report names that gap. The warmup and alchemy drops are structural exclusions rather
than numbered sanitation rules, but they are reported the same way.
"""

import json
from collections.abc import Hashable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from horde_sdk.generation_parameters.image.consts import KNOWN_IMAGE_SAMPLERS

from hordelib.kudos_training.manifest import default_manifest
from hordelib.kudos_training.schema import SNAPSHOT_SCHEMA_VERSION
from hordelib.utils.optional_deps import require

if TYPE_CHECKING:
    import pandas as pd

_MAD_TO_SIGMA = 0.6745
"""Scale factor making a MAD-based z-score comparable to a normal-deviate z-score."""


@dataclass(frozen=True)
class SanitizeConfig:
    """Rule toggles and bounds for one sanitize run.

    Attributes:
        drop_alchemy: Drop alchemy records; the image kudos model does not price them.
        drop_warmup: Drop the corpus warmup block, which exists to be excluded.
        drop_faulted: Rule 1: drop faulted jobs.
        drop_unlabeled_swap: Rule 2: drop the first job after a model swap on production sessions
            (load warmup contaminates the window). Corpus rows are exempt as a class: their cache
            regime is a controlled variable and their cold cells are the labeled load measurement.
        drop_degenerate_timestamps: Rule 3: drop rows whose stage timestamps are out of order,
            whose target window is missing or non-positive, or whose window exceeds the sanity
            bound.
        drop_incompatible_adaptive_execution: Drop adaptive-sampler rows that do not declare the
            execution contract used by the current manifest. The requested trajectory length alone
            does not characterize legacy adaptive execution.
        max_window_seconds: Rule 3's upper sanity bound on the target window.
        mad_outliers: Rule 5: robust per-cell pruning via MAD z-score on the log target.
        mad_z_bound: Rule 5's z-score bound.
        mad_min_group: Rule 5 skips groups smaller than this; a MAD from two points is noise.
        residual_prune: Rule 6: one round of residual-based pruning after a preliminary fit.
        residual_z_bound: Rule 6's robust z-score bound on fit residuals.
        residual_max_fraction: Rule 6 drops at most this fraction of its input rows, so the rule
            cannot become a self-confirming filter.
        min_cell_survivors: Neither statistical rule (5 or 6) may take a (machine, cell) group
            below this many rows; with three replicates a cell, at most one can be an outlier
            before "outlier" stops being a meaningful claim.
        seed: Seed for the preliminary fit rule 6 trains.
    """

    drop_alchemy: bool = True
    drop_warmup: bool = True
    drop_faulted: bool = True
    drop_unlabeled_swap: bool = True
    drop_degenerate_timestamps: bool = True
    drop_incompatible_adaptive_execution: bool = True
    max_window_seconds: float = 3600.0
    mad_outliers: bool = True
    mad_z_bound: float = 3.5
    mad_min_group: int = 3
    residual_prune: bool = True
    residual_z_bound: float = 3.5
    residual_max_fraction: float = 0.02
    min_cell_survivors: int = 2
    seed: int = 22


@dataclass(frozen=True)
class SanitizeResult:
    """The cleaned snapshot a sanitize run produced, and what each rule removed."""

    clean_path: Path
    content_hash: str
    rows_in: int
    rows_out: int
    dropped_by_rule: dict[str, int] = field(default_factory=dict)


def sanitize(
    snapshot_path: Path,
    *,
    out_dir: Path,
    config: SanitizeConfig | None = None,
) -> SanitizeResult:
    """Apply the sanitation rules to a snapshot and write the cleaned result.

    Args:
        snapshot_path: An assembled snapshot parquet.
        out_dir: Directory the cleaned snapshot and report are written into.
        config: Rule toggles and bounds; defaults apply the design document's defaults.

    Returns:
        The cleaned snapshot and per-rule drop counts.
    """
    require("pandas", extra="kudos-training", feature="kudos-train sanitize")
    import pandas as pd

    active_config = config if config is not None else SanitizeConfig()
    required_execution_contract = default_manifest().sampler_semantics.execution_contract_version
    frame = pd.read_parquet(snapshot_path)
    rows_in = len(frame)

    dropped_by_rule: dict[str, int] = {}
    dropped_job_ids: dict[str, list[str]] = {}

    def apply_rule(name: str, mask: "pd.Series") -> None:
        """Drop rows where *mask* is true, recording the count and job ids under *name*."""
        nonlocal frame
        dropped = frame[mask]
        dropped_by_rule[name] = len(dropped)
        if len(dropped):
            dropped_job_ids[name] = list(dropped["job_id"])
            frame = frame[~mask]

    if active_config.drop_alchemy:
        apply_rule("drop_alchemy", frame["is_alchemy"])
    if active_config.drop_warmup:
        apply_rule("drop_warmup", frame["warmup"])
    if active_config.drop_faulted:
        apply_rule("rule1_drop_faulted", frame["faulted"])
    if active_config.drop_unlabeled_swap:
        apply_rule("rule2_drop_unlabeled_swap", _unlabeled_swap_mask(frame))
    if active_config.drop_degenerate_timestamps:
        window = frame["sampler_window_seconds"]
        degenerate = (
            ~frame["stage_order_ok"] | window.isna() | (window <= 0) | (window > active_config.max_window_seconds)
        )
        apply_rule("rule3_drop_degenerate_timestamps", degenerate)
    if active_config.drop_incompatible_adaptive_execution:
        adaptive_without_contract = (frame["sampler_name"] == KNOWN_IMAGE_SAMPLERS.k_dpm_adaptive.value) & (
            frame["sampler_execution_contract_version"] != required_execution_contract.value
        )
        apply_rule("drop_incompatible_adaptive_execution", adaptive_without_contract)
    if active_config.mad_outliers:
        candidate = _mad_outlier_mask(frame, active_config)
        apply_rule("rule5_mad_outliers", _bound_by_cell_survivors(frame, candidate, active_config))
    if active_config.residual_prune:
        candidate = _residual_prune_mask(frame, active_config)
        apply_rule("rule6_residual_prune", _bound_by_cell_survivors(frame, candidate, active_config))

    clean_path, content_hash = _write_clean(frame, snapshot_path=snapshot_path, out_dir=out_dir)
    _write_report(
        clean_path,
        snapshot_path=snapshot_path,
        config=active_config,
        rows_in=rows_in,
        rows_out=len(frame),
        dropped_by_rule=dropped_by_rule,
        dropped_job_ids=dropped_job_ids,
    )
    return SanitizeResult(
        clean_path=clean_path,
        content_hash=content_hash,
        rows_in=rows_in,
        rows_out=len(frame),
        dropped_by_rule=dropped_by_rule,
    )


def _unlabeled_swap_mask(frame: "pd.DataFrame") -> "pd.Series":
    """Flag the first job after a model transition on production sessions, per session, in pop order.

    Corpus rows are exempt as a class: the corpus ordering makes cache state a controlled variable
    (models are deliberately interleaved while resident, and the cold cells are the labeled
    load-cost measurements), so a corpus model transition is a designed condition rather than
    contamination. On production sessions the session's first job is flagged too: an unlabeled
    initial load is warmup by another name.
    """
    import pandas as pd

    ordered = frame.sort_values("time_popped")
    mask_by_index: dict[Hashable, bool] = {}
    group_columns = ["machine_id", "stats_file"]
    for _, session in ordered.groupby(group_columns, sort=False):
        previous_model = None
        for index, row in session.iterrows():
            is_production = row["source_kind"] == "production"
            is_swap = row["model_name"] != previous_model
            mask_by_index[index] = is_production and is_swap
            previous_model = row["model_name"]

    return pd.Series([mask_by_index[index] for index in frame.index], index=frame.index)


def _cell_keys(frame: "pd.DataFrame") -> "pd.Series":
    """Return the per-row grouping key rule 5 prunes within.

    Corpus rows group by their cell id; production rows, which have no cell, group by the shape
    axes so a slow shape is not read as an outlier of a fast one.
    """
    shape_key = (
        frame["model_name"].astype(str)
        + "|"
        + frame["width"].astype(str)
        + "x"
        + frame["height"].astype(str)
        + "|s"
        + frame["trajectory_steps"].astype(str)
        + "|n"
        + frame["n_images"].astype(str)
        + "|"
        + frame["sampler_name"].astype(str)
        + "|"
        + frame["scheduler"].astype(str)
        + "|pp"
        + frame["post_processing"].map(lambda chain: ",".join(chain) if chain is not None else "")
    )
    return frame["cell_id"].where(frame["cell_id"].notna(), shape_key)


def _bound_by_cell_survivors(
    frame: "pd.DataFrame",
    candidate: "pd.Series",
    config: SanitizeConfig,
) -> "pd.Series":
    """Unflag candidate drops that would take a (machine, cell) group below the survivor floor.

    Within each group the excess flags are released in frame order; the rules that produce the
    candidates already ranked severity by flagging only what crossed their bound, so which excess
    flag survives matters less than the floor itself.
    """
    bounded = candidate.copy()
    keys = _cell_keys(frame)
    for _, indices in frame.groupby([frame["machine_id"], keys], sort=False).groups.items():
        group_flags = bounded.loc[indices]
        flagged = int(group_flags.sum())
        allowed = max(0, len(indices) - config.min_cell_survivors)
        if flagged > allowed:
            flagged_indices = group_flags[group_flags].index
            for index in flagged_indices[allowed:]:
                bounded.loc[index] = False
    return bounded


def _mad_outlier_mask(frame: "pd.DataFrame", config: SanitizeConfig) -> "pd.Series":
    """Rule 5: MAD z-score on the log target, per (machine, cell) group.

    A group whose MAD is zero offers no scale to judge deviation against and is skipped rather
    than letting a zero denominator flag every non-identical value.
    """
    import pandas as pd

    log_target = np.log(frame["sampler_window_seconds"].to_numpy(dtype=np.float64))
    keys = _cell_keys(frame)
    mask = pd.Series(False, index=frame.index)
    for _, indices in frame.groupby([frame["machine_id"], keys], sort=False).groups.items():
        if len(indices) < config.mad_min_group:
            continue
        positions = frame.index.get_indexer(indices)
        values = log_target[positions]
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        if mad == 0.0:
            continue
        z_scores = _MAD_TO_SIGMA * np.abs(values - median) / mad
        mask.loc[indices] = z_scores > config.mad_z_bound
    return mask


def _residual_prune_mask(frame: "pd.DataFrame", config: SanitizeConfig) -> "pd.Series":
    """Rule 6: one bounded round of residual pruning against a preliminary LightGBM fit."""
    import pandas as pd

    require("lightgbm", extra="kudos-training", feature="kudos-train sanitize (residual pruning)")
    import lightgbm as lgb

    from hordelib.kudos_training.encoding import frame_to_matrix

    if len(frame) < 50:
        # A preliminary fit on a handful of rows would memorize them; the rule only means
        # something once there is enough data for residuals to reflect structure.
        return pd.Series(False, index=frame.index)

    features = frame_to_matrix(frame, default_manifest())
    log_target = np.log(frame["sampler_window_seconds"].to_numpy(dtype=np.float64))

    model = lgb.LGBMRegressor(
        objective="huber",
        n_estimators=200,
        learning_rate=0.1,
        random_state=config.seed,
        verbose=-1,
    )
    model.fit(features, log_target)
    residuals = np.abs(log_target - model.predict(features))

    median = float(np.median(residuals))
    mad = float(np.median(np.abs(residuals - median)))
    if mad == 0.0:
        return pd.Series(False, index=frame.index)
    z_scores = _MAD_TO_SIGMA * np.abs(residuals - median) / mad

    over_bound = z_scores > config.residual_z_bound
    max_drops = int(len(frame) * config.residual_max_fraction)
    if over_bound.sum() > max_drops:
        # Keep only the worst offenders inside the bounded budget.
        cutoff_positions = np.argsort(z_scores)[::-1][:max_drops]
        bounded = np.zeros(len(frame), dtype=bool)
        bounded[cutoff_positions] = True
        over_bound = bounded
    return pd.Series(over_bound, index=frame.index)


def _write_clean(frame: "pd.DataFrame", *, snapshot_path: Path, out_dir: Path) -> tuple[Path, str]:
    """Write the cleaned frame as a content-hash-named parquet."""
    import hashlib

    out_dir.mkdir(parents=True, exist_ok=True)
    staging_path = out_dir / f"{snapshot_path.stem}.clean.tmp"
    frame.to_parquet(staging_path, engine="pyarrow", index=False)
    content_hash = hashlib.sha256(staging_path.read_bytes()).hexdigest()[:16]
    clean_path = out_dir / f"clean-{content_hash}.parquet"
    staging_path.replace(clean_path)
    return clean_path, content_hash


def _write_report(
    clean_path: Path,
    *,
    snapshot_path: Path,
    config: SanitizeConfig,
    rows_in: int,
    rows_out: int,
    dropped_by_rule: dict[str, int],
    dropped_job_ids: dict[str, list[str]],
) -> None:
    """Write the sanitation report beside the cleaned snapshot."""
    report = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "input_snapshot": snapshot_path.name,
        "rows_in": rows_in,
        "rows_out": rows_out,
        "dropped_by_rule": dropped_by_rule,
        "dropped_job_ids": dropped_job_ids,
        "config": config.__dict__,
        "known_gaps": [
            "rule 4 (concurrent slot occupancy) is absent: the worker does not record queue depth at dispatch",
        ],
    }
    clean_path.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")


__all__ = ["SanitizeConfig", "SanitizeResult", "sanitize"]
