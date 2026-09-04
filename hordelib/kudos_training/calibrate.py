"""The calibrate stage: measurements from several machines in, reference-machine seconds out.

A price is charged in one currency, so the model is trained on one machine's clock: measurements
taken elsewhere are only admissible once they are expressed in that machine's seconds. One machine
is declared the reference, and every other machine is mapped onto it through the cells both have
measured, which are the only common ruler available.

The map is two parameters wide. A shape model is fitted on reference rows alone, and each other
machine's overlap rows are regressed against its predictions as ``log t_m = a_m + b_m * f_hat(x)``:
``a_m`` is the machine's constant offset and ``b_m`` how its cost scales with the work. Whatever an
affine map cannot absorb shows up as residual spread, which is what the bar tests. A machine whose
residuals are wide is not describable as a scaled reference machine, so its data is refused rather
than averaged in, where it would move prices without ever being visible as hardware.

Cells the reference machine never ran are carried through as ``out_of_regime``: their mapping
extrapolates the fit instead of interpolating within it. They still enter training, being the only
measurements of those cells, and the label keeps that visible to every later stage.

Columns this stage adds to a cleaned frame, which the snapshot row model does not describe because
cleaned frames are not validated through it:

* ``measured_machine_id``: the machine that took the measurement, kept after mapping.
* ``measured_seconds``: the machine's own clock reading, before mapping.
* ``out_of_regime``: the reference machine never ran this row's cell.
* ``calibrated``: the row's target is expressed in reference-machine seconds.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from hordelib.kudos_training.ledger import KudosPolicyLedger, default_ledger
from hordelib.kudos_training.manifest import KudosFeatureManifest, default_manifest
from hordelib.kudos_training.train import TrainConfig, build_reference_estimator, prepare_features_and_target
from hordelib.utils.optional_deps import require

DEFAULT_SPREAD_BAR = 1.5
"""Default bound on a machine's residual spread, matching the pricing fairness bar."""

MINIMUM_OVERLAP_CELLS = 5
"""Fewer shared cells than this cannot separate a machine's offset from its scaling."""

_RESIDUAL_QUANTILES = (0.05, 0.95)
"""The residual band the spread is measured over; the tails are single jobs, not machine behaviour."""

_MINIMUM_SLOPE = 1e-9
"""Below this the fitted map cannot be inverted, so the machine's rows cannot be mapped at all."""


class CalibrationError(RuntimeError):
    """Raised when a frame cannot be calibrated onto the reference machine at all."""


@dataclass(frozen=True)
class MachineCalibration:
    """The fitted map from one machine's seconds onto the reference machine's."""

    machine_id: str
    n_overlap_cells: int
    n_overlap_rows: int
    intercept: float
    """``a_m``: the machine's constant offset in log seconds."""

    slope: float
    """``b_m``: how the machine's log seconds scale with the reference model's prediction."""

    residual_spread: float
    """``exp(p95 - p05)`` of the fit residuals: the factor the map fails to explain."""

    passes: bool
    rows: int
    out_of_regime_rows: int


@dataclass(frozen=True)
class CalibrationResult:
    """One calibration pass: the report, and the mapped frame when every machine passed."""

    report_path: Path
    calibrated_path: Path | None
    content_hash: str
    reference_machine: str
    bar: float
    machines: tuple[MachineCalibration, ...]
    passed: bool


def calibrate(
    clean_path: Path,
    *,
    out_dir: Path,
    reference_machine: str | None = None,
    bar: float = DEFAULT_SPREAD_BAR,
    manifest: KudosFeatureManifest | None = None,
    config: TrainConfig | None = None,
    ledger: KudosPolicyLedger | None = None,
) -> CalibrationResult:
    """Map every machine's measurements in a cleaned snapshot onto the reference machine.

    Args:
        clean_path: A sanitized snapshot parquet, possibly spanning several machines.
        out_dir: Directory the report and the calibrated snapshot are written into.
        reference_machine: The machine every other is mapped onto. Defaults to the policy ledger's
            reference machine, which is the machine the served prices are anchored to.
        bar: Largest residual spread a machine may show and still be admitted.
        manifest: Feature manifest the shape model encodes against. Defaults to the shipped revision.
        config: Shape-model hyperparameters. Defaults to :class:`TrainConfig` defaults.
        ledger: Policy ledger the default reference machine is read from.

    Returns:
        The report path, the calibrated snapshot when every machine passed, and the per-machine fits.

    Raises:
        CalibrationError: If the reference machine has no rows, or a machine shares too few cells
            with the reference machine to fit a map.
        ValueError: If the snapshot still carries missing or non-positive targets.
    """
    require("pandas", extra="kudos-training", feature="kudos-train calibrate")
    require("pyarrow", extra="kudos-training", feature="kudos-train calibrate")
    import pandas as pd

    active_manifest = manifest if manifest is not None else default_manifest()
    active_config = config if config is not None else TrainConfig()
    reference_id = reference_machine if reference_machine is not None else _default_reference_machine(ledger)

    frame = pd.read_parquet(clean_path)
    features, log_target = prepare_features_and_target(frame, active_manifest)
    machine_ids = frame["machine_id"].astype(str).to_numpy()

    reference_mask = machine_ids == reference_id
    if not reference_mask.any():
        present = ", ".join(sorted(set(machine_ids)))
        raise CalibrationError(
            f"{clean_path} carries no rows measured on reference machine {reference_id!r} (present: {present})",
        )

    shape_model = build_reference_estimator(active_manifest, active_config)
    shape_model.fit(features[reference_mask], log_target[reference_mask])
    predicted = np.asarray(shape_model.predict(features), dtype=np.float64)

    cell_ids = frame["cell_id"].astype(object).to_numpy()
    reference_cells = {cell for cell in cell_ids[reference_mask] if isinstance(cell, str)}

    in_reference_regime = np.array(
        [isinstance(cell, str) and cell in reference_cells for cell in cell_ids],
        dtype=bool,
    )

    mapped_seconds = np.exp(log_target)
    out_of_regime = np.zeros(len(frame), dtype=bool)
    calibrations: list[MachineCalibration] = []
    for machine_id in sorted(set(machine_ids) - {reference_id}):
        machine_mask = machine_ids == machine_id
        overlap_mask = machine_mask & in_reference_regime
        overlap_cells = {cell for cell in cell_ids[overlap_mask] if isinstance(cell, str)}
        if len(overlap_cells) < MINIMUM_OVERLAP_CELLS:
            raise CalibrationError(
                f"machine {machine_id!r} shares {len(overlap_cells)} cells with reference machine "
                f"{reference_id!r}; at least {MINIMUM_OVERLAP_CELLS} are needed to fit a map",
            )

        slope, intercept = np.polyfit(predicted[overlap_mask], log_target[overlap_mask], 1)
        residuals = log_target[overlap_mask] - (intercept + slope * predicted[overlap_mask])
        low, high = np.quantile(residuals, _RESIDUAL_QUANTILES)
        spread = float(np.exp(high - low))

        out_of_regime[machine_mask & ~in_reference_regime] = True
        if abs(slope) >= _MINIMUM_SLOPE:
            mapped_seconds[machine_mask] = np.exp((log_target[machine_mask] - intercept) / slope)
        else:
            mapped_seconds[machine_mask] = np.nan

        calibrations.append(
            MachineCalibration(
                machine_id=machine_id,
                n_overlap_cells=len(overlap_cells),
                n_overlap_rows=int(overlap_mask.sum()),
                intercept=float(intercept),
                slope=float(slope),
                residual_spread=spread,
                passes=bool(spread <= bar and abs(slope) >= _MINIMUM_SLOPE),
                rows=int(machine_mask.sum()),
                out_of_regime_rows=int((machine_mask & ~in_reference_regime).sum()),
            ),
        )

    calibrated_frame = frame.copy()
    calibrated_frame["measured_machine_id"] = machine_ids
    calibrated_frame["measured_seconds"] = np.exp(log_target)
    calibrated_frame["out_of_regime"] = out_of_regime
    calibrated_frame["calibrated"] = True
    calibrated_frame["sampler_window_seconds"] = mapped_seconds

    passed = all(calibration.passes for calibration in calibrations)
    out_dir.mkdir(parents=True, exist_ok=True)
    staging_path = out_dir / "calibrated.parquet.tmp"
    calibrated_frame.to_parquet(staging_path, engine="pyarrow", index=False)
    content_hash = hashlib.sha256(staging_path.read_bytes()).hexdigest()[:16]

    calibrated_path: Path | None = None
    if passed:
        calibrated_path = out_dir / f"calibrated-{content_hash}.parquet"
        staging_path.replace(calibrated_path)
    else:
        staging_path.unlink()

    report: dict[str, Any] = {
        "clean_snapshot": clean_path.name,
        "content_hash": content_hash,
        "reference_machine": reference_id,
        "bar": bar,
        "passed": passed,
        "rows": int(len(frame)),
        "reference_rows": int(reference_mask.sum()),
        "reference_cells": len(reference_cells),
        "machines": [calibration.__dict__ for calibration in calibrations],
    }
    report_path = out_dir / f"calibration-{content_hash}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return CalibrationResult(
        report_path=report_path,
        calibrated_path=calibrated_path,
        content_hash=content_hash,
        reference_machine=reference_id,
        bar=bar,
        machines=tuple(calibrations),
        passed=passed,
    )


def format_calibration_table(result: CalibrationResult) -> str:
    """Render the per-machine fits as aligned text, one row per non-reference machine."""
    header = ("machine_id", "cells", "rows", "a_m", "b_m", "spread", "out_of_regime", "passes")
    rows = [
        (
            calibration.machine_id,
            str(calibration.n_overlap_cells),
            str(calibration.n_overlap_rows),
            f"{calibration.intercept:.4f}",
            f"{calibration.slope:.4f}",
            f"{calibration.residual_spread:.3f}",
            str(calibration.out_of_regime_rows),
            "yes" if calibration.passes else "NO",
        )
        for calibration in result.machines
    ]
    if not rows:
        return f"no machines besides reference {result.reference_machine}"
    widths = [max(len(row[index]) for row in (header, *rows)) for index in range(len(header))]
    return "\n".join(
        "  ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=True)).rstrip()
        for row in (header, *rows)
    )


def _default_reference_machine(ledger: KudosPolicyLedger | None) -> str:
    """Return the machine the served prices are anchored to."""
    active_ledger = ledger if ledger is not None else default_ledger()
    return str(active_ledger.reference_machine)


__all__ = [
    "DEFAULT_SPREAD_BAR",
    "MINIMUM_OVERLAP_CELLS",
    "CalibrationError",
    "CalibrationResult",
    "MachineCalibration",
    "calibrate",
    "format_calibration_table",
]
