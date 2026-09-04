"""The evaluate stage: a training run in, the per-cell report and v21 comparison out.

Two artifacts decide the rollout conversation and both are produced here: the per-cell
predicted-versus-actual table (the same instrument that diagnosed the v21 mispricing) and the
payment-per-second spread ratio across cells, for the candidate and, when the live npz is
supplied, for v21 on the same rows. The v22 acceptance bar is a spread of at most 1.5 on the
reference machine's test data; v21 measured 7.35 in the field.

The v21 evaluator vendored here reproduces the AI-Horde server's pricing path: the npz MLP
predicts per-image seconds from the v21 hand-written feature vector, and payment composes as
``npz_seconds / basis_seconds x 11 x baseline_multiplier x n_images (+3 per lora, +1 per TI)``.
That reconstruction reproduced 99.6% of 2473 observed payments within 1%.

The candidate's own prices are composed through the policy ledger rather than read off the
prediction, so the per-cell report shows the kudos a job would actually be charged beside the
seconds the model predicts for it. A row whose baseline the ledger does not price is left blank and
counted rather than priced at par, since a silent par would read as a deliberate policy.
"""

import json
from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from hordelib.kudos_training.encoding import frame_to_matrix
from hordelib.kudos_training.ledger import (
    DEFAULT_BASIS_KUDOS,
    KudosPolicyLedger,
    PayloadFeatures,
    PredictedSeconds,
    PricingBasis,
    compose_user_price,
    default_ledger,
)
from hordelib.kudos_training.manifest import KudosFeatureManifest, default_manifest
from hordelib.utils.optional_deps import require

if TYPE_CHECKING:
    import lightgbm as lgb
    import pandas as pd

_MIN_CELL_ROWS = 2
"""Cells with fewer rows than this are excluded from per-cell aggregates and the spread ratio."""

_V21_KUDOS_BASIS = 11.0
"""``KUDOS_BASIS`` (10) plus the basis adjustment (1) the server applies to image jobs."""

_V21_LORA_ADDER = 3.0
_V21_TI_ADDER = 1.0

_V21_POST_PROCESSORS = sorted(
    [
        "4x_AnimeSharp",
        "CodeFormers",
        "GFPGAN",
        "NMKD_Siax",
        "RealESRGAN_x2plus",
        "RealESRGAN_x4plus_anime_6B",
        "RealESRGAN_x4plus",
        "strip_background",
    ],
)
_V21_SAMPLERS = sorted(
    [
        "ddim",
        "k_dpm_2_a",
        "k_dpm_2",
        "k_dpm_adaptive",
        "k_dpm_fast",
        "k_dpmpp_2m",
        "k_dpmpp_2s_a",
        "k_dpmpp_sde",
        "k_euler_a",
        "k_euler",
        "k_heun",
        "k_lms",
        "plms",
        "uni_pc_bh2",
        "uni_pc",
    ],
)
_V21_CONTROL_TYPES = sorted(
    ["canny", "depth", "fakescribbles", "hed", "hough", "None", "normal", "openpose", "scribble", "seg"],
)
_V21_SOURCE_PROCESSING = sorted(["img2img", "inpainting", "outpainting", "txt2img"])

_V21_BASELINE_MULTIPLIERS = {
    "stable_diffusion_xl": 2.0,
    "stable_cascade": 4.0,
    "flux_1": 8.0,
}
"""The server's per-baseline payment multipliers; baselines not listed multiply by 1."""

_V21_BASIS_PAYLOAD = {
    "width": 512,
    "height": 512,
    "steps": 50,
    "cfg_scale": 7.5,
    "denoising_strength": 1.0,
    "karras": True,
    "hires_fix": False,
    "source_image": False,
    "source_mask": False,
    "source_processing": "txt2img",
    "sampler_name": "k_euler",
    "control_type": None,
    "post_processing": [],
}
"""The v21 basis job; every price is expressed relative to its predicted seconds."""


@dataclass(frozen=True)
class EvaluateResult:
    """Headline numbers of one evaluation, with the full tables written into the run directory."""

    report_path: Path
    candidate_spread: float | None
    v21_spread: float | None
    test_median_ape: float | None
    ledger_version: str
    """Policy ledger revision the composed prices in the report were priced under."""

    rows_without_ledger_baseline: int
    """Rows the ledger could not price because it carries no premium for their baseline."""


def _float_or(value: Any, default: float) -> float:
    """Return *value* as a finite float, or *default* when it is absent, NaN, or infinite."""
    if value is None:
        return default
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        return default
    return as_float if np.isfinite(as_float) else default


class _V21Model:
    """The live v21 npz MLP and the hand-written encoding it expects."""

    def __init__(self, npz_path: Path) -> None:
        with np.load(npz_path) as loaded:
            self._layers = tuple(
                (loaded[f"w{index}"].astype(np.float32), loaded[f"b{index}"].astype(np.float32)) for index in range(4)
            )
        self._basis_seconds = self.predict_seconds(_V21_BASIS_PAYLOAD)

    def predict_seconds(self, payload: dict[str, Any]) -> float:
        """Run the npz forward pass on one payload, returning predicted per-image seconds."""
        vector = self._encode(payload)
        for weights, biases in self._layers[:-1]:
            vector = np.maximum(vector @ weights.T + biases, 0)
        weights, biases = self._layers[-1]
        return float((vector @ weights.T + biases).item())

    def price_row(self, row: Mapping[Hashable, Any]) -> float:
        """Price one snapshot row the way the server pays for it."""
        karras = row["scheduler"] == "karras"
        source_processing = row["source_processing"] if row["source_processing"] is not None else "txt2img"
        payload = {
            "width": row["width"],
            "height": row["height"],
            "steps": row["trajectory_steps"],
            # Parquet round-trips absent floats as NaN rather than None, so both spellings of
            # "not recorded" must fall to the server-side defaults.
            "cfg_scale": _float_or(row["cfg_scale"], 7.5),
            "denoising_strength": _float_or(row["denoising_strength"], 1.0),
            "karras": karras,
            "hires_fix": row["hires_fix"],
            "source_image": source_processing in ("img2img", "inpainting"),
            "source_mask": source_processing == "inpainting",
            "source_processing": source_processing,
            "sampler_name": row["sampler_name"],
            "control_type": row["control_type"],
            "post_processing": list(row["post_processing"]) if row["post_processing"] is not None else [],
        }
        per_image = self.predict_seconds(payload) / self._basis_seconds * _V21_KUDOS_BASIS
        multiplier = _V21_BASELINE_MULTIPLIERS.get(row["baseline"] or "", 1.0)
        price = per_image * row["n_images"] * multiplier
        price += _V21_LORA_ADDER * row["loras_count"] + _V21_TI_ADDER * row["tis_count"]
        return price

    def _encode(self, payload: dict[str, Any]) -> np.ndarray:
        """Reproduce the v21 hand-written feature vector."""
        denoising_strength = 1.0
        control_strength = 1.0
        has_source_image = bool(payload.get("source_image"))
        has_control_type = bool(payload.get("control_type"))
        if has_source_image:
            denoising_strength = float(payload.get("denoising_strength", 1.0))
            if has_control_type:
                control_strength = float(payload.get("control_strength", denoising_strength))
                denoising_strength = 1.0
        floats = np.asarray(
            [
                payload["height"] / 1024,
                payload["width"] / 1024,
                payload["steps"] / 100,
                payload["cfg_scale"] / 30,
                denoising_strength,
                control_strength,
                1.0 if payload.get("karras") else 0.0,
                1.0 if payload.get("hires_fix") else 0.0,
                1.0 if payload.get("source_image") else 0.0,
                1.0 if payload.get("source_mask") else 0.0,
            ],
            dtype=np.float32,
        )
        sampler = payload.get("sampler_name") or "k_euler"
        if sampler not in _V21_SAMPLERS:
            sampler = "k_euler"
        control_type = payload.get("control_type") or "None"
        if control_type not in _V21_CONTROL_TYPES:
            control_type = "None"
        source_processing = payload.get("source_processing", "txt2img")
        if source_processing not in _V21_SOURCE_PROCESSING:
            source_processing = "img2img"
        post_processing = [name for name in payload.get("post_processing", []) if name in _V21_POST_PROCESSORS]

        vector = np.zeros(
            len(floats)
            + len(_V21_SAMPLERS)
            + len(_V21_CONTROL_TYPES)
            + len(_V21_SOURCE_PROCESSING)
            + len(_V21_POST_PROCESSORS),
            dtype=np.float32,
        )
        vector[: len(floats)] = floats
        offset = len(floats)
        vector[offset + _V21_SAMPLERS.index(sampler)] = 1.0
        offset += len(_V21_SAMPLERS)
        vector[offset + _V21_CONTROL_TYPES.index(control_type)] = 1.0
        offset += len(_V21_CONTROL_TYPES)
        vector[offset + _V21_SOURCE_PROCESSING.index(source_processing)] = 1.0
        offset += len(_V21_SOURCE_PROCESSING)
        for name in post_processing:
            vector[offset + _V21_POST_PROCESSORS.index(name)] += 1.0
        return vector


def evaluate(
    run_dir: Path,
    clean_path: Path,
    *,
    manifest: KudosFeatureManifest | None = None,
    against_npz: Path | None = None,
    ledger: KudosPolicyLedger | None = None,
) -> EvaluateResult:
    """Evaluate a training run: per-cell table, spread ratios, and the v21 comparison.

    Args:
        run_dir: A train-stage run directory (``model.txt`` + ``splits.json``).
        clean_path: The cleaned snapshot the run trained from.
        manifest: Feature manifest the run encoded against. Defaults to the shipped revision.
        against_npz: The live v21 npz; when given, v21 prices the same rows for comparison.
        ledger: Policy ledger the composed prices are charged under. Defaults to the shipped
            revision.

    Returns:
        Headline numbers; the full report lands in ``run_dir/evaluation.json`` and the per-cell
        table in ``run_dir/per_cell.csv``.
    """
    require("pandas", extra="kudos-training", feature="kudos-train evaluate")
    require("lightgbm", extra="kudos-training", feature="kudos-train evaluate")
    import lightgbm as lgb
    import pandas as pd

    active_manifest = manifest if manifest is not None else default_manifest()
    active_ledger = ledger if ledger is not None else default_ledger()

    frame = pd.read_parquet(clean_path)
    splits = json.loads((run_dir / "splits.json").read_text(encoding="utf-8"))
    frame["split"] = frame["job_id"].map(splits)

    booster = lgb.Booster(model_file=str(run_dir / "model.txt"))
    features = frame_to_matrix(frame, active_manifest)
    frame["predicted_seconds"] = np.exp(np.asarray(booster.predict(features), dtype=np.float64))
    frame["ape"] = (frame["predicted_seconds"] - frame["sampler_window_seconds"]).abs() / frame[
        "sampler_window_seconds"
    ]
    # The candidate prices resource-seconds directly, so its payment-per-actual-second is the
    # prediction ratio; a perfectly calibrated model would hold it at 1.0 for every cell.
    frame["candidate_pay_per_second"] = frame["predicted_seconds"] / frame["sampler_window_seconds"]

    basis = PricingBasis(
        basis_seconds=_predicted_basis_seconds(booster, active_manifest),
        basis_kudos=DEFAULT_BASIS_KUDOS,
    )
    ledger_prices, rows_without_ledger_baseline = _ledger_prices(frame, active_ledger, basis)
    frame["ledger_price_kudos"] = ledger_prices
    frame["ledger_pay_per_second"] = frame["ledger_price_kudos"] / frame["sampler_window_seconds"]

    v21_model = _V21Model(against_npz) if against_npz is not None else None
    if v21_model is not None:
        frame["v21_price"] = [v21_model.price_row(row) for row in frame.to_dict(orient="records")]
        frame["v21_pay_per_second"] = frame["v21_price"] / frame["sampler_window_seconds"]

    per_cell = _per_cell_table(frame)
    per_cell.to_csv(run_dir / "per_cell.csv", index=False)

    candidate_spread = _spread_ratio(per_cell["candidate_pay_per_second"])
    v21_spread = _spread_ratio(per_cell["v21_pay_per_second"]) if v21_model is not None else None

    split_metrics = {
        str(split_name): {
            "rows": int(len(split_frame)),
            "median_ape": float(split_frame["ape"].median()),
            "p90_ape": float(split_frame["ape"].quantile(0.90)),
        }
        for split_name, split_frame in frame.groupby("split")
    }
    test_median_ape = split_metrics.get("test", {}).get("median_ape")

    report = {
        "run_dir": run_dir.name,
        "clean_snapshot": clean_path.name,
        "rows": int(len(frame)),
        "cells_in_spread": int(per_cell["cell_id"].notna().sum()),
        "ledger_version": active_ledger.ledger_version,
        "ledger_reference_machine": active_ledger.reference_machine,
        "basis_seconds": basis.basis_seconds,
        "basis_kudos": basis.basis_kudos,
        "rows_without_ledger_baseline": rows_without_ledger_baseline,
        "candidate_pay_per_second_spread": candidate_spread,
        "v21_pay_per_second_spread": v21_spread,
        "acceptance_target_spread": 1.5,
        "split_metrics": split_metrics,
        "notes": [
            "spread ratios are computed over all cleaned rows (per-cell medians), not the test slice alone: "
            "the standard corpus has ~3 replicates per cell, so a per-split per-cell table would be empty",
            "ledger_price_kudos is what a job would be charged under the policy ledger, composed from the "
            "predicted seconds; it is a policy output and not a model metric",
        ],
    }
    report_path = run_dir / "evaluation.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return EvaluateResult(
        report_path=report_path,
        candidate_spread=candidate_spread,
        v21_spread=v21_spread,
        test_median_ape=test_median_ape,
        ledger_version=active_ledger.ledger_version,
        rows_without_ledger_baseline=rows_without_ledger_baseline,
    )


def _predicted_basis_seconds(booster: "lgb.Booster", manifest: KudosFeatureManifest) -> float:
    """Return the candidate model's predicted seconds for the manifest's basis job.

    The horde's price scale is anchored on one reference job rather than on an absolute rate, so a
    candidate that is uniformly faster than v21 must not deflate every price; taking the anchor from
    the candidate itself keeps the scale fixed while the model changes underneath it.
    """
    vector = manifest.to_vector(manifest.basis_payload).reshape(1, -1)
    return float(np.exp(np.asarray(booster.predict(vector), dtype=np.float64)).item())


def _ledger_prices(
    frame: "pd.DataFrame",
    ledger: KudosPolicyLedger,
    basis: PricingBasis,
) -> tuple[list[float], int]:
    """Compose the ledger price of every row, blanking the rows the ledger cannot price.

    Args:
        frame: Rows carrying a ``predicted_seconds`` column.
        ledger: The policy ledger to price under.
        basis: The seconds-to-kudos anchor.

    Returns:
        One price per row in frame order (NaN where the ledger prices no such baseline), and how
        many rows were left unpriced.
    """
    prices: list[float] = []
    unpriced = 0
    for row in frame.to_dict(orient="records"):
        baseline = row["baseline"]
        if not ledger.knows_baseline(baseline):
            prices.append(float("nan"))
            unpriced += 1
            continue
        breakdown = compose_user_price(
            PredictedSeconds(sampler_window=float(row["predicted_seconds"])),
            PayloadFeatures(
                baseline=str(baseline),
                model_name=row["model_name"],
                loras_count=int(row["loras_count"]),
                tis_count=int(row["tis_count"]),
            ),
            ledger,
            basis,
        )
        prices.append(breakdown.total_kudos)
    return prices, unpriced


def _per_cell_table(frame: "pd.DataFrame") -> "pd.DataFrame":
    """Aggregate predicted-versus-actual per corpus cell, over all cleaned rows."""
    cells = frame[frame["cell_id"].notna()]
    aggregations = {
        "rows": ("job_id", "size"),
        "actual_median_seconds": ("sampler_window_seconds", "median"),
        "predicted_median_seconds": ("predicted_seconds", "median"),
        "median_ape": ("ape", "median"),
        "ledger_price_kudos": ("ledger_price_kudos", "median"),
        "ledger_pay_per_second": ("ledger_pay_per_second", "median"),
        "candidate_pay_per_second": ("candidate_pay_per_second", "median"),
    }
    if "v21_pay_per_second" in frame.columns:
        aggregations["v21_pay_per_second"] = ("v21_pay_per_second", "median")
    table = cells.groupby("cell_id").agg(**aggregations).reset_index()
    return table[table["rows"] >= _MIN_CELL_ROWS].sort_values("candidate_pay_per_second")


def _spread_ratio(pay_per_second: "pd.Series") -> float | None:
    """Return max/min of the per-cell payment-per-second medians, the headline fairness number."""
    values = pay_per_second.dropna()
    if len(values) < 2 or float(values.min()) <= 0:
        return None
    return float(values.max()) / float(values.min())


__all__ = ["EvaluateResult", "evaluate"]
