"""Behavioural tests for the kudos-training pipeline stages.

Everything runs from synthetic fixtures on a temporary directory: no network, no GPU, no real
stats. The assemble tests exercise the pairing contract (pop-order sorting, missing records, axis
divergence); the sanitize tests exercise each rule against rows crafted to trip exactly it; the
train/evaluate test drives the remaining stages end to end on a corpus small enough to fit in CI.
"""

import hashlib
import json
import math
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
from horde_sdk.generation_parameters.image.sampler_work import SamplerExecutionContractVersion
from loguru import logger

from hordelib.kudos_training.assemble import AssemblyError, PairingError, assemble
from hordelib.kudos_training.sanitize import SanitizeConfig, sanitize
from hordelib.kudos_training.schema import SnapshotRow, SourceKind
from hordelib.kudos_training.train import TrainConfig, train

pd = pytest.importorskip("pandas")

MACHINE_ID = "test-rig"
OTHER_MANIFEST_SHA256 = "4d1b" + "0" * 60
"""A well-formed manifest hash that is not the shipped revision's."""
SAMPLER_CONSTRAINTS_ARTIFACT_SHA256 = "b46f7becf7ea5583ea0a7fe8ba1253e21b528b0ebda5b15e78445ef79e2e8533"


@pytest.fixture()
def machines_path(tmp_path: Path) -> Path:
    path = tmp_path / "machines.toml"
    path.write_text(f'[machines.{MACHINE_ID}]\ngpu_model = "Test GPU"\nvram_mb = 1\n', encoding="utf-8")
    return path


def _cell(cell_id: str, **overrides: Any) -> dict[str, Any]:
    cell: dict[str, Any] = {
        "cell_id": cell_id,
        "group": cell_id.split(".", 1)[0],
        "model": "Model A",
        "width": 512,
        "height": 512,
        "steps": 30,
        "cfg_scale": 7.5,
        "n_iter": 1,
        "sampler_name": "k_euler",
        "scheduler": "karras",
        "source_processing": "txt2img",
        "replicates": 3,
    }
    cell.update(overrides)
    return cell


def _definition(
    cells: list[dict[str, Any]], jobs: list[dict[str, Any]], *, warmup_job_count: int = 1
) -> dict[str, Any]:
    return {
        "scenario_name": "pricing-corpus",
        "scenario_revision": "1",
        "tier": "smoke",
        "warmup_job_count": warmup_job_count,
        "shuffle_seeds": ["seed-a"],
        "prompts": ["a prompt"],
        "cells": cells,
        "jobs": jobs,
        "same_model_adjacencies": 0,
        "post_processing_proximities": 0,
    }


def _job(position: int, cell: dict[str, Any], replicate: int) -> dict[str, Any]:
    return {
        "position": position,
        "cell_id": cell["cell_id"],
        "group": cell["group"],
        "permutation": "warmup" if cell["group"] == "warmup" else "seed-a",
        "replicate": replicate,
        "seed": f"pc1:{cell['cell_id']}:{replicate}",
        "prompt_index": 0,
        "model": cell["model"],
    }


def _record(cell: dict[str, Any], *, popped: float, window: float = 10.0, faulted: bool = False) -> dict[str, Any]:
    inference_start = popped + 1.0
    return {
        "event": "job_completed",
        "baseline": "stable_diffusion_1",
        "job": {
            "job_id": f"job-{popped:.0f}",
            "is_alchemy": False,
            "faulted": faulted,
            "time_popped": popped,
            "stage_timestamps": {
                "PENDING_INFERENCE": popped,
                "INFERENCE_IN_PROGRESS": inference_start,
                "PENDING_SAFETY_CHECK": inference_start + window,
                "FINALIZED": inference_start + window + 1.0,
            },
            "queue_wait_seconds": 1.0,
            "e2e_seconds": window + 2.0,
            "safety_seconds": 0.5,
            "model_name": cell["model"],
            "steps": cell["steps"],
            "width": cell["width"],
            "height": cell["height"],
            "loras_count": len(cell.get("lora_version_ids", [])),
            "tis_count": len(cell.get("ti_names", [])),
            "control_type": cell.get("control_type"),
            "post_processing": cell.get("post_processing", []),
            "sampler_name": cell["sampler_name"],
            "scheduler": cell["scheduler"],
            "cfg_scale": cell["cfg_scale"],
            "hires_fix": cell.get("hires_fix", False),
            "batch_count": cell["n_iter"],
            "megapixelsteps": 7.8,
            "sampling_seconds": window * 0.6,
            "kudos_reward": 0.0,
        },
    }


def _write_stats(
    path: Path,
    records: list[dict[str, Any]],
    *,
    scenario_id: str | None = "pricing-corpus",
    execution_contract_version: SamplerExecutionContractVersion | None = SamplerExecutionContractVersion.V1,
) -> None:
    config: dict[str, Any] = {
        "max_threads": 1,
        "horde_sdk_version": "0.29.0",
        "sampler_constraints_artifact_sha256": SAMPLER_CONSTRAINTS_ARTIFACT_SHA256,
        "sampler_execution_contract_version": execution_contract_version,
    }
    if scenario_id is not None:
        config["scenario_id"] = scenario_id
        config["scenario_revision"] = "1"
    session_start = {"event": "session_start", "worker_version": "17.8.5", "timestamp": 1000.0, "config": config}
    session_end = {"event": "session_end", "worker_version": "17.8.5", "timestamp": 9999.0}
    lines = [session_start, *records, session_end]
    path.write_text("\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8")


def _standard_fixture(tmp_path: Path) -> tuple[Path, Path]:
    """Two cells x three replicates plus a warmup job, with the record stream out of pop order."""
    warmup_cell = _cell("warmup.a", group="warmup", replicates=1)
    cell_a = _cell("g1.fast")
    cell_b = _cell(
        "g5.pp",
        model="Model B",
        source_processing="img2img",
        denoising_strength=0.65,
        post_processing=["RealESRGAN_x4plus"],
    )
    cells = [warmup_cell, cell_a, cell_b]

    jobs = [_job(0, warmup_cell, 0)]
    records = [_record(warmup_cell, popped=1000.0)]
    popped = 1100.0
    position = 1
    for replicate in range(3):
        for cell in (cell_a, cell_b):
            jobs.append(_job(position, cell, replicate))
            window = 10.0 if cell is cell_a else 25.0
            records.append(_record(cell, popped=popped, window=window + replicate))
            position += 1
            popped += 100.0

    # A post-processing job finalizes late, so its record lands after its successor's.
    records[2], records[3] = records[3], records[2]

    definition_path = tmp_path / "definition.json"
    definition_path.write_text(json.dumps(_definition(cells, jobs)), encoding="utf-8")
    stats_path = tmp_path / "stats.jsonl"
    _write_stats(stats_path, records)
    return stats_path, definition_path


def test_assemble_pairs_labels_and_windows(tmp_path: Path, machines_path: Path) -> None:
    stats_path, definition_path = _standard_fixture(tmp_path)

    result = assemble(
        [stats_path],
        machine_id=MACHINE_ID,
        out_dir=tmp_path / "snapshots",
        definition_paths=[definition_path],
        machines_path=machines_path,
        resolve_baselines=False,
    )

    assert result.total_rows == 7
    assert result.sessions[0].source_kind is SourceKind.CORPUS
    assert result.sessions[0].missing_positions == ()
    assert result.snapshot_path.exists()
    assert result.content_hash in result.snapshot_path.name

    frame = pd.read_parquet(result.snapshot_path)
    assert list(frame["cell_id"].dropna().unique()) == ["warmup.a", "g1.fast", "g5.pp"]
    assert int(frame["warmup"].sum()) == 1
    # The swapped record stream must still pair by pop order: every row's model matches its cell.
    by_position = frame.set_index("position")
    assert by_position.loc[1, "model_name"] == "Model A"
    assert by_position.loc[2, "model_name"] == "Model B"
    # Window = PENDING_SAFETY_CHECK - INFERENCE_IN_PROGRESS from the fixture's construction.
    assert by_position.loc[1, "sampler_window_seconds"] == pytest.approx(10.0)
    assert by_position.loc[2, "sampler_window_seconds"] == pytest.approx(25.0)
    assert by_position.loc[1, "trajectory_steps"] == 30
    assert by_position.loc[1, "sampler_execution_contract_version"] == SamplerExecutionContractVersion.V1.value
    # The cell spec's denoising strength rides along on corpus rows.
    assert by_position.loc[2, "denoising_strength"] == pytest.approx(0.65)


def test_assemble_tolerates_a_missing_record(tmp_path: Path, machines_path: Path) -> None:
    stats_path, definition_path = _standard_fixture(tmp_path)
    lines = stats_path.read_text(encoding="utf-8").splitlines()
    # Drop position 2's record (a Model B row mid-stream); its neighbours still pair.
    del lines[4]
    stats_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    result = assemble(
        [stats_path],
        machine_id=MACHINE_ID,
        out_dir=tmp_path / "snapshots",
        definition_paths=[definition_path],
        machines_path=machines_path,
        resolve_baselines=False,
    )

    assert result.total_rows == 6
    assert result.sessions[0].missing_positions == (2,)


def test_assemble_reads_rotated_parts_as_one_session(tmp_path: Path, machines_path: Path) -> None:
    stats_path, definition_path = _standard_fixture(tmp_path)
    lines = stats_path.read_text(encoding="utf-8").splitlines()
    split = len(lines) // 2
    # The worker's naming scheme is what the assembler keys on: one session stamp, numbered parts.
    first = tmp_path / "stats-v0.0.0-20000101-000000-000.jsonl"
    second = tmp_path / "stats-v0.0.0-20000101-000000-001.jsonl"
    first.write_text("\n".join(lines[:split]) + "\n", encoding="utf-8")
    second.write_text("\n".join(lines[split:]) + "\n", encoding="utf-8")
    stats_path.unlink()

    result = assemble(
        [second],
        machine_id=MACHINE_ID,
        out_dir=tmp_path / "out",
        definition_paths=[definition_path],
        machines_path=machines_path,
        resolve_baselines=False,
    )

    assert len(result.sessions) == 1
    assert result.sessions[0].stats_file == first.name
    assert result.sessions[0].missing_positions == ()
    frame = pd.read_parquet(result.snapshot_path)
    assert set(frame["stats_file"]) == {first.name}
    assert len(frame) == result.total_rows


def test_assemble_raises_on_axis_divergence(tmp_path: Path, machines_path: Path) -> None:
    stats_path, definition_path = _standard_fixture(tmp_path)
    lines = stats_path.read_text(encoding="utf-8").splitlines()
    diverged = json.loads(lines[2])
    diverged["job"]["steps"] = 99
    diverged["job"]["model_name"] = "Model C"
    lines[2] = json.dumps(diverged)
    stats_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(PairingError):
        assemble(
            [stats_path],
            machine_id=MACHINE_ID,
            out_dir=tmp_path / "snapshots",
            definition_paths=[definition_path],
            machines_path=machines_path,
            resolve_baselines=False,
        )


def test_assemble_requires_definition_for_corpus_sessions(tmp_path: Path, machines_path: Path) -> None:
    stats_path, _ = _standard_fixture(tmp_path)
    with pytest.raises(AssemblyError, match="definition"):
        assemble(
            [stats_path],
            machine_id=MACHINE_ID,
            out_dir=tmp_path / "snapshots",
            machines_path=machines_path,
            resolve_baselines=False,
        )


def test_assemble_labels_production_sessions(tmp_path: Path, machines_path: Path) -> None:
    cell = _cell("unused")
    records = [_record(cell, popped=1000.0 + offset * 100) for offset in range(3)]
    stats_path = tmp_path / "production.jsonl"
    _write_stats(stats_path, records, scenario_id=None)

    result = assemble(
        [stats_path],
        machine_id=MACHINE_ID,
        out_dir=tmp_path / "snapshots",
        machines_path=machines_path,
        resolve_baselines=False,
    )

    assert result.sessions[0].source_kind is SourceKind.PRODUCTION
    frame = pd.read_parquet(result.snapshot_path)
    assert frame["cell_id"].isna().all()
    assert (frame["source_kind"] == "production").all()


def test_unknown_machine_id_is_rejected(tmp_path: Path, machines_path: Path) -> None:
    stats_path, definition_path = _standard_fixture(tmp_path)
    with pytest.raises(AssemblyError, match="unknown machine id"):
        assemble(
            [stats_path],
            machine_id="not-a-machine",
            out_dir=tmp_path / "snapshots",
            definition_paths=[definition_path],
            machines_path=machines_path,
            resolve_baselines=False,
        )


@contextmanager
def _captured_warnings() -> Iterator[list[str]]:
    """Collect warning-level log lines; the pipeline logs through loguru, which pytest does not see."""
    messages: list[str] = []
    sink_id = logger.add(lambda message: messages.append(str(message)), level="WARNING", format="{message}")
    try:
        yield messages
    finally:
        logger.remove(sink_id)


def _stamp_manifest_revision(definition_path: Path, manifest_sha256: str) -> None:
    """Stamp a definition artifact with the feature manifest revision its cells were encoded under."""
    artifact = json.loads(definition_path.read_text(encoding="utf-8"))
    artifact["manifest_sha256"] = manifest_sha256
    definition_path.write_text(json.dumps(artifact), encoding="utf-8")


def test_assemble_accepts_a_definition_stamped_with_the_shipped_manifest(
    tmp_path: Path,
    machines_path: Path,
) -> None:
    from hordelib.kudos_training import default_manifest

    stats_path, definition_path = _standard_fixture(tmp_path)
    _stamp_manifest_revision(definition_path, default_manifest().content_sha256())

    result = assemble(
        [stats_path],
        machine_id=MACHINE_ID,
        out_dir=tmp_path / "snapshots",
        definition_paths=[definition_path],
        machines_path=machines_path,
        resolve_baselines=False,
    )
    assert result.total_rows == 7


def test_assemble_refuses_a_definition_from_another_manifest_revision(
    tmp_path: Path,
    machines_path: Path,
) -> None:
    stats_path, definition_path = _standard_fixture(tmp_path)
    _stamp_manifest_revision(definition_path, OTHER_MANIFEST_SHA256)

    with pytest.raises(AssemblyError, match="another manifest revision"):
        assemble(
            [stats_path],
            machine_id=MACHINE_ID,
            out_dir=tmp_path / "snapshots",
            definition_paths=[definition_path],
            machines_path=machines_path,
            resolve_baselines=False,
        )


def test_assemble_downgrades_a_manifest_mismatch_on_request(tmp_path: Path, machines_path: Path) -> None:
    stats_path, definition_path = _standard_fixture(tmp_path)
    _stamp_manifest_revision(definition_path, OTHER_MANIFEST_SHA256)

    with _captured_warnings() as warnings:
        result = assemble(
            [stats_path],
            machine_id=MACHINE_ID,
            out_dir=tmp_path / "snapshots",
            definition_paths=[definition_path],
            machines_path=machines_path,
            resolve_baselines=False,
            allow_manifest_mismatch=True,
        )

    assert result.total_rows == 7
    assert any(OTHER_MANIFEST_SHA256 in message for message in warnings)


def test_assemble_accepts_a_definition_without_a_manifest_revision(tmp_path: Path, machines_path: Path) -> None:
    stats_path, definition_path = _standard_fixture(tmp_path)
    assert "manifest_sha256" not in json.loads(definition_path.read_text(encoding="utf-8"))

    result = assemble(
        [stats_path],
        machine_id=MACHINE_ID,
        out_dir=tmp_path / "snapshots",
        definition_paths=[definition_path],
        machines_path=machines_path,
        resolve_baselines=False,
    )
    assert result.total_rows == 7


def _snapshot_row(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "machine_id": MACHINE_ID,
        "source_kind": "corpus",
        "stats_file": "stats.jsonl",
        "job_id": "job-0",
        "worker_version": "17.8.5",
        "horde_sdk_version": "0.29.0",
        "sampler_constraints_artifact_sha256": SAMPLER_CONSTRAINTS_ARTIFACT_SHA256,
        "sampler_execution_contract_version": SamplerExecutionContractVersion.V1,
        "session_started_at": 1000.0,
        "time_popped": 1000.0,
        "scenario_id": "pricing-corpus",
        "scenario_revision": "1",
        "cell_id": "g1.fast",
        "cell_group": "g1",
        "replicate": 0,
        "permutation": "seed-a",
        "position": 1,
        "source_processing": "txt2img",
        "lora_role": None,
        "cold_cell": False,
        "warmup": False,
        "model_name": "Model A",
        "baseline": "stable_diffusion_1",
        "baseline_resolved": False,
        "width": 512,
        "height": 512,
        "trajectory_steps": 30,
        "cfg_scale": 7.5,
        "denoising_strength": None,
        "sampler_name": "k_euler",
        "scheduler": "karras",
        "n_images": 1,
        "loras_count": 0,
        "tis_count": 0,
        "control_type": None,
        "hires_fix": False,
        "post_processing": (),
        "is_alchemy": False,
        "degraded_features": False,
        "sampler_window_seconds": 10.0,
        "e2e_seconds": 12.0,
        "sampling_seconds": 6.0,
        "queue_wait_seconds": 1.0,
        "safety_seconds": 0.5,
        "kudos_reward": 0.0,
        "faulted": False,
        "stage_order_ok": True,
    }
    row.update(overrides)
    return SnapshotRow.model_validate(row).model_dump(mode="json")


def _write_snapshot_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    pd.DataFrame(rows).to_parquet(path, engine="pyarrow", index=False)


def test_sanitize_rules_drop_what_they_claim(tmp_path: Path) -> None:
    rows: list[dict[str, Any]] = []
    popped = 1000.0

    def add(**overrides: Any) -> None:
        nonlocal popped
        rows.append(_snapshot_row(job_id=f"job-{len(rows)}", time_popped=popped, **overrides))
        popped += 60.0

    add(warmup=True, cell_id="warmup.a", cell_group="warmup")
    add(faulted=True)
    add(stage_order_ok=False)
    add(sampler_window_seconds=-1.0)
    # An outlier trio: two agreeing replicates and one far off; rule 5 takes exactly the outlier.
    add(cell_id="g1.trio", sampler_window_seconds=10.0, replicate=0)
    add(cell_id="g1.trio", sampler_window_seconds=10.5, replicate=1)
    add(cell_id="g1.trio", sampler_window_seconds=90.0, replicate=2)
    # A clean pair that no rule may touch.
    add(cell_id="g1.pair", sampler_window_seconds=12.0, replicate=0)
    add(cell_id="g1.pair", sampler_window_seconds=12.4, replicate=1)

    snapshot_path = tmp_path / "snapshot-test.parquet"
    _write_snapshot_parquet(snapshot_path, rows)

    result = sanitize(snapshot_path, out_dir=tmp_path / "clean", config=SanitizeConfig(residual_prune=False))

    assert result.dropped_by_rule["drop_warmup"] == 1
    assert result.dropped_by_rule["rule1_drop_faulted"] == 1
    assert result.dropped_by_rule["rule3_drop_degenerate_timestamps"] == 2
    assert result.dropped_by_rule["rule5_mad_outliers"] == 1
    assert result.rows_out == 4

    clean = pd.read_parquet(result.clean_path)
    assert set(clean[clean["cell_id"] == "g1.trio"]["sampler_window_seconds"]) == {10.0, 10.5}
    report = json.loads(result.clean_path.with_suffix(".json").read_text(encoding="utf-8"))
    assert report["rows_in"] == len(rows)
    assert "rule5_mad_outliers" in report["dropped_job_ids"]


def test_sanitize_excludes_adaptive_rows_without_the_manifest_execution_contract(tmp_path: Path) -> None:
    rows = [
        _snapshot_row(
            job_id="adaptive-legacy",
            sampler_name="k_dpm_adaptive",
            sampler_execution_contract_version=None,
        ),
        _snapshot_row(
            job_id="adaptive-current",
            sampler_name="k_dpm_adaptive",
            sampler_execution_contract_version=SamplerExecutionContractVersion.V1,
            time_popped=1060.0,
        ),
        _snapshot_row(job_id="fixed-legacy", sampler_execution_contract_version=None, time_popped=1120.0),
    ]
    snapshot_path = tmp_path / "snapshot-adaptive-contract.parquet"
    _write_snapshot_parquet(snapshot_path, rows)

    result = sanitize(
        snapshot_path,
        out_dir=tmp_path / "clean",
        config=SanitizeConfig(mad_outliers=False, residual_prune=False),
    )

    assert result.dropped_by_rule["drop_incompatible_adaptive_execution"] == 1
    clean = pd.read_parquet(result.clean_path)
    assert set(clean["job_id"]) == {"adaptive-current", "fixed-legacy"}


def test_sanitize_survivor_floor_holds(tmp_path: Path) -> None:
    # Two wildly different rows and one moderate one: without the floor, statistics could take
    # the cell below two survivors.
    rows = [
        _snapshot_row(job_id="a", cell_id="g1.small", sampler_window_seconds=10.0, replicate=0),
        _snapshot_row(job_id="b", cell_id="g1.small", sampler_window_seconds=10.1, replicate=1, time_popped=1060.0),
        _snapshot_row(job_id="c", cell_id="g1.small", sampler_window_seconds=500.0, replicate=2, time_popped=1120.0),
    ]
    snapshot_path = tmp_path / "snapshot-floor.parquet"
    _write_snapshot_parquet(snapshot_path, rows)

    result = sanitize(snapshot_path, out_dir=tmp_path / "clean", config=SanitizeConfig(residual_prune=False))

    assert result.rows_out >= 2


def test_sanitize_swap_rule_targets_production_only(tmp_path: Path) -> None:
    rows = []
    for index, (source_kind, model) in enumerate(
        [
            ("production", "Model A"),
            ("production", "Model B"),
            ("production", "Model B"),
            ("corpus", "Model A"),
            ("corpus", "Model B"),
        ],
    ):
        overrides: dict[str, Any] = {
            "job_id": f"job-{index}",
            "time_popped": 1000.0 + index * 60,
            "source_kind": source_kind,
            "model_name": model,
        }
        if source_kind == "production":
            overrides.update({"cell_id": None, "cell_group": None, "replicate": None, "permutation": None})
            overrides.update({"position": None, "source_processing": None, "scenario_id": None})
            overrides.update({"scenario_revision": None})
        rows.append(_snapshot_row(**overrides))

    snapshot_path = tmp_path / "snapshot-swap.parquet"
    _write_snapshot_parquet(snapshot_path, rows)

    result = sanitize(
        snapshot_path,
        out_dir=tmp_path / "clean",
        config=SanitizeConfig(mad_outliers=False, residual_prune=False),
    )

    # Production: the session-initial load and the swap onto Model B; corpus rows are exempt.
    assert result.dropped_by_rule["rule2_drop_unlabeled_swap"] == 2
    clean = pd.read_parquet(result.clean_path)
    assert int((clean["source_kind"] == "corpus").sum()) == 2


def test_train_and_evaluate_end_to_end(tmp_path: Path) -> None:
    pytest.importorskip("lightgbm")
    from hordelib.kudos_training.evaluate import evaluate

    rows = []
    popped = 1000.0
    for replicate in range(20):
        for cell_id, steps, width, window in (
            ("g1.fast", 10, 512, 4.0),
            ("g1.slow", 50, 512, 16.0),
            ("g2.big", 30, 1024, 11.0),
        ):
            rows.append(
                _snapshot_row(
                    job_id=f"job-{cell_id}-{replicate}",
                    cell_id=cell_id,
                    time_popped=popped,
                    trajectory_steps=steps,
                    width=width,
                    height=width,
                    replicate=replicate,
                    sampler_window_seconds=window + 0.2 * (replicate % 5),
                ),
            )
            popped += 30.0

    clean_path = tmp_path / "clean-test.parquet"
    _write_snapshot_parquet(clean_path, rows)

    run = train(clean_path, out_dir=tmp_path / "runs", config=TrainConfig(n_estimators=80, early_stopping_rounds=20))
    assert (run.run_dir / "model.txt").exists()
    assert (run.run_dir / "splits.json").exists()
    assert run.split_mode == "time_ordered_rows"
    assert 0 <= run.metrics["test"]["median_ape"] < 1.0

    evaluation = evaluate(run.run_dir, clean_path)
    assert evaluation.report_path.exists()
    assert evaluation.candidate_spread is not None
    per_cell = pd.read_csv(run.run_dir / "per_cell.csv")
    assert set(per_cell["cell_id"]) == {"g1.fast", "g1.slow", "g2.big"}


def test_train_rejects_a_sampler_observed_at_only_one_trajectory_length(tmp_path: Path) -> None:
    pytest.importorskip("lightgbm")
    rows = [
        _snapshot_row(job_id=f"job-{replicate}", replicate=replicate, time_popped=1000.0 + replicate)
        for replicate in range(6)
    ]
    clean_path = tmp_path / "undercovered.parquet"
    _write_snapshot_parquet(clean_path, rows)

    with pytest.raises(ValueError, match="trajectory coverage"):
        train(clean_path, out_dir=tmp_path / "runs")


def _session_fixture(tmp_path: Path, *, created_at: float | None = 980.0, machine: bool = True) -> Path:
    """A definition artifact beside its own stats session, plus a decoy session from another run."""
    stats_path, definition_path = _standard_fixture(tmp_path)
    session_path = tmp_path / "stats-v0.0.0-20000102-000000-000.jsonl"
    stats_path.replace(session_path)

    # A session from an unrelated run, started before the artifact, so discovery has to choose
    # rather than take the only file in the directory.
    decoy = tmp_path / "stats-v0.0.0-20000101-000000-000.jsonl"
    _write_stats(decoy, [_record(_cell("g1.fast"), popped=1.0)])
    decoy_lines = decoy.read_text(encoding="utf-8").splitlines()
    decoy_start = json.loads(decoy_lines[0])
    decoy_start["timestamp"] = 500.0
    decoy_lines[0] = json.dumps(decoy_start)
    decoy.write_text("\n".join(decoy_lines) + "\n", encoding="utf-8")

    artifact = json.loads(definition_path.read_text(encoding="utf-8"))
    if created_at is not None:
        artifact["created_at"] = created_at
    if machine:
        artifact["machine"] = {
            "machine_id": MACHINE_ID,
            "hostname": "test-host",
            "gpu_model": "Test GPU",
            "vram_mb": 1,
            "os": "Test OS",
            "worker_version": "18.4.1",
        }
    definition_path.write_text(json.dumps(artifact), encoding="utf-8")
    return definition_path


def test_resolve_session_finds_the_session_the_artifact_was_written_for(
    tmp_path: Path,
    machines_path: Path,
) -> None:
    from hordelib.kudos_training.assemble import resolve_session

    definition_path = _session_fixture(tmp_path)
    session = resolve_session(definition_path)

    assert session.machine is not None
    assert session.machine.machine_id == MACHINE_ID
    assert [path.name for path in session.stats_paths] == ["stats-v0.0.0-20000102-000000-000.jsonl"]

    result = assemble(
        list(session.stats_paths),
        machine_id=session.machine.machine_id,
        out_dir=tmp_path / "snapshots",
        definition_paths=[definition_path],
        machines_path=machines_path,
        resolve_baselines=False,
    )
    assert result.total_rows == 7


def test_resolve_session_rejects_an_artifact_without_created_at(tmp_path: Path) -> None:
    from hordelib.kudos_training.assemble import resolve_session

    definition_path = _session_fixture(tmp_path, created_at=None)
    with pytest.raises(AssemblyError, match="created_at"):
        resolve_session(definition_path)


def test_resolve_session_rejects_a_session_outside_the_search_window(tmp_path: Path) -> None:
    from hordelib.kudos_training.assemble import resolve_session

    # The fixture session starts at t=1000; an artifact written a day later belongs to no session here.
    definition_path = _session_fixture(tmp_path, created_at=1000.0 + 86400.0)
    with pytest.raises(AssemblyError, match="no stats session"):
        resolve_session(definition_path)


def test_machines_add_writes_a_parseable_entry_and_refuses_duplicates(tmp_path: Path, machines_path: Path) -> None:
    import tomllib

    from hordelib.kudos_training.assemble import CorpusDefinition, add_machine, load_machines

    definition_path = _session_fixture(tmp_path)
    definition = CorpusDefinition.model_validate_json(definition_path.read_text(encoding="utf-8"))
    assert definition.machine is not None

    other = definition.machine.model_copy(update={"machine_id": "test-rig-2"})
    entry = add_machine(other, notes="Second rig.", machines_path=machines_path)
    assert entry == {"gpu_model": "Test GPU", "vram_mb": 1, "os": "Test OS", "notes": "Second rig."}

    with machines_path.open("rb") as handle:
        table = tomllib.load(handle)
    assert table["machines"]["test-rig-2"]["notes"] == "Second rig."
    assert set(load_machines(machines_path)) == {MACHINE_ID, "test-rig-2"}

    with pytest.raises(AssemblyError, match="already declares machine"):
        add_machine(other, machines_path=machines_path)


def test_machines_add_warns_when_a_registered_id_carries_different_facts(
    tmp_path: Path,
    machines_path: Path,
) -> None:
    import tomllib

    from hordelib.kudos_training.assemble import CorpusDefinition, add_machine

    definition_path = _session_fixture(tmp_path)
    definition = CorpusDefinition.model_validate_json(definition_path.read_text(encoding="utf-8"))
    assert definition.machine is not None
    relabeled = definition.machine.model_copy(update={"gpu_model": "Another GPU"})

    before = machines_path.read_text(encoding="utf-8")
    with _captured_warnings() as warnings, pytest.raises(AssemblyError, match="already declares machine"):
        add_machine(relabeled, machines_path=machines_path)

    assert any("gpu_model" in message and "Another GPU" in message for message in warnings)
    assert machines_path.read_text(encoding="utf-8") == before
    with machines_path.open("rb") as handle:
        assert tomllib.load(handle)["machines"][MACHINE_ID]["gpu_model"] == "Test GPU"


def _bundle_fixture(tmp_path: Path) -> Path:
    """A run bundle: the definition artifact, two rotated stats parts and a manifest over both."""
    stats_path, definition_path = _standard_fixture(tmp_path)

    lines = stats_path.read_text(encoding="utf-8").splitlines()
    first_part = tmp_path / "stats-v0.0.0-20000102-000000-000.jsonl"
    second_part = tmp_path / "stats-v0.0.0-20000102-000000-001.jsonl"
    first_part.write_text("\n".join(lines[:4]) + "\n", encoding="utf-8")
    second_part.write_text("\n".join(lines[4:]) + "\n", encoding="utf-8")
    stats_path.unlink()

    # The bundle names its parts, so the artifact needs no write time for discovery to work.
    artifact = json.loads(definition_path.read_text(encoding="utf-8"))
    artifact["machine"] = {"machine_id": MACHINE_ID, "gpu_model": "Test GPU", "vram_mb": 1, "os": "Test OS"}
    definition_path.write_text(json.dumps(artifact), encoding="utf-8")

    def described(path: Path, role: str) -> dict[str, Any]:
        payload = path.read_bytes()
        return {
            "name": path.name,
            "role": role,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
        }

    bundle = {
        "bundle_format": "1",
        "machine": artifact["machine"],
        "tier": "smoke",
        "scenario_name": "pricing-corpus",
        "scenario_revision": "1",
        "created_at": 980.0,
        "created_at_utc": "2000-01-01T00:16:20Z",
        "files": [
            described(definition_path, "definition"),
            # Listed out of rotation order, so the reader has to sort rather than trust the manifest.
            described(second_part, "stats"),
            described(first_part, "stats"),
        ],
    }
    (tmp_path / "bundle.json").write_text(json.dumps(bundle), encoding="utf-8")
    return definition_path


def test_resolve_session_reads_the_parts_a_bundle_manifest_lists(tmp_path: Path, machines_path: Path) -> None:
    from hordelib.kudos_training.assemble import resolve_session

    definition_path = _bundle_fixture(tmp_path)
    session = resolve_session(definition_path)

    assert [path.name for path in session.stats_paths] == [
        "stats-v0.0.0-20000102-000000-000.jsonl",
        "stats-v0.0.0-20000102-000000-001.jsonl",
    ]
    assert session.machine is not None

    result = assemble(
        list(session.stats_paths),
        machine_id=session.machine.machine_id,
        out_dir=tmp_path / "snapshots",
        definition_paths=[definition_path],
        machines_path=machines_path,
        resolve_baselines=False,
    )
    assert result.total_rows == 7


def test_resolve_session_refuses_a_tampered_stats_part(tmp_path: Path) -> None:
    from hordelib.kudos_training.assemble import resolve_session

    definition_path = _bundle_fixture(tmp_path)
    tampered = tmp_path / "stats-v0.0.0-20000102-000000-001.jsonl"
    tampered.write_text(tampered.read_text(encoding="utf-8").replace('"kudos_reward": 0.0', '"kudos_reward": 9.0'))

    with pytest.raises(AssemblyError, match="hashes to"):
        resolve_session(definition_path)


def test_resolve_session_refuses_a_bundle_missing_a_listed_file(tmp_path: Path) -> None:
    from hordelib.kudos_training.assemble import resolve_session

    definition_path = _bundle_fixture(tmp_path)
    (tmp_path / "stats-v0.0.0-20000102-000000-001.jsonl").unlink()

    with pytest.raises(AssemblyError, match="which is missing from"):
        resolve_session(definition_path)


_CALIBRATION_CONFIG = TrainConfig(n_estimators=400, min_child_samples=2)
"""Shape-model settings sized for the handful of replicates a synthetic frame carries."""


def _two_machine_frame(
    *,
    intercept: float,
    slope: float,
    noise: float = 0.0,
    other_id: str = "test-rig-2",
) -> list[dict[str, Any]]:
    """Reference rows plus a second machine whose log seconds are an affine image of them.

    The second machine also runs one cell the reference never ran, so the out-of-regime label has
    something to catch.
    """
    shared_cells = [
        ("g1.a", 10, 512, 4.0),
        ("g1.b", 20, 512, 7.0),
        ("g1.c", 30, 512, 10.0),
        ("g2.d", 40, 768, 18.0),
        ("g2.e", 50, 768, 24.0),
        ("g2.f", 60, 1024, 40.0),
    ]
    rows: list[dict[str, Any]] = []
    popped = 1000.0
    for replicate in range(4):
        for cell_id, steps, width, window in shared_cells:
            reference_seconds = window + 0.1 * replicate
            rows.append(
                _snapshot_row(
                    job_id=f"ref-{cell_id}-{replicate}",
                    machine_id=MACHINE_ID,
                    cell_id=cell_id,
                    replicate=replicate,
                    time_popped=popped,
                    trajectory_steps=steps,
                    width=width,
                    height=width,
                    sampler_window_seconds=reference_seconds,
                ),
            )
            popped += 30.0
            wobble = noise * (1 if (replicate + steps) % 2 else -1)
            rows.append(
                _snapshot_row(
                    job_id=f"other-{cell_id}-{replicate}",
                    machine_id=other_id,
                    cell_id=cell_id,
                    replicate=replicate,
                    time_popped=popped,
                    trajectory_steps=steps,
                    width=width,
                    height=width,
                    sampler_window_seconds=math.exp(intercept + slope * math.log(reference_seconds) + wobble),
                ),
            )
            popped += 30.0

    for replicate in range(4):
        rows.append(
            _snapshot_row(
                job_id=f"other-only-{replicate}",
                machine_id=other_id,
                cell_id="g3.other-only",
                replicate=replicate,
                time_popped=popped,
                trajectory_steps=70,
                width=1024,
                height=1024,
                sampler_window_seconds=math.exp(intercept + slope * math.log(55.0 + 0.1 * replicate)),
            ),
        )
        popped += 30.0
    return rows


def test_calibrate_recovers_the_affine_map_and_flags_out_of_regime_cells(tmp_path: Path) -> None:
    pytest.importorskip("lightgbm")
    from hordelib.kudos_training.calibrate import calibrate

    rows = _two_machine_frame(intercept=0.4, slope=1.1, noise=0.01)
    clean_path = tmp_path / "clean-two-machines.parquet"
    _write_snapshot_parquet(clean_path, rows)

    result = calibrate(
        clean_path,
        out_dir=tmp_path / "calibration",
        reference_machine=MACHINE_ID,
        config=_CALIBRATION_CONFIG,
    )

    assert result.passed
    assert result.calibrated_path is not None
    (calibration,) = result.machines
    assert calibration.machine_id == "test-rig-2"
    assert calibration.n_overlap_cells == 6
    assert calibration.n_overlap_rows == 24
    assert calibration.intercept == pytest.approx(0.4, abs=0.2)
    assert calibration.slope == pytest.approx(1.1, abs=0.15)
    assert calibration.residual_spread < 1.5

    calibrated = pd.read_parquet(result.calibrated_path)
    assert bool(calibrated["calibrated"].all())
    other = calibrated[calibrated["measured_machine_id"] == "test-rig-2"]
    assert bool(other[other["cell_id"] == "g3.other-only"]["out_of_regime"].all())
    assert not bool(other[other["cell_id"] == "g1.a"]["out_of_regime"].any())
    assert not bool(calibrated[calibrated["measured_machine_id"] == MACHINE_ID]["out_of_regime"].any())
    # Mapping a shared cell back must land near what the reference machine measured on it.
    mapped = float(other[other["cell_id"] == "g2.f"]["sampler_window_seconds"].median())
    assert mapped == pytest.approx(40.15, rel=0.15)
    assert float(other["measured_seconds"].iloc[0]) != pytest.approx(float(other["sampler_window_seconds"].iloc[0]))

    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["reference_machine"] == MACHINE_ID
    assert report["machines"][0]["passes"] is True


def test_calibrate_fails_the_bar_on_a_noisy_machine(tmp_path: Path) -> None:
    pytest.importorskip("lightgbm")
    from hordelib.kudos_training.calibrate import calibrate

    rows = _two_machine_frame(intercept=0.4, slope=1.1, noise=0.9)
    clean_path = tmp_path / "clean-noisy.parquet"
    _write_snapshot_parquet(clean_path, rows)

    result = calibrate(
        clean_path,
        out_dir=tmp_path / "calibration",
        reference_machine=MACHINE_ID,
        config=_CALIBRATION_CONFIG,
    )

    assert not result.passed
    assert result.calibrated_path is None
    assert result.machines[0].residual_spread > 1.5
    assert not list((tmp_path / "calibration").glob("*.parquet"))
    assert result.report_path.exists()


def test_calibrate_needs_enough_shared_cells(tmp_path: Path) -> None:
    pytest.importorskip("lightgbm")
    from hordelib.kudos_training.calibrate import CalibrationError, calibrate

    rows = [row for row in _two_machine_frame(intercept=0.4, slope=1.1) if row["cell_id"] not in ("g2.d", "g2.e")]
    clean_path = tmp_path / "clean-thin-overlap.parquet"
    _write_snapshot_parquet(clean_path, rows)

    with pytest.raises(CalibrationError, match="shares 4 cells"):
        calibrate(
            clean_path,
            out_dir=tmp_path / "calibration",
            reference_machine=MACHINE_ID,
            config=_CALIBRATION_CONFIG,
        )


def test_train_refuses_uncalibrated_multi_machine_data(tmp_path: Path) -> None:
    pytest.importorskip("lightgbm")
    rows = _two_machine_frame(intercept=0.4, slope=1.1)
    clean_path = tmp_path / "clean-uncalibrated.parquet"
    _write_snapshot_parquet(clean_path, rows)

    with pytest.raises(ValueError, match="not calibrated"):
        train(clean_path, out_dir=tmp_path / "runs")
