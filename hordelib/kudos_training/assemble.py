"""The assemble stage: worker stats JSONL in, labeled parquet snapshot out.

One snapshot is assembled from one or more stats files measured on a single named machine. Rows
from a pricing-corpus session are paired to the corpus definition artifact emitted beside the
stats stream and carry cell labels; rows from production sessions carry the production overlay
label instead. All later stages consume snapshots only, keyed by the content hash embedded in the
snapshot filename.

Pairing rule: ``job_completed`` records land in the stats file in FINALIZED order, and a
post-processing job finalizes after its successor can, so file order swaps neighbours. Records are
therefore sorted by ``time_popped`` (pop order equals submission order under the serialized
harness) and then paired positionally against the definition's job list, verifying every request
axis per pair. A missing record (a job that never produced one) is tolerated by skipping its job;
an axis mismatch that skipping cannot explain is a hard error, because it means the streams have
diverged and every later label would be wrong.
"""

import hashlib
import json
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from horde_sdk.generation_parameters.image.sampler_work import SamplerExecutionContractVersion
from loguru import logger
from pydantic import BaseModel, ConfigDict

from hordelib.kudos_training.schema import (
    SNAPSHOT_SCHEMA_VERSION,
    SnapshotRow,
    SourceKind,
)
from hordelib.utils.optional_deps import require

DEFAULT_MACHINES_PATH = Path(__file__).parent / "machines.toml"
"""The checked-in table of trusted developer machines."""

PRICING_CORPUS_SCENARIO_ID = "pricing-corpus"
"""The scenario id pricing-corpus sessions stamp into their session config."""

_MAX_CONSECUTIVE_MISSING_JOBS = 3
"""How many definition jobs in a row may lack a record before pairing is declared broken."""

_INFERENCE_START_STAGE = "INFERENCE_IN_PROGRESS"
_DISAGGREGATED_WINDOW_END_STAGE = "DISAGGREGATION_DECODING"
_MONOLITHIC_WINDOW_END_STAGE = "PENDING_SAFETY_CHECK"


class AssemblyError(RuntimeError):
    """Raised when a stats stream cannot be assembled into trustworthy rows."""


class PairingError(AssemblyError):
    """Raised when corpus records cannot be reconciled with the definition's job list."""


class _SessionStartConfig(BaseModel):
    """The slice of the session config snapshot the assembler reads."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    scenario_id: str | None = None
    scenario_revision: str | None = None
    horde_sdk_version: str | None = None
    sampler_constraints_artifact_sha256: str | None = None
    sampler_execution_contract_version: SamplerExecutionContractVersion | None = None


class _SessionStart(BaseModel):
    """A ``session_start`` event."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    worker_version: str
    timestamp: float
    config: _SessionStartConfig


class _JobRecord(BaseModel):
    """The slice of a ``job_completed`` record's ``job`` object the assembler reads."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    job_id: str
    is_alchemy: bool = False
    faulted: bool
    time_popped: float
    stage_timestamps: dict[str, float]
    queue_wait_seconds: float
    e2e_seconds: float
    safety_seconds: float | None = None
    model_name: str
    steps: int
    width: int
    height: int
    loras_count: int = 0
    tis_count: int = 0
    control_type: str | None = None
    post_processing: tuple[str, ...] = ()
    sampler_name: str | None = None
    scheduler: str | None = None
    cfg_scale: float | None = None
    hires_fix: bool = False
    batch_count: int = 1
    sampling_seconds: float | None = None
    kudos_reward: float | None = None


class _JobCompletedEvent(BaseModel):
    """A ``job_completed`` event, with the baseline the worker reports beside the job."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    job: _JobRecord
    baseline: str | None = None


class DefinitionCell(BaseModel):
    """A pricing-corpus cell, as serialized into the definition artifact."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    cell_id: str
    group: str
    model: str
    width: int
    height: int
    steps: int
    cfg_scale: float
    n_iter: int
    sampler_name: str
    scheduler: str
    source_processing: str
    denoising_strength: float | None = None
    hires_fix: bool = False
    post_processing: tuple[str, ...] = ()
    control_type: str | None = None
    lora_version_ids: tuple[str, ...] = ()
    ti_names: tuple[str, ...] = ()
    lora_role: str | None = None
    requires_model_switch: bool = False


class DefinitionJob(BaseModel):
    """One ordered corpus job, as serialized into the definition artifact."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    position: int
    cell_id: str
    group: str
    permutation: str
    replicate: int


class CorpusDefinition(BaseModel):
    """The definition artifact a pricing-corpus run emits beside its stats stream."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    scenario_name: str
    scenario_revision: str
    tier: str
    warmup_job_count: int
    cells: tuple[DefinitionCell, ...]
    jobs: tuple[DefinitionJob, ...]

    def cells_by_id(self) -> dict[str, DefinitionCell]:
        """Return the cells keyed by cell id."""
        return {cell.cell_id: cell for cell in self.cells}


@dataclass(frozen=True)
class SessionReport:
    """What assembly did with one stats file."""

    stats_file: str
    source_kind: SourceKind
    rows: int
    faulted_rows: int
    missing_positions: tuple[int, ...] = ()
    """Definition positions that produced no record (corpus sessions only)."""


@dataclass(frozen=True)
class AssemblyResult:
    """The snapshot a run of the assemble stage produced, and how it got there."""

    snapshot_path: Path
    content_hash: str
    machine_id: str
    total_rows: int
    sessions: tuple[SessionReport, ...] = field(default_factory=tuple)


def load_machines(path: Path | None = None) -> dict[str, dict[str, Any]]:
    """Load the trusted-machines table.

    Args:
        path: The table to read. Defaults to the checked-in ``machines.toml``.

    Returns:
        Machine metadata keyed by machine id.

    Raises:
        AssemblyError: If the table has no ``machines`` section.
    """
    resolved = path if path is not None else DEFAULT_MACHINES_PATH
    with resolved.open("rb") as handle:
        table = tomllib.load(handle)
    machines = table.get("machines")
    if not isinstance(machines, dict) or not machines:
        raise AssemblyError(f"{resolved} declares no [machines.<id>] sections")
    return machines


def assemble(
    stats_paths: list[Path],
    *,
    machine_id: str,
    out_dir: Path,
    definition_paths: list[Path] | None = None,
    machines_path: Path | None = None,
    resolve_baselines: bool = True,
) -> AssemblyResult:
    """Assemble stats files from one machine into a parquet snapshot.

    Args:
        stats_paths: Stats JSONL files, each holding exactly one session.
        machine_id: The measuring machine's id; must exist in the machines table.
        out_dir: Directory the snapshot is written into (created if absent).
        definition_paths: Pricing-corpus definition artifacts. Required for every
            pricing-corpus session among *stats_paths*; matched by scenario revision.
        machines_path: Alternate machines table, for tests.
        resolve_baselines: Whether to correct baselines through the model reference, which may
            fetch from the network. Disable for offline runs; rows then keep the worker-reported
            value with ``baseline_resolved`` False.

    Returns:
        The written snapshot and per-session reports.

    Raises:
        AssemblyError: If the machine is unknown, a session is malformed, or a corpus
            session has no matching definition.
        PairingError: If corpus records diverge from the definition's job list.
    """
    machines = load_machines(machines_path)
    if machine_id not in machines:
        known = ", ".join(sorted(machines))
        raise AssemblyError(f"unknown machine id {machine_id!r}; machines.toml knows: {known}")

    definitions = [_load_definition(path) for path in (definition_paths or [])]

    rows: list[SnapshotRow] = []
    reports: list[SessionReport] = []
    for stats_path in stats_paths:
        session_rows, report = _assemble_session(stats_path, machine_id=machine_id, definitions=definitions)
        rows.extend(session_rows)
        reports.append(report)

    if not rows:
        raise AssemblyError("no job_completed records found in any input file")

    if resolve_baselines:
        rows = _resolve_baselines(rows)

    snapshot_path, content_hash = _write_snapshot(rows, machine_id=machine_id, out_dir=out_dir, reports=reports)
    logger.info(f"assembled {len(rows)} rows from {len(stats_paths)} file(s) into {snapshot_path}")
    return AssemblyResult(
        snapshot_path=snapshot_path,
        content_hash=content_hash,
        machine_id=machine_id,
        total_rows=len(rows),
        sessions=tuple(reports),
    )


def _load_definition(path: Path) -> CorpusDefinition:
    """Load and validate one definition artifact."""
    return CorpusDefinition.model_validate_json(path.read_text(encoding="utf-8"))


def _assemble_session(
    stats_path: Path,
    *,
    machine_id: str,
    definitions: list[CorpusDefinition],
) -> tuple[list[SnapshotRow], SessionReport]:
    """Assemble one stats file into rows."""
    session_start, records = _parse_stats_file(stats_path)
    records.sort(key=lambda event: event.job.time_popped)

    scenario_id = session_start.config.scenario_id
    is_corpus = scenario_id == PRICING_CORPUS_SCENARIO_ID

    if is_corpus:
        definition = _match_definition(session_start, definitions, stats_path)
        pairs, missing_positions = _pair_records(records, definition, stats_path)
        rows = [
            _build_row(
                event,
                session_start=session_start,
                machine_id=machine_id,
                stats_file=stats_path.name,
                job=job,
                cell=definition.cells_by_id()[job.cell_id],
            )
            for job, event in pairs
        ]
    else:
        missing_positions = []
        rows = [
            _build_row(
                event,
                session_start=session_start,
                machine_id=machine_id,
                stats_file=stats_path.name,
                job=None,
                cell=None,
            )
            for event in records
        ]

    report = SessionReport(
        stats_file=stats_path.name,
        source_kind=SourceKind.CORPUS if is_corpus else SourceKind.PRODUCTION,
        rows=len(rows),
        faulted_rows=sum(1 for row in rows if row.faulted),
        missing_positions=tuple(missing_positions),
    )
    return rows, report


def _parse_stats_file(stats_path: Path) -> tuple[_SessionStart, list[_JobCompletedEvent]]:
    """Parse one stats JSONL file into its session_start and job_completed events."""
    session_starts: list[_SessionStart] = []
    records: list[_JobCompletedEvent] = []
    with stats_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                event = json.loads(stripped)
            except json.JSONDecodeError as error:
                raise AssemblyError(f"{stats_path}:{line_number} is not valid JSON: {error}") from error
            event_kind = event.get("event")
            if event_kind == "session_start":
                session_starts.append(_SessionStart.model_validate(event))
            elif event_kind == "job_completed":
                records.append(_JobCompletedEvent.model_validate(event))

    if len(session_starts) != 1:
        raise AssemblyError(f"{stats_path} holds {len(session_starts)} session_start events; expected exactly 1")
    return session_starts[0], records


def _match_definition(
    session_start: _SessionStart,
    definitions: list[CorpusDefinition],
    stats_path: Path,
) -> CorpusDefinition:
    """Find the definition artifact matching a pricing-corpus session."""
    revision = session_start.config.scenario_revision
    matches = [
        definition
        for definition in definitions
        if definition.scenario_name == PRICING_CORPUS_SCENARIO_ID and definition.scenario_revision == revision
    ]
    if not matches:
        raise AssemblyError(
            f"{stats_path} is a pricing-corpus session (revision {revision!r}) but no matching "
            "definition artifact was supplied; pass it via definition_paths/--definition",
        )
    if len(matches) > 1:
        raise AssemblyError(f"{stats_path} matches {len(matches)} definition artifacts; supply exactly one")
    return matches[0]


def _pair_records(
    records: list[_JobCompletedEvent],
    definition: CorpusDefinition,
    stats_path: Path,
) -> tuple[list[tuple[DefinitionJob, _JobCompletedEvent]], list[int]]:
    """Pair pop-ordered records positionally to the definition's job list.

    Returns:
        The (job, record) pairs and the positions of jobs that produced no record.

    Raises:
        PairingError: If a record mismatches every candidate job within the skip bound.
    """
    cells = definition.cells_by_id()
    pairs: list[tuple[DefinitionJob, _JobCompletedEvent]] = []
    missing_positions: list[int] = []

    job_index = 0
    for event in records:
        skipped = 0
        while job_index < len(definition.jobs):
            job = definition.jobs[job_index]
            mismatches = _axis_mismatches(cells[job.cell_id], event)
            if not mismatches:
                pairs.append((job, event))
                job_index += 1
                break
            skipped += 1
            if skipped > _MAX_CONSECUTIVE_MISSING_JOBS:
                raise PairingError(
                    f"{stats_path}: record {event.job.job_id} does not match position {job.position} "
                    f"({job.cell_id}: {'; '.join(mismatches)}) nor the {skipped - 1} preceding candidates; "
                    "record stream and definition have diverged",
                )
            missing_positions.append(job.position)
            job_index += 1
        else:
            raise PairingError(
                f"{stats_path}: record {event.job.job_id} has no remaining definition job to pair with",
            )

    missing_positions.extend(job.position for job in definition.jobs[job_index:])
    return pairs, missing_positions


def _axis_mismatches(cell: DefinitionCell, event: _JobCompletedEvent) -> list[str]:
    """Compare every request axis between a cell and a record; empty means they match."""
    record = event.job
    expectations: list[tuple[str, object, object]] = [
        ("model", cell.model, record.model_name),
        ("steps", cell.steps, record.steps),
        ("width", cell.width, record.width),
        ("height", cell.height, record.height),
        ("sampler_name", cell.sampler_name, record.sampler_name),
        ("scheduler", cell.scheduler, record.scheduler),
        ("cfg_scale", cell.cfg_scale, record.cfg_scale),
        ("n_iter", cell.n_iter, record.batch_count),
        ("loras_count", len(cell.lora_version_ids), record.loras_count),
        ("tis_count", len(cell.ti_names), record.tis_count),
        ("post_processing", tuple(cell.post_processing), tuple(record.post_processing)),
        ("control_type", cell.control_type, record.control_type),
        ("hires_fix", cell.hires_fix, record.hires_fix),
    ]
    return [
        f"{name}: expected {expected!r} got {actual!r}"
        for name, expected, actual in expectations
        if expected != actual
    ]


def _build_row(
    event: _JobCompletedEvent,
    *,
    session_start: _SessionStart,
    machine_id: str,
    stats_file: str,
    job: DefinitionJob | None,
    cell: DefinitionCell | None,
) -> SnapshotRow:
    """Build one snapshot row from a record and its (optional) corpus labels."""
    record = event.job
    stages = record.stage_timestamps

    window_start = stages.get(_INFERENCE_START_STAGE)
    window_end = stages.get(_DISAGGREGATED_WINDOW_END_STAGE, stages.get(_MONOLITHIC_WINDOW_END_STAGE))
    sampler_window_seconds = None
    if window_start is not None and window_end is not None:
        sampler_window_seconds = window_end - window_start

    stage_values = list(stages.values())
    stage_order_ok = all(later >= earlier for earlier, later in zip(stage_values, stage_values[1:], strict=False))

    degraded_features = record.sampler_name is None or record.scheduler is None or record.cfg_scale is None

    return SnapshotRow(
        machine_id=machine_id,
        source_kind=SourceKind.CORPUS if job is not None else SourceKind.PRODUCTION,
        stats_file=stats_file,
        job_id=record.job_id,
        worker_version=session_start.worker_version,
        horde_sdk_version=session_start.config.horde_sdk_version,
        sampler_constraints_artifact_sha256=session_start.config.sampler_constraints_artifact_sha256,
        sampler_execution_contract_version=session_start.config.sampler_execution_contract_version,
        session_started_at=session_start.timestamp,
        time_popped=record.time_popped,
        scenario_id=session_start.config.scenario_id,
        scenario_revision=session_start.config.scenario_revision,
        cell_id=job.cell_id if job is not None else None,
        cell_group=job.group if job is not None else None,
        replicate=job.replicate if job is not None else None,
        permutation=job.permutation if job is not None else None,
        position=job.position if job is not None else None,
        source_processing=cell.source_processing if cell is not None else None,
        lora_role=cell.lora_role if cell is not None else None,
        cold_cell=cell.requires_model_switch if cell is not None else False,
        warmup=job.group == "warmup" if job is not None else False,
        model_name=record.model_name,
        baseline=event.baseline,
        baseline_resolved=False,
        width=record.width,
        height=record.height,
        trajectory_steps=record.steps,
        cfg_scale=record.cfg_scale,
        denoising_strength=cell.denoising_strength if cell is not None else None,
        sampler_name=record.sampler_name,
        scheduler=record.scheduler,
        n_images=record.batch_count,
        loras_count=record.loras_count,
        tis_count=record.tis_count,
        control_type=record.control_type,
        hires_fix=record.hires_fix,
        post_processing=tuple(record.post_processing),
        is_alchemy=record.is_alchemy,
        degraded_features=degraded_features,
        sampler_window_seconds=sampler_window_seconds,
        e2e_seconds=record.e2e_seconds,
        sampling_seconds=record.sampling_seconds,
        queue_wait_seconds=record.queue_wait_seconds,
        safety_seconds=record.safety_seconds,
        kudos_reward=record.kudos_reward,
        faulted=record.faulted,
        stage_order_ok=stage_order_ok,
    )


def _resolve_baselines(rows: list[SnapshotRow]) -> list[SnapshotRow]:
    """Override each row's baseline with the model reference's answer, where one exists.

    The worker's stats stream has been observed reporting ``stable_diffusion_1`` for SDXL models
    under the harness, and baseline is a load-bearing training feature, so the model reference is
    treated as the authority. A model the reference does not know (or a reference that cannot be
    read at all) leaves the worker-reported value in place, with ``baseline_resolved`` False so the
    trainer can see which rows carry the unreliable spelling.
    """
    baselines_by_model: dict[str, str | None] = {}
    try:
        from horde_model_reference import MODEL_REFERENCE_CATEGORY
        from horde_model_reference.model_reference_manager import ModelReferenceManager

        manager = ModelReferenceManager()
        for model_name in sorted({row.model_name for row in rows}):
            record = manager.get_model_or_none(MODEL_REFERENCE_CATEGORY.image_generation, model_name)
            baseline = getattr(record, "baseline", None)
            baselines_by_model[model_name] = str(baseline) if baseline is not None else None
    except Exception as error:
        logger.warning(f"model reference unavailable; keeping worker-reported baselines: {error}")

    unresolved = sorted(name for name, baseline in baselines_by_model.items() if baseline is None)
    if unresolved:
        logger.warning(f"model reference has no baseline for: {', '.join(unresolved)}")

    resolved_rows: list[SnapshotRow] = []
    for row in rows:
        baseline = baselines_by_model.get(row.model_name)
        if baseline is None:
            resolved_rows.append(row)
        else:
            resolved_rows.append(row.model_copy(update={"baseline": baseline, "baseline_resolved": True}))
    return resolved_rows


def _write_snapshot(
    rows: list[SnapshotRow],
    *,
    machine_id: str,
    out_dir: Path,
    reports: list[SessionReport],
) -> tuple[Path, str]:
    """Write rows as a content-hash-named parquet snapshot plus a metadata sidecar."""
    require("pandas", extra="kudos-training", feature="kudos-train assemble")
    require("pyarrow", extra="kudos-training", feature="kudos-train assemble")
    import pandas as pd

    frame = pd.DataFrame([row.model_dump(mode="json") for row in rows])

    out_dir.mkdir(parents=True, exist_ok=True)
    staging_path = out_dir / f"snapshot-{machine_id}.parquet.tmp"
    frame.to_parquet(staging_path, engine="pyarrow", index=False)

    content_hash = hashlib.sha256(staging_path.read_bytes()).hexdigest()[:16]
    snapshot_path = out_dir / f"snapshot-{machine_id}-{content_hash}.parquet"
    staging_path.replace(snapshot_path)

    sidecar = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "machine_id": machine_id,
        "content_hash": content_hash,
        "total_rows": len(rows),
        "sessions": [
            {
                "stats_file": report.stats_file,
                "source_kind": report.source_kind,
                "rows": report.rows,
                "faulted_rows": report.faulted_rows,
                "missing_positions": list(report.missing_positions),
            }
            for report in reports
        ],
    }
    snapshot_path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    return snapshot_path, content_hash


__all__ = [
    "DEFAULT_MACHINES_PATH",
    "PRICING_CORPUS_SCENARIO_ID",
    "AssemblyError",
    "AssemblyResult",
    "CorpusDefinition",
    "DefinitionCell",
    "DefinitionJob",
    "PairingError",
    "SessionReport",
    "assemble",
    "load_machines",
]
