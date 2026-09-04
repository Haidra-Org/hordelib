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

import gzip
import hashlib
import json
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from horde_sdk.generation_parameters.image.sampler_work import SamplerExecutionContractVersion
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from hordelib.kudos_training.manifest import default_manifest
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

BUNDLE_MANIFEST_FILENAME = "bundle.json"
"""The manifest a run bundle carries beside its definition artifact and stats parts."""

_SUPPORTED_BUNDLE_FORMATS = frozenset({"1"})
"""Bundle layouts this assembler can verify; an unknown one is refused rather than guessed at."""

_MACHINE_FACT_FIELDS = ("gpu_model", "vram_mb", "os")
"""The hardware facts the machines table records, and the ones a re-registration is compared against."""

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


class MachineDescriptor(BaseModel):
    """The measuring machine, as the worker stamps it into a definition artifact.

    Only ``machine_id`` is guaranteed: the worker fills the rest from what the host can report, and
    a field it cannot determine is absent rather than guessed at.
    """

    model_config = ConfigDict(extra="ignore", frozen=True)

    machine_id: str
    hostname: str | None = None
    gpu_model: str | None = None
    vram_mb: int | None = None
    driver_version: str | None = None
    os: str | None = None
    worker_version: str | None = None
    hordelib_version: str | None = None
    torch_version: str | None = None


class CorpusDefinition(BaseModel):
    """The definition artifact a pricing-corpus run emits beside its stats stream."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    scenario_name: str
    scenario_revision: str
    tier: str
    warmup_job_count: int
    cells: tuple[DefinitionCell, ...]
    jobs: tuple[DefinitionJob, ...]
    created_at: float | None = None
    """Epoch seconds the artifact was written; its session's ``session_start`` follows shortly after.

    Artifacts that predate the field carry no write time, which is why session discovery is an
    opt-in path rather than the default.
    """

    machine: MachineDescriptor | None = None
    """The machine that ran the corpus, when the worker recorded it."""

    manifest_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    """The feature manifest revision the run encoded its cells under, when the worker stamped it.

    Artifacts that predate the field carry no revision and are assembled without the check.
    """

    def cells_by_id(self) -> dict[str, DefinitionCell]:
        """Return the cells keyed by cell id."""
        return {cell.cell_id: cell for cell in self.cells}


class BundleFile(BaseModel):
    """One file a run bundle's manifest lists, with what it must hash and weigh."""

    model_config = ConfigDict(extra="ignore", frozen=True, populate_by_name=True)

    name: str
    role: str
    """Whether the file is the run's ``definition`` artifact or one of its ``stats`` parts."""

    sha256: str
    size_bytes: int = Field(alias="bytes")


class BundleManifest(BaseModel):
    """A run bundle's manifest: which files belong to the run and what they must hash to."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    bundle_format: str
    files: tuple[BundleFile, ...]


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


def format_machines_table(machines: dict[str, dict[str, Any]]) -> str:
    """Render the trusted-machines table as aligned text, one row per machine."""
    columns = ("gpu_model", "vram_mb", "cpu_class", "os", "notes")
    header = ("machine_id", *columns)
    rows = [
        (machine_id, *("" if (value := entry.get(column)) is None else str(value) for column in columns))
        for machine_id, entry in sorted(machines.items())
    ]
    widths = [max(len(row[index]) for row in (header, *rows)) for index in range(len(header))]
    return "\n".join(
        "  ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=True)).rstrip()
        for row in (header, *rows)
    )


def _toml_value(value: str | int) -> str:
    """Render a scalar as TOML. JSON string escaping is a subset of TOML basic-string escaping."""
    return str(value) if isinstance(value, int) else json.dumps(value)


def warn_on_machine_facts_mismatch(machine: MachineDescriptor, entry: dict[str, Any]) -> list[str]:
    """Warn when a registered machine id is stamped with different hardware facts.

    Clocks are calibrated per machine id, so the same id measured on different hardware would pool
    two machines' timings into one and map both onto the reference wrongly. A fact either side does
    not report is not a difference.

    Args:
        machine: The descriptor read from a definition artifact.
        entry: The machines-table entry already registered under that id.

    Returns:
        The names of the facts that differ, in table order.
    """
    differing: list[str] = []
    for fact in _MACHINE_FACT_FIELDS:
        stamped = getattr(machine, fact)
        recorded = entry.get(fact)
        if stamped is None or recorded is None or stamped == recorded:
            continue
        differing.append(fact)
        logger.warning(
            f"machine {machine.machine_id!r} is registered with {fact} {recorded!r} but this run stamps "
            f"{stamped!r}; the same id on different hardware pools two machines' clocks as one",
        )
    return differing


def add_machine(
    machine: MachineDescriptor,
    *,
    notes: str | None = None,
    machines_path: Path | None = None,
) -> dict[str, Any]:
    """Append a machine to the trusted-machines table.

    Registration is the act of trusting a machine's measurements, so an id already present is an
    error rather than an update: re-registering would silently retag data measured elsewhere.

    Args:
        machine: The descriptor read from a definition artifact.
        notes: Free-text note recorded beside the entry.
        machines_path: Alternate machines table, for tests.

    Returns:
        The written entry.

    Raises:
        AssemblyError: If the id is already registered.
    """
    resolved = machines_path if machines_path is not None else DEFAULT_MACHINES_PATH
    with resolved.open("rb") as handle:
        existing = tomllib.load(handle).get("machines") or {}
    if machine.machine_id in existing:
        entry = existing[machine.machine_id]
        warn_on_machine_facts_mismatch(machine, entry)
        rendered = "\n".join(f"  {key} = {value!r}" for key, value in sorted(entry.items()))
        raise AssemblyError(f"{resolved} already declares machine {machine.machine_id!r}:\n{rendered}")

    fields: list[tuple[str, str | int]] = []
    if machine.gpu_model is not None:
        fields.append(("gpu_model", machine.gpu_model))
    if machine.vram_mb is not None:
        fields.append(("vram_mb", machine.vram_mb))
    if machine.os is not None:
        fields.append(("os", machine.os))
    if notes is not None:
        fields.append(("notes", notes))

    body = f"\n[machines.{machine.machine_id}]\n" + "".join(f"{key} = {_toml_value(value)}\n" for key, value in fields)
    with resolved.open("a", encoding="utf-8") as handle:
        handle.write(body)
    return dict(fields)


@dataclass(frozen=True)
class SessionInputs:
    """The stats session and machine a definition artifact points at."""

    definition_path: Path
    definition: CorpusDefinition
    stats_paths: tuple[Path, ...]
    """The session's parts in rotation order; the first carries ``session_start``."""

    machine: MachineDescriptor | None


_SESSION_SEARCH_WINDOW_SECONDS = 3600.0
"""How long after an artifact is written its session may still start.

The worker takes tens of seconds to reach its first job, and a stats directory accumulates many
sessions, so the match is the nearest start at or after the artifact rather than any start on the
day.
"""


def resolve_session(definition_path: Path) -> SessionInputs:
    """Find the stats session a definition artifact was written for.

    The artifact and its session land in the same directory, so the session is identified by its
    scenario revision and by starting just after the artifact was written.

    Args:
        definition_path: A pricing-corpus definition artifact.

    Returns:
        The definition, its session's parts in rotation order, and the machine it names.

    Raises:
        AssemblyError: If a bundle manifest beside the artifact does not describe the files on disk,
            the artifact predates the recorded write time, or no session in its directory matches
            it.
    """
    definition = _load_definition(definition_path)

    bundle_path = definition_path.parent / BUNDLE_MANIFEST_FILENAME
    if bundle_path.is_file():
        bundle = _verify_bundle(bundle_path)
        bundle_stats_paths = _bundle_stats_paths(bundle, bundle_path.parent)
        if bundle_stats_paths:
            return SessionInputs(
                definition_path=definition_path,
                definition=definition,
                stats_paths=bundle_stats_paths,
                machine=definition.machine,
            )

    if definition.created_at is None:
        raise AssemblyError(
            f"{definition_path} carries no created_at, so its stats session cannot be identified; "
            "pass --stats, --definition and --machine explicitly",
        )

    best: tuple[float, Path] | None = None
    for candidate in sorted(definition_path.parent.glob("stats-v*-000.jsonl*")):
        if candidate.suffix not in (".jsonl", ".gz"):
            continue
        session_start = _read_session_start(candidate)
        if session_start is None:
            continue
        if session_start.config.scenario_id != PRICING_CORPUS_SCENARIO_ID:
            continue
        if session_start.config.scenario_revision != definition.scenario_revision:
            continue
        delay = session_start.timestamp - definition.created_at
        if delay < 0 or delay > _SESSION_SEARCH_WINDOW_SECONDS:
            continue
        if best is None or delay < best[0]:
            best = (delay, candidate)

    if best is None:
        raise AssemblyError(
            f"no stats session in {definition_path.parent} starts within "
            f"{_SESSION_SEARCH_WINDOW_SECONDS:.0f}s of {definition_path.name} at scenario revision "
            f"{definition.scenario_revision!r}; pass --stats and --definition explicitly",
        )

    return SessionInputs(
        definition_path=definition_path,
        definition=definition,
        stats_paths=tuple(_group_rotated_parts([best[1]])[0]),
        machine=definition.machine,
    )


def _verify_bundle(bundle_path: Path) -> BundleManifest:
    """Verify every file a bundle manifest lists against its recorded hash and size.

    A bundle crosses a machine boundary as an archive, so a part that was truncated, edited or lost
    in transit would otherwise be assembled as if it had been measured.

    Args:
        bundle_path: The bundle manifest beside the definition artifact.

    Returns:
        The parsed manifest.

    Raises:
        AssemblyError: On a bundle layout this reader does not know, or at the first listed file
            that is missing or does not match what the manifest records for it.
    """
    bundle = BundleManifest.model_validate_json(bundle_path.read_text(encoding="utf-8"))
    if bundle.bundle_format not in _SUPPORTED_BUNDLE_FORMATS:
        known = ", ".join(sorted(_SUPPORTED_BUNDLE_FORMATS))
        raise AssemblyError(
            f"{bundle_path} declares bundle_format {bundle.bundle_format!r}; this reader knows: {known}",
        )

    for entry in bundle.files:
        path = bundle_path.parent / entry.name
        if not path.is_file():
            raise AssemblyError(f"{bundle_path} lists {entry.name}, which is missing from {bundle_path.parent}")
        size_bytes = path.stat().st_size
        if size_bytes != entry.size_bytes:
            raise AssemblyError(f"{path} is {size_bytes} bytes; {bundle_path.name} lists {entry.size_bytes}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != entry.sha256:
            raise AssemblyError(f"{path} hashes to {digest}; {bundle_path.name} lists {entry.sha256}")
    return bundle


def _bundle_stats_paths(bundle: BundleManifest, bundle_dir: Path) -> tuple[Path, ...]:
    """Return the bundle's stats parts in rotation order, so the part carrying session_start is first."""
    parts = [bundle_dir / entry.name for entry in bundle.files if entry.role == "stats"]
    return tuple(sorted(parts, key=_rotation_sort_key))


def _rotation_sort_key(stats_path: Path) -> tuple[int, str]:
    """Order stats parts by rotation index, keeping names outside the worker's scheme sorted by name."""
    match = _ROTATED_STATS_RE.match(stats_path.name)
    return (int(match.group("index")) if match is not None else 0, stats_path.name)


def _read_session_start(stats_path: Path) -> "_SessionStart | None":
    """Read a stats file's ``session_start`` without parsing the records behind it."""
    opener = gzip.open if stats_path.suffix == ".gz" else open
    with opener(stats_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                event = json.loads(stripped)
            except json.JSONDecodeError:
                return None
            if event.get("event") == "session_start":
                return _SessionStart.model_validate(event)
    return None


def assemble(
    stats_paths: list[Path],
    *,
    machine_id: str,
    out_dir: Path,
    definition_paths: list[Path] | None = None,
    machines_path: Path | None = None,
    resolve_baselines: bool = True,
    allow_manifest_mismatch: bool = False,
) -> AssemblyResult:
    """Assemble stats files from one machine into a parquet snapshot.

    Args:
        stats_paths: Stats JSONL files. The worker's rotated parts of one session are read together;
            naming any part of a session is enough.
        machine_id: The measuring machine's id; must exist in the machines table.
        out_dir: Directory the snapshot is written into (created if absent).
        definition_paths: Pricing-corpus definition artifacts. Required for every
            pricing-corpus session among *stats_paths*; matched by scenario revision.
        machines_path: Alternate machines table, for tests.
        resolve_baselines: Whether to correct baselines through the model reference, which may
            fetch from the network. Disable for offline runs; rows then keep the worker-reported
            value with ``baseline_resolved`` False.
        allow_manifest_mismatch: Whether to downgrade a definition stamped with another feature
            manifest revision from a refusal to a warning.

    Returns:
        The written snapshot and per-session reports.

    Raises:
        AssemblyError: If the machine is unknown, a session is malformed, a definition was encoded
            under another feature manifest revision, or a corpus session has no matching
            definition.
        PairingError: If corpus records diverge from the definition's job list.
    """
    machines = load_machines(machines_path)
    if machine_id not in machines:
        known = ", ".join(sorted(machines))
        raise AssemblyError(f"unknown machine id {machine_id!r}; machines.toml knows: {known}")

    definitions: list[CorpusDefinition] = []
    for path in definition_paths or []:
        definition = _load_definition(path)
        _check_manifest_revision(definition, path, allow_mismatch=allow_manifest_mismatch)
        definitions.append(definition)

    rows: list[SnapshotRow] = []
    reports: list[SessionReport] = []
    for session_parts in _group_rotated_parts(stats_paths):
        session_rows, report = _assemble_session(session_parts, machine_id=machine_id, definitions=definitions)
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


def _check_manifest_revision(definition: CorpusDefinition, definition_path: Path, *, allow_mismatch: bool) -> None:
    """Refuse a definition whose cells were encoded under another feature manifest revision.

    The manifest is what a cell's features mean, so rows labeled under one revision and trained
    under another carry labels the model cannot reproduce. A definition that stamps no revision
    predates the field and is taken as it is.

    Raises:
        AssemblyError: On a mismatch, unless *allow_mismatch* downgrades it to a warning.
    """
    stamped = definition.manifest_sha256
    if stamped is None:
        return
    shipped = default_manifest().content_sha256()
    if stamped == shipped:
        return

    message = (
        f"{definition_path} carries manifest_sha256 {stamped}, but the shipped feature manifest is "
        f"{shipped}; its rows were encoded under another manifest revision"
    )
    if allow_mismatch:
        logger.warning(f"{message}; assembling anyway as asked")
        return
    raise AssemblyError(f"{message}; re-encode the run or pass --allow-manifest-mismatch")


def _assemble_session(
    session_parts: list[Path],
    *,
    machine_id: str,
    definitions: list[CorpusDefinition],
) -> tuple[list[SnapshotRow], SessionReport]:
    """Assemble one session, given in rotation order, into rows.

    The session is named after its first part throughout the snapshot and the report.
    """
    stats_path = session_parts[0]
    session_start, records = _parse_stats_file(session_parts)
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


_ROTATED_STATS_RE = re.compile(r"^(?P<stem>stats-v.+-\d{8}-\d{6})-(?P<index>\d+)\.jsonl(?:\.gz)?$")
"""The worker's stats file name: one session stamp, rotated into numbered parts at a size limit."""


def _group_rotated_parts(stats_paths: list[Path]) -> list[list[Path]]:
    """Group stats paths into sessions, each ordered by rotation index.

    The worker starts a new numbered part every few megabytes of a session, and only the first part
    carries ``session_start``, so a session is read as the whole sequence. Naming any one part of a
    session pulls in every sibling part in that directory; a file outside the worker's naming scheme is
    a session on its own. The order of first appearance among *stats_paths* is preserved.
    """
    sessions: dict[tuple[Path, str], dict[int, Path]] = {}
    ordered_keys: list[tuple[Path, str]] = []
    for stats_path in stats_paths:
        match = _ROTATED_STATS_RE.match(stats_path.name)
        if match is None:
            key = (stats_path.parent, stats_path.name)
            if key not in sessions:
                sessions[key] = {0: stats_path}
                ordered_keys.append(key)
            continue
        key = (stats_path.parent, match.group("stem"))
        if key in sessions:
            continue
        parts: dict[int, Path] = {}
        for sibling in stats_path.parent.iterdir():
            sibling_match = _ROTATED_STATS_RE.match(sibling.name)
            if sibling_match is not None and sibling_match.group("stem") == match.group("stem"):
                parts[int(sibling_match.group("index"))] = sibling
        parts.setdefault(int(match.group("index")), stats_path)
        sessions[key] = parts
        ordered_keys.append(key)
    return [[parts[index] for index in sorted(parts)] for parts in (sessions[key] for key in ordered_keys)]


def _parse_stats_file(session_parts: list[Path]) -> tuple[_SessionStart, list[_JobCompletedEvent]]:
    """Parse one session's stats JSONL parts into its session_start and job_completed events."""
    session_starts: list[_SessionStart] = []
    records: list[_JobCompletedEvent] = []
    for stats_path in session_parts:
        # The worker compresses rotated stats files in place, so a corpus session is as likely to
        # arrive as ``.jsonl.gz`` as plain JSONL.
        opener = gzip.open if stats_path.suffix == ".gz" else open
        with opener(stats_path, "rt", encoding="utf-8") as handle:
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
        names = ", ".join(path.name for path in session_parts)
        raise AssemblyError(f"{names} holds {len(session_starts)} session_start events; expected exactly 1")
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
    "BUNDLE_MANIFEST_FILENAME",
    "DEFAULT_MACHINES_PATH",
    "PRICING_CORPUS_SCENARIO_ID",
    "AssemblyError",
    "AssemblyResult",
    "BundleFile",
    "BundleManifest",
    "CorpusDefinition",
    "DefinitionCell",
    "DefinitionJob",
    "MachineDescriptor",
    "PairingError",
    "SessionInputs",
    "SessionReport",
    "add_machine",
    "assemble",
    "format_machines_table",
    "load_machines",
    "resolve_session",
    "warn_on_machine_facts_mismatch",
]
