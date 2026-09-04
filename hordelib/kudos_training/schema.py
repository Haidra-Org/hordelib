"""Column contract for kudos-training snapshots.

Every pipeline stage reads and writes snapshots through the names defined here, so a renamed or
retyped column is a visible edit to this module rather than a silent drift between stages. The
row model is the pydantic validation applied at assembly; parquet is the storage format.
"""

from enum import StrEnum

from horde_sdk.generation_parameters.image.sampler_work import SamplerExecutionContractVersion
from pydantic import BaseModel, ConfigDict

SNAPSHOT_SCHEMA_VERSION = 2
"""Version of the snapshot column contract, recorded in every snapshot's metadata sidecar."""


class SourceKind(StrEnum):
    """Provenance class of a snapshot row.

    The pricing corpus is the primary training data; production records are admissible only as a
    validation overlay and covariance reference, and carry this label so no stage can confuse the
    two.
    """

    CORPUS = "corpus"
    PRODUCTION = "production"


class SnapshotRow(BaseModel):
    """One assembled training row: request-side features, measured targets, and labels.

    Feature columns stay in payload form (names, counts, flags); encoding through the feature
    manifest happens at training time so a snapshot outlives manifest revisions.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    # --- identity and provenance -----------------------------------------------------------
    machine_id: str
    """Key into ``machines.toml``; every row is labeled with the machine that measured it."""

    source_kind: SourceKind
    """Whether the row came from a pricing-corpus run or a production session."""

    stats_file: str
    """Basename of the stats JSONL file the row was read from."""

    job_id: str
    """The worker's job id, for tracing a row back to its record."""

    worker_version: str
    """Worker version that produced the session."""

    horde_sdk_version: str | None
    """SDK version whose sampler semantics the worker implemented, when recorded."""

    sampler_constraints_artifact_sha256: str | None
    """SHA-256 of the SDK sampler-constraints artifact used by the worker, when recorded."""

    sampler_execution_contract_version: SamplerExecutionContractVersion | None
    """SDK execution contract implemented by the worker, used as provenance rather than a feature."""

    session_started_at: float
    """Unix timestamp of the session's ``session_start`` event."""

    time_popped: float
    """Unix timestamp the job was popped; the row-ordering key within a session."""

    scenario_id: str | None
    """Scenario provenance from the session config, when the session declared one."""

    scenario_revision: str | None
    """Scenario revision from the session config, when the session declared one."""

    # --- corpus cell labels (None on production rows) --------------------------------------
    cell_id: str | None
    """Pricing-corpus cell this job realizes."""

    cell_group: str | None
    """The cell's group (g1..g11 or warmup)."""

    replicate: int | None
    """Which replicate of its cell this job is."""

    permutation: str | None
    """``warmup`` or the shuffle seed of the permutation block the job belongs to."""

    position: int | None
    """Zero-based position in the corpus job list, warmup included."""

    source_processing: str | None
    """The cell's source mode (txt2img/img2img/inpainting); not observable from the record."""

    lora_role: str | None
    """``miss`` or ``hit`` for the labeled lora-cache cells, else None."""

    cold_cell: bool
    """Whether the cell deliberately schedules a model switch (labeled cold-load measurement)."""

    warmup: bool
    """Whether the job belongs to the warmup block, which is excluded from training."""

    # --- request-side features (payload form) ----------------------------------------------
    model_name: str
    """Model the job ran."""

    baseline: str | None
    """Model baseline: resolved from the model reference when possible, else as the worker reported.

    The worker's stats field is unreliable under the harness (observed reporting
    ``stable_diffusion_1`` for SDXL models), so the assembler re-resolves it by model name.
    """

    baseline_resolved: bool
    """True when the baseline came from the model reference rather than the worker record."""

    width: int
    """Requested image width in pixels."""

    height: int
    """Requested image height in pixels."""

    trajectory_steps: int
    """Requested denoising-trajectory step count, before any sampler-specific work expansion."""

    cfg_scale: float | None
    """Requested guidance scale; None on degraded records."""

    denoising_strength: float | None
    """Denoising strength, from the corpus cell spec; the stats record does not carry it."""

    sampler_name: str | None
    """Requested sampler; None on degraded records."""

    scheduler: str | None
    """Requested sigma schedule; None on degraded records."""

    n_images: int
    """Batch count; the server-side multiplication this model absorbs as a feature."""

    loras_count: int
    """Number of loras the job applied."""

    tis_count: int
    """Number of textual inversions the job applied."""

    control_type: str | None
    """Controlnet type, when the job used one."""

    hires_fix: bool
    """Whether the job ran the two-pass hires fix."""

    post_processing: tuple[str, ...]
    """Post-processor chain the job ran, in order."""

    is_alchemy: bool
    """Whether the record is an alchemy job; such rows are excluded from image-model training."""

    degraded_features: bool
    """True when sampler/scheduler/cfg are absent (records predating worker commit 3bde5749)."""

    # --- measured targets and diagnostics ---------------------------------------------------
    sampler_window_seconds: float | None
    """Primary target: occupancy of the serialized sampler resource (see assembler window rule)."""

    e2e_seconds: float
    """Pop to finalized; diagnostic only, never a target."""

    sampling_seconds: float | None
    """The comfy sampling phase alone, as reported by the worker."""

    queue_wait_seconds: float
    """Pop to inference start; carries lora fetch waits on miss cells."""

    safety_seconds: float | None
    """Safety-check duration, when the record carries one."""

    kudos_reward: float | None
    """Kudos the horde paid for the job; zero on harness runs."""

    faulted: bool
    """Faulted rows are carried into the snapshot and dropped by sanitation rule 1."""

    stage_order_ok: bool
    """Whether the record's stage timestamps are monotonic; sanitation rule 3 reads this."""


SNAPSHOT_COLUMNS: tuple[str, ...] = tuple(SnapshotRow.model_fields)
"""Snapshot column names, in row-model declaration order."""

__all__ = [
    "SNAPSHOT_COLUMNS",
    "SNAPSHOT_SCHEMA_VERSION",
    "SnapshotRow",
    "SourceKind",
]
