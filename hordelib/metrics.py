"""In-process performance metrics for consumers that cannot read an OTel backend.

The logfire instrumentation throughout hordelib measures model-load phases, sampling
progress, and ad-hoc download performance, but ships those numbers to an external
collector. Embedding applications (notably the AI-Horde worker and its benchmark
harness) need the same numbers in-process, immediately after each job.

This module is the comfy-free accumulation point: producers inside the inference
process dual-write to logfire and to the :class:`MetricsCollector` singleton, and the
embedder calls :meth:`MetricsCollector.snapshot_and_reset_job` after each job (and
:meth:`MetricsCollector.drain_download_events` periodically, since downloads complete
on background threads independent of job boundaries).

Importing this module must never trigger a ComfyUI or torch import — it is part of the
public API surface re-exported from :mod:`hordelib.api` and is contract-tested to be
importable without ``hordelib.initialise()``.
"""

import threading
import time
from typing import Literal

from pydantic import BaseModel

ModelLoadPhase = Literal["disk_to_ram", "ram_to_vram"]

DownloadCategory = Literal["lora", "ti", "checkpoint", "other"]


class ModelLoadEvent(BaseModel):
    """One observed model-load phase (disk read or GPU transfer)."""

    model_name: str
    phase: ModelLoadPhase
    duration_seconds: float
    timestamp: float
    """Epoch time the phase completed."""


class SamplingStats(BaseModel):
    """Aggregate sampling progress observed during one job."""

    steps_completed: int
    total_steps: int
    duration_seconds: float
    """Wall time from the first to the last observed progress sample."""
    iterations_per_second: float
    """Overall rate: ``steps_completed / duration_seconds`` (0 when duration is 0)."""
    its_samples: list[float] = []
    """Instantaneous rates between consecutive progress samples, for percentile math."""


class DownloadEvent(BaseModel):
    """One completed (or definitively failed) ad-hoc model download."""

    name: str
    category: DownloadCategory
    size_bytes: int
    duration_seconds: float
    megabytes_per_second: float
    retries: int
    success: bool
    timestamp: float
    """Epoch time the download finished (or was given up on)."""


class JobPhaseMetrics(BaseModel):
    """Everything the collector observed between two job snapshots."""

    model_loads: list[ModelLoadEvent] = []
    sampling: SamplingStats | None = None
    downloads: list[DownloadEvent] = []
    """Left empty by the collector itself; embedders may attach drained download
    events when composing a per-job record."""
    vram_used_high_water_mb: int | None = None
    ram_used_high_water_mb: int | None = None
    phase_seconds: dict[str, float] = {}
    """Total seconds spent in named non-sampling GPU phases this job (e.g. ``vae_decode``,
    ``vae_encode``). Lets an embedder see where time goes between sampling runs."""


class MetricsCollector:
    """Thread-safe accumulator for per-job performance events.

    Producers (the checkpoint loader, the comfy GPU-load wrapper, the native progress
    hook, the ad-hoc download managers) call the ``record_*`` methods from whatever
    thread they run on. The embedder snapshots per-job state after each job and drains
    download events on its own schedule.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._model_loads: list[ModelLoadEvent] = []
        self._download_events: list[DownloadEvent] = []
        self._vram_used_high_water_mb: int | None = None
        self._ram_used_high_water_mb: int | None = None
        self._phase_seconds: dict[str, float] = {}
        self._reset_sampling_locked()

    def _reset_sampling_locked(self) -> None:
        self._sampling_first_ts: float | None = None
        self._sampling_last_ts: float | None = None
        self._sampling_last_step: int | None = None
        self._sampling_total_steps: int = 0
        self._sampling_steps_completed: int = 0
        self._sampling_its_samples: list[float] = []

    def record_model_load(self, event: ModelLoadEvent) -> None:
        """Record one completed model-load phase."""
        with self._lock:
            self._model_loads.append(event)

    def record_download(self, event: DownloadEvent) -> None:
        """Record one finished ad-hoc download attempt (success or terminal failure)."""
        with self._lock:
            self._download_events.append(event)

    def record_phase(self, name: str, duration_seconds: float) -> None:
        """Accumulate time spent in a named non-sampling phase for the current job."""
        if duration_seconds <= 0:
            return
        with self._lock:
            self._phase_seconds[name] = self._phase_seconds.get(name, 0.0) + duration_seconds

    def record_sampling_step(self, step: int, total: int, timestamp: float | None = None) -> None:
        """Record one absolute progress sample from the sampler.

        A step value lower than the previous sample starts a new sampling phase (e.g.
        the second pass of hires-fix); completed steps accumulate across phases.
        """
        now = timestamp if timestamp is not None else time.time()
        with self._lock:
            if self._sampling_first_ts is None:
                self._sampling_first_ts = now
            if total > 0:
                self._sampling_total_steps = total

            if self._sampling_last_step is not None and step >= self._sampling_last_step:
                step_delta = step - self._sampling_last_step
                time_delta = now - (self._sampling_last_ts or now)
                if step_delta > 0:
                    self._sampling_steps_completed += step_delta
                    if time_delta > 0:
                        self._sampling_its_samples.append(step_delta / time_delta)
            else:
                # First sample of a phase: count the step itself (samplers report 1-based).
                self._sampling_steps_completed += max(step, 0)

            self._sampling_last_step = step
            self._sampling_last_ts = now

    def record_memory_sample(self, *, vram_used_mb: int | None = None, ram_used_mb: int | None = None) -> None:
        """Record a point-in-time memory usage sample; high-water marks are kept."""
        with self._lock:
            if vram_used_mb is not None and (
                self._vram_used_high_water_mb is None or vram_used_mb > self._vram_used_high_water_mb
            ):
                self._vram_used_high_water_mb = vram_used_mb
            if ram_used_mb is not None and (
                self._ram_used_high_water_mb is None or ram_used_mb > self._ram_used_high_water_mb
            ):
                self._ram_used_high_water_mb = ram_used_mb

    def _sampling_stats_locked(self) -> SamplingStats | None:
        if self._sampling_first_ts is None or self._sampling_last_ts is None:
            return None
        duration = self._sampling_last_ts - self._sampling_first_ts
        return SamplingStats(
            steps_completed=self._sampling_steps_completed,
            total_steps=self._sampling_total_steps,
            duration_seconds=duration,
            iterations_per_second=(self._sampling_steps_completed / duration) if duration > 0 else 0.0,
            its_samples=list(self._sampling_its_samples),
        )

    def snapshot_and_reset_job(self) -> JobPhaseMetrics:
        """Return everything observed since the last snapshot and reset per-job state.

        Download events are *not* included or reset here — they complete on background
        threads with no job affinity; use :meth:`drain_download_events`.
        """
        with self._lock:
            snapshot = JobPhaseMetrics(
                model_loads=list(self._model_loads),
                sampling=self._sampling_stats_locked(),
                vram_used_high_water_mb=self._vram_used_high_water_mb,
                ram_used_high_water_mb=self._ram_used_high_water_mb,
                phase_seconds=dict(self._phase_seconds),
            )
            self._model_loads = []
            self._vram_used_high_water_mb = None
            self._ram_used_high_water_mb = None
            self._phase_seconds = {}
            self._reset_sampling_locked()
            return snapshot

    def drain_download_events(self) -> list[DownloadEvent]:
        """Return and clear all download events recorded since the last drain."""
        with self._lock:
            events = self._download_events
            self._download_events = []
            return events


_collector = MetricsCollector()


def get_metrics_collector() -> MetricsCollector:
    """Return the process-wide metrics collector."""
    return _collector
