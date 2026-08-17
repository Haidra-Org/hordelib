"""Unit tests for the in-process MetricsCollector (no comfy, no GPU)."""

import pytest

from hordelib.metrics import (
    DownloadEvent,
    JobPhaseMetrics,
    JobVramFootprint,
    MetricsCollector,
    ModelLoadEvent,
)


@pytest.fixture
def collector() -> MetricsCollector:
    return MetricsCollector()


def _load_event(phase: str = "disk_to_ram") -> ModelLoadEvent:
    return ModelLoadEvent(
        model_name="Deliberate",
        phase=phase,  # type: ignore[arg-type]
        duration_seconds=1.5,
        timestamp=1000.0,
    )


def _download_event(success: bool = True) -> DownloadEvent:
    return DownloadEvent(
        name="some lora",
        category="lora",
        size_bytes=150 * 1024 * 1024,
        duration_seconds=10.0,
        megabytes_per_second=15.0,
        retries=0,
        success=success,
        timestamp=1000.0,
    )


class TestSnapshotAndReset:
    def test_empty_snapshot(self, collector: MetricsCollector) -> None:
        snapshot = collector.snapshot_and_reset_job()
        assert snapshot.model_loads == []
        assert snapshot.sampling is None
        assert snapshot.downloads == []
        assert snapshot.vram_used_high_water_mb is None
        assert snapshot.ram_used_high_water_mb is None

    def test_model_loads_collected_and_reset(self, collector: MetricsCollector) -> None:
        collector.record_model_load(_load_event("disk_to_ram"))
        collector.record_model_load(_load_event("ram_to_vram"))

        snapshot = collector.snapshot_and_reset_job()
        assert [event.phase for event in snapshot.model_loads] == ["disk_to_ram", "ram_to_vram"]

        assert collector.snapshot_and_reset_job().model_loads == []

    def test_memory_high_water_keeps_max_and_resets(self, collector: MetricsCollector) -> None:
        collector.record_memory_sample(vram_used_mb=4000, ram_used_mb=10_000)
        collector.record_memory_sample(vram_used_mb=7000, ram_used_mb=9_000)
        collector.record_memory_sample(vram_used_mb=5000)

        snapshot = collector.snapshot_and_reset_job()
        assert snapshot.vram_used_high_water_mb == 7000
        assert snapshot.ram_used_high_water_mb == 10_000

        assert collector.snapshot_and_reset_job().vram_used_high_water_mb is None

    def test_downloads_not_included_in_job_snapshot(self, collector: MetricsCollector) -> None:
        collector.record_download(_download_event())
        assert collector.snapshot_and_reset_job().downloads == []
        # ... and the snapshot must not have consumed them either.
        assert len(collector.drain_download_events()) == 1


class TestComponentCacheCounters:
    def test_empty_snapshot_defaults(self, collector: MetricsCollector) -> None:
        snapshot = collector.snapshot_and_reset_job()
        assert snapshot.component_cache_hits == 0
        assert snapshot.component_cache_misses == 0
        assert snapshot.component_cache_evictions == 0
        assert snapshot.component_cache_held_mb is None

    def test_counts_accumulate_and_reset(self, collector: MetricsCollector) -> None:
        collector.record_component_cache_hit()
        collector.record_component_cache_hit()
        collector.record_component_cache_miss()
        collector.record_component_cache_evictions(3)
        collector.record_component_cache_evictions(0)  # a no-op count is ignored
        collector.record_component_cache_held_mb(1234.5)

        snapshot = collector.snapshot_and_reset_job()
        assert snapshot.component_cache_hits == 2
        assert snapshot.component_cache_misses == 1
        assert snapshot.component_cache_evictions == 3
        assert snapshot.component_cache_held_mb == 1234.5

        after = collector.snapshot_and_reset_job()
        assert after.component_cache_hits == 0
        assert after.component_cache_misses == 0
        assert after.component_cache_evictions == 0
        assert after.component_cache_held_mb is None


class TestSampling:
    def test_steady_progress_rates(self, collector: MetricsCollector) -> None:
        # 1-based sampler steps, one step per second
        for step, ts in [(1, 100.0), (2, 101.0), (3, 102.0), (4, 103.0)]:
            collector.record_sampling_step(step, 4, ts)

        stats = collector.snapshot_and_reset_job().sampling
        assert stats is not None
        assert stats.steps_completed == 4
        assert stats.total_steps == 4
        assert stats.duration_seconds == pytest.approx(3.0)
        assert stats.iterations_per_second == pytest.approx(4 / 3)
        assert stats.its_samples == pytest.approx([1.0, 1.0, 1.0])

    def test_step_reset_starts_new_phase(self, collector: MetricsCollector) -> None:
        # First pass: 2 steps; second pass (e.g. hires fix) restarts at 1.
        collector.record_sampling_step(1, 2, 100.0)
        collector.record_sampling_step(2, 2, 101.0)
        collector.record_sampling_step(1, 3, 102.0)
        collector.record_sampling_step(3, 3, 103.0)

        stats = collector.snapshot_and_reset_job().sampling
        assert stats is not None
        assert stats.steps_completed == 2 + 1 + 2
        assert stats.total_steps == 3

    def test_zero_duration_rate_is_zero(self, collector: MetricsCollector) -> None:
        collector.record_sampling_step(1, 1, 100.0)
        stats = collector.snapshot_and_reset_job().sampling
        assert stats is not None
        assert stats.iterations_per_second == 0.0


class TestDownloadDrain:
    def test_drain_returns_and_clears(self, collector: MetricsCollector) -> None:
        collector.record_download(_download_event(success=True))
        collector.record_download(_download_event(success=False))

        events = collector.drain_download_events()
        assert [event.success for event in events] == [True, False]
        assert collector.drain_download_events() == []


class TestVramFootprint:
    def test_absent_by_default(self, collector: MetricsCollector) -> None:
        assert collector.snapshot_and_reset_job().vram_footprint is None

    def test_recorded_then_reset(self, collector: MetricsCollector) -> None:
        collector.record_vram_footprint(JobVramFootprint(stage="whole_job", model_name="Deliberate"))

        snapshot = collector.snapshot_and_reset_job()
        assert snapshot.vram_footprint is not None
        assert snapshot.vram_footprint.model_name == "Deliberate"

        assert collector.snapshot_and_reset_job().vram_footprint is None

    def test_latest_record_wins(self, collector: MetricsCollector) -> None:
        collector.record_vram_footprint(JobVramFootprint(peak_device_used_mb=1000.0))
        collector.record_vram_footprint(JobVramFootprint(peak_device_used_mb=2000.0))

        snapshot = collector.snapshot_and_reset_job()
        assert snapshot.vram_footprint is not None
        assert snapshot.vram_footprint.peak_device_used_mb == 2000.0

    def test_every_field_defaults_to_none(self) -> None:
        footprint = JobVramFootprint()
        assert footprint.model_dump() == dict.fromkeys(JobVramFootprint.model_fields)

    def test_round_trips_through_json(self) -> None:
        footprint = JobVramFootprint(
            peak_resident_weights_mb=13100.5,
            peak_device_used_mb=13500.0,
            sum_component_weights_mb=16400.0,
            resident_weights_after_job_mb=11500.0,
            model_name="Flux.1-Schnell fp8 (Compact)",
            baseline="flux_1",
            width=1024,
            height=1024,
            batch_size=1,
            stage="sample_stage",
        )
        metrics = JobPhaseMetrics(vram_footprint=footprint, vram_device_used_high_water_mb=13500)

        restored = JobPhaseMetrics.model_validate_json(metrics.model_dump_json())
        assert restored == metrics
        assert restored.vram_footprint == footprint

    def test_older_producer_payload_still_validates(self) -> None:
        """A payload from a producer predating the footprint fields deserializes with them unset."""
        restored = JobPhaseMetrics.model_validate_json('{"vram_used_high_water_mb": 9000}')
        assert restored.vram_used_high_water_mb == 9000
        assert restored.vram_device_used_high_water_mb is None
        assert restored.vram_footprint is None


class TestDeviceUsedHighWater:
    def test_tracked_separately_from_the_process_view(self, collector: MetricsCollector) -> None:
        collector.record_memory_sample(vram_used_mb=9000, vram_device_used_mb=12000)
        collector.record_memory_sample(vram_used_mb=11000, vram_device_used_mb=11500)

        snapshot = collector.snapshot_and_reset_job()
        assert snapshot.vram_used_high_water_mb == 11000
        assert snapshot.vram_device_used_high_water_mb == 12000

        assert collector.snapshot_and_reset_job().vram_device_used_high_water_mb is None
