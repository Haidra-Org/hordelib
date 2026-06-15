"""GPU-free unit tests for the always-on phase-timing hooks.

``phase_timing`` patches ``comfy.model_management.load_models_gpu`` and ``comfy.sd.VAE``
encode/decode to record durations into the MetricsCollector independently of logfire. These
tests inject fake comfy modules so the recording logic can be exercised without comfy/GPU.
"""

import sys
import types

import pytest

from hordelib.execution import phase_timing
from hordelib.metrics import MetricsCollector


class _Clock:
    """A perf_counter stand-in that returns successive scripted values."""

    def __init__(self, values: list[float]) -> None:
        self._values = list(values)

    def __call__(self) -> float:
        # Repeat the last value once exhausted, so an unexpected extra read doesn't IndexError.
        return self._values.pop(0) if len(self._values) > 1 else self._values[0]


@pytest.fixture(autouse=True)
def isolated_collector(monkeypatch: pytest.MonkeyPatch) -> MetricsCollector:
    """Fresh collector + reset the module install guard for every test."""
    collector = MetricsCollector()
    monkeypatch.setattr("hordelib.metrics._collector", collector)
    monkeypatch.setattr(phase_timing, "_installed", False)
    return collector


def _install_fake_model_management(monkeypatch: pytest.MonkeyPatch, load_impl) -> types.ModuleType:
    comfy_pkg = sys.modules.get("comfy") or types.ModuleType("comfy")
    monkeypatch.setitem(sys.modules, "comfy", comfy_pkg)
    mm = types.ModuleType("comfy.model_management")
    mm.load_models_gpu = load_impl  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "comfy.model_management", mm)
    # `from comfy import model_management` reads the attribute off the package object; binding it
    # here makes the fake win even when real comfy.model_management was already imported this process.
    monkeypatch.setattr(comfy_pkg, "model_management", mm, raising=False)
    return mm


def _install_fake_sd(monkeypatch: pytest.MonkeyPatch) -> type:
    comfy_pkg = sys.modules.get("comfy") or types.ModuleType("comfy")
    monkeypatch.setitem(sys.modules, "comfy", comfy_pkg)
    sd = types.ModuleType("comfy.sd")

    class _VAE:
        def decode(self, latent: object) -> str:
            return f"decoded:{latent}"

        def encode(self, image: object) -> str:
            return f"encoded:{image}"

    sd.VAE = _VAE  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "comfy.sd", sd)
    # `from comfy import sd` resolves `sd` as an attribute of the comfy package, not via
    # sys.modules; without this bind a real comfy.sd imported earlier in the process shadows the
    # fake and the patch lands on the real VAE, leaving phases["vae_decode"] unset (KeyError).
    monkeypatch.setattr(comfy_pkg, "sd", sd, raising=False)
    return _VAE


class TestModelLoadPatch:
    def test_records_ram_to_vram_above_threshold(
        self,
        monkeypatch: pytest.MonkeyPatch,
        isolated_collector: MetricsCollector,
    ) -> None:
        calls: list[tuple] = []
        mm = _install_fake_model_management(monkeypatch, lambda *a, **k: calls.append((a, k)))
        monkeypatch.setattr(phase_timing.time, "perf_counter", _Clock([0.0, 0.2]))

        assert phase_timing._patch_model_load() is True
        mm.load_models_gpu("modelA")

        snapshot = isolated_collector.snapshot_and_reset_job()
        assert [e.phase for e in snapshot.model_loads] == ["ram_to_vram"]
        assert snapshot.model_loads[0].duration_seconds == pytest.approx(0.2)
        assert calls == [(("modelA",), {})]  # original still invoked

    def test_skips_sub_threshold_noop_loads(
        self,
        monkeypatch: pytest.MonkeyPatch,
        isolated_collector: MetricsCollector,
    ) -> None:
        _install_fake_model_management(monkeypatch, lambda *a, **k: None)
        # 10ms < the 50ms record threshold -> a resident no-op, not a real transfer.
        monkeypatch.setattr(phase_timing.time, "perf_counter", _Clock([0.0, 0.01]))

        phase_timing._patch_model_load()
        sys.modules["comfy.model_management"].load_models_gpu()

        assert isolated_collector.snapshot_and_reset_job().model_loads == []

    def test_idempotent_does_not_double_wrap(
        self,
        monkeypatch: pytest.MonkeyPatch,
        isolated_collector: MetricsCollector,
    ) -> None:
        mm = _install_fake_model_management(monkeypatch, lambda *a, **k: None)

        assert phase_timing._patch_model_load() is True
        wrapped_once = mm.load_models_gpu
        # Second patch must detect the marker and leave the existing wrapper in place.
        assert phase_timing._patch_model_load() is True
        assert mm.load_models_gpu is wrapped_once

        monkeypatch.setattr(phase_timing.time, "perf_counter", _Clock([0.0, 0.2]))
        mm.load_models_gpu()
        # Exactly one record per call -> no double-counting from re-wrapping.
        assert len(isolated_collector.snapshot_and_reset_job().model_loads) == 1

    def test_records_once_when_wrapping_a_preexisting_wrapper(
        self,
        monkeypatch: pytest.MonkeyPatch,
        isolated_collector: MetricsCollector,
    ) -> None:
        # Simulate comfy_patches._load_models_gpu_hijack already wrapping load_models_gpu: the
        # phase-timing wrapper must record exactly once per call, not once per layer.
        inner_calls: list[int] = []

        def _preexisting_hijack(*args: object, **kwargs: object) -> None:
            inner_calls.append(1)

        mm = _install_fake_model_management(monkeypatch, _preexisting_hijack)
        monkeypatch.setattr(phase_timing.time, "perf_counter", _Clock([0.0, 0.2]))

        phase_timing._patch_model_load()
        mm.load_models_gpu()

        assert inner_calls == [1]
        assert len(isolated_collector.snapshot_and_reset_job().model_loads) == 1

    def test_returns_false_when_comfy_absent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "comfy", types.ModuleType("comfy"))
        monkeypatch.delitem(sys.modules, "comfy.model_management", raising=False)
        assert phase_timing._patch_model_load() is False


class TestVAEPatch:
    def test_records_decode_and_encode_phases(
        self,
        monkeypatch: pytest.MonkeyPatch,
        isolated_collector: MetricsCollector,
    ) -> None:
        vae_cls = _install_fake_sd(monkeypatch)
        monkeypatch.setattr(phase_timing.time, "perf_counter", _Clock([0.0, 0.1, 0.1, 0.3]))

        assert phase_timing._patch_vae() is True
        vae = vae_cls()
        assert vae.decode("L") == "decoded:L"  # original behavior preserved
        assert vae.encode("I") == "encoded:I"

        phases = isolated_collector.snapshot_and_reset_job().phase_seconds
        assert phases["vae_decode"] == pytest.approx(0.1)
        assert phases["vae_encode"] == pytest.approx(0.2)

    def test_idempotent_does_not_double_wrap(self, monkeypatch: pytest.MonkeyPatch) -> None:
        vae_cls = _install_fake_sd(monkeypatch)
        assert phase_timing._patch_vae() is True
        decode_once = vae_cls.decode
        assert phase_timing._patch_vae() is True
        assert vae_cls.decode is decode_once


class TestInstall:
    def test_install_idempotent_and_reports_state(
        self,
        monkeypatch: pytest.MonkeyPatch,
        isolated_collector: MetricsCollector,
    ) -> None:
        _install_fake_model_management(monkeypatch, lambda *a, **k: None)
        _install_fake_sd(monkeypatch)

        assert phase_timing.install_phase_timing_hooks() is True
        assert phase_timing._installed is True
        # Second install short-circuits on the guard.
        assert phase_timing.install_phase_timing_hooks() is True
