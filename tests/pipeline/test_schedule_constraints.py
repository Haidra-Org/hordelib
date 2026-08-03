"""Schedule selection and the solver/schedule compatibility constraint. No GPU required.

Two behaviours are covered. First, a horde request names a schedule only through its karras flag, so
an explicitly named schedule has to survive normalization rather than being overwritten by the flag.
Second, a sampler that diverges on the resolved schedule is moved onto one where it converges, because
the alternative is billing for colour noise.
"""

import pytest

from hordelib.pipeline.constants import (
    DIVERGENT_SCHEDULES,
    SAMPLERS_MAP,
    SCHEDULE_SENSITIVE_FALLBACK,
    SCHEDULE_SENSITIVE_SAMPLERS,
    SCHEDULERS,
    resolve_schedule,
)
from hordelib.pipeline.payload import ImageGenPayload


class TestScheduleVocabulary:
    def test_fallback_is_a_real_schedule(self):
        assert SCHEDULE_SENSITIVE_FALLBACK in SCHEDULERS

    def test_fallback_is_not_itself_divergent(self):
        # A fallback drawn from the divergent set would substitute one broken run for another.
        assert SCHEDULE_SENSITIVE_FALLBACK not in DIVERGENT_SCHEDULES

    def test_sensitive_samplers_are_real_samplers(self):
        for sampler in SCHEDULE_SENSITIVE_SAMPLERS:
            assert sampler in SAMPLERS_MAP, f"{sampler} is not an offered sampler"

    def test_divergent_schedules_are_real_schedules(self):
        for schedule in DIVERGENT_SCHEDULES:
            assert schedule in SCHEDULERS


class TestResolveSchedule:
    def test_sensitive_sampler_on_divergent_schedule_is_substituted(self):
        resolved, substituted = resolve_schedule("dpmpp_3m_sde", "normal")
        assert resolved == SCHEDULE_SENSITIVE_FALLBACK
        assert substituted is True

    def test_sensitive_sampler_on_a_convergent_schedule_is_left_alone(self):
        for schedule in ("karras", "simple", "sgm_uniform", "exponential"):
            resolved, substituted = resolve_schedule("dpmpp_3m_sde", schedule)
            assert resolved == schedule, schedule
            assert substituted is False, schedule

    def test_the_substitute_holds_across_the_step_range(self):
        # Measured: sweeping dpmpp_3m_sde over all nine schedules at 8 and 25 steps, only `simple` and
        # `sgm_uniform` converge at both. `karras` converges at 25 and fails at 8, so it cannot be the
        # substitute: a low-step request would be moved onto a schedule that is still broken for it.
        assert SCHEDULE_SENSITIVE_FALLBACK in {"simple", "sgm_uniform"}

    def test_insensitive_sampler_keeps_the_divergent_schedule(self):
        # Only the sensitive samplers are constrained; nothing else changes behaviour.
        for sampler in ("k_euler", "k_dpmpp_2m", "dpmpp_2m_sde"):
            resolved, substituted = resolve_schedule(sampler, "normal")
            assert resolved == "normal", sampler
            assert substituted is False, sampler

    def test_missing_sampler_or_schedule_is_not_substituted(self):
        assert resolve_schedule(None, "normal") == ("normal", False)
        assert resolve_schedule("dpmpp_3m_sde", None) == (None, False)


class TestNormalizeSchedulerSelection:
    """``normalize_horde_payload`` derives the schedule from karras unless one was named."""

    @staticmethod
    def _normalize(payload: dict) -> dict:
        from hordelib.pipeline.horde_compat import normalize_horde_payload

        class _Context:
            horde_model_name = "stable_diffusion"
            is_inpainting_model = False

        normalized, _faults = normalize_horde_payload(payload, _Context())
        return normalized

    def test_karras_true_selects_karras(self):
        assert self._normalize({"karras": True})["scheduler"] == "karras"

    def test_karras_false_selects_normal(self):
        assert self._normalize({"karras": False})["scheduler"] == "normal"

    def test_absent_karras_selects_normal(self):
        # The legacy default for a direct caller that sets neither.
        assert self._normalize({})["scheduler"] == "normal"

    @pytest.mark.parametrize("schedule", ["simple", "sgm_uniform", "exponential", "ddim_uniform"])
    def test_explicit_schedule_survives_the_karras_flag(self, schedule: str):
        # The flag cannot express these, so it must not overwrite a caller that named one.
        normalized = self._normalize({"karras": True, "scheduler": schedule})
        assert normalized["scheduler"] == schedule

    def test_unknown_schedule_falls_back_to_the_flag(self):
        normalized = self._normalize({"karras": True, "scheduler": "not_a_schedule"})
        assert normalized["scheduler"] == "karras"


class TestEnforceScheduleConstraints:
    @staticmethod
    def _enforce(payload: ImageGenPayload):
        from hordelib.pipeline.horde_compat import enforce_schedule_constraints

        return enforce_schedule_constraints(payload)

    def test_divergent_combination_is_corrected_and_disclosed(self):
        payload = ImageGenPayload.from_horde_dict({"sampler_name": "dpmpp_3m_sde", "scheduler": "normal"})
        corrected, faults = self._enforce(payload)

        assert corrected.scheduler == SCHEDULE_SENSITIVE_FALLBACK
        assert len(faults) == 1, "the substitution must be disclosed on the result"
        assert "dpmpp_3m_sde" in (faults[0].ref or "")
        assert corrected is not payload, "the input payload should not be mutated in place"
        assert payload.scheduler == "normal"

    def test_safe_combination_is_untouched_and_silent(self):
        payload = ImageGenPayload.from_horde_dict({"sampler_name": "k_dpmpp_2m", "scheduler": "normal"})
        corrected, faults = self._enforce(payload)

        assert corrected.scheduler == "normal"
        assert faults == []
        assert corrected is payload

    def test_sensitive_sampler_on_karras_is_untouched_and_silent(self):
        payload = ImageGenPayload.from_horde_dict({"sampler_name": "dpmpp_3m_sde", "scheduler": "karras"})
        corrected, faults = self._enforce(payload)

        assert corrected.scheduler == "karras"
        assert faults == []

    def test_a_karras_false_horde_request_ends_up_convergent(self):
        # The end-to-end shape of the fix: the flag resolves to `normal`, which this then corrects.
        normalized = TestNormalizeSchedulerSelection._normalize(
            {"sampler_name": "dpmpp_3m_sde", "karras": False},
        )
        corrected, faults = self._enforce(ImageGenPayload.from_horde_dict(normalized))

        assert corrected.scheduler == SCHEDULE_SENSITIVE_FALLBACK
        assert len(faults) == 1
