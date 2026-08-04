"""The sigma schedules ComfyUI carries as nodes: run state, computation, and routing. No GPU required.

Three behaviours matter here. The schedule must equal what the upstream node would have produced, or
the request is served a different schedule than the one it names. It must reach the sampler without
appearing in the graph, because no graph this package runs has an input that can carry it. And with no
schedule set, nothing may change: the override has to be invisible to every payload that predates it.
"""

import pytest
from horde_model_reference.meta_consts import KNOWN_IMAGE_GENERATION_BASELINE

from hordelib.execution.sigma_schedules import (
    GITS_COEFFICIENT,
    SigmaScheduleRequest,
    clear_run_schedule,
    current_run_schedule,
    set_run_schedule,
    sigmas_for_run,
)
from hordelib.pipeline.constants import (
    SCHEDULERS,
    SIGMA_GENERATOR_GRAPH_SCHEDULE,
    SIGMA_GENERATOR_SCHEDULES,
    SigmaGeneratorSchedule,
)
from hordelib.pipeline.context import ModelContext
from hordelib.pipeline.families.image_gen.bindings import comfy_scheduler
from hordelib.pipeline.horde_compat import UnsupportedScheduleForBaselineError, resolve_sigma_schedule
from hordelib.pipeline.payload import ImageGenPayload


@pytest.fixture(autouse=True)
def _no_leftover_schedule():
    clear_run_schedule()
    yield
    clear_run_schedule()


def _context(baseline: KNOWN_IMAGE_GENERATION_BASELINE | None) -> ModelContext:
    return ModelContext(horde_model_name="test_model", baseline=baseline)


class TestVocabulary:
    def test_both_generators_are_offered_to_callers(self):
        for schedule in SIGMA_GENERATOR_SCHEDULES:
            assert schedule in SCHEDULERS

    def test_the_graph_placeholder_is_a_schedule_comfyui_names(self):
        # An input outside the node's declared list fails prompt validation for the whole graph.
        assert SIGMA_GENERATOR_GRAPH_SCHEDULE in SCHEDULERS
        assert SIGMA_GENERATOR_GRAPH_SCHEDULE not in SIGMA_GENERATOR_SCHEDULES

    def test_a_generator_schedule_survives_the_payload_validator(self):
        payload = ImageGenPayload.from_horde_dict({"scheduler": "align_your_steps"})
        assert payload.scheduler == "align_your_steps"


class TestGraphPlaceholder:
    def test_a_generator_schedule_never_reaches_the_graph(self):
        for schedule in sorted(SIGMA_GENERATOR_SCHEDULES):
            payload = ImageGenPayload.from_horde_dict({"scheduler": schedule})
            assert comfy_scheduler(payload) == SIGMA_GENERATOR_GRAPH_SCHEDULE

    def test_every_other_schedule_is_passed_through_unchanged(self):
        for schedule in SCHEDULERS:
            if schedule in SIGMA_GENERATOR_SCHEDULES:
                continue
            payload = ImageGenPayload.from_horde_dict({"scheduler": schedule})
            assert comfy_scheduler(payload) == schedule


class TestRequestResolution:
    def test_a_named_schedule_needs_no_request(self):
        payload = ImageGenPayload.from_horde_dict({"scheduler": "karras"})
        assert resolve_sigma_schedule(payload, _context(KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_1)) is None

    def test_gits_needs_no_family(self):
        payload = ImageGenPayload.from_horde_dict({"scheduler": "gits"})
        request = resolve_sigma_schedule(payload, _context(KNOWN_IMAGE_GENERATION_BASELINE.flux_1))
        assert request == SigmaScheduleRequest(schedule=SigmaGeneratorSchedule.GITS)

    @pytest.mark.parametrize(
        ("baseline", "expected_model_type"),
        [
            (KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_1, "SD1"),
            (KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_xl, "SDXL"),
        ],
    )
    def test_align_your_steps_carries_the_measured_family(self, baseline, expected_model_type):
        payload = ImageGenPayload.from_horde_dict({"scheduler": "align_your_steps"})
        request = resolve_sigma_schedule(payload, _context(baseline))
        assert request is not None
        assert request.align_your_steps_model_type == expected_model_type

    @pytest.mark.parametrize(
        "baseline",
        [
            KNOWN_IMAGE_GENERATION_BASELINE.flux_1,
            KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_2_768,
            None,
        ],
    )
    def test_align_your_steps_refuses_a_family_it_has_no_levels_for(self, baseline):
        # Substituting another family's levels would run a schedule measured for a different model.
        payload = ImageGenPayload.from_horde_dict({"scheduler": "align_your_steps"})
        with pytest.raises(UnsupportedScheduleForBaselineError):
            resolve_sigma_schedule(payload, _context(baseline))


class TestRunScopedState:
    def test_set_and_clear(self):
        request = SigmaScheduleRequest(schedule=SigmaGeneratorSchedule.GITS)
        set_run_schedule(request)
        assert current_run_schedule() == request
        clear_run_schedule()
        assert current_run_schedule() is None

    def test_setting_replaces_rather_than_accumulating(self):
        set_run_schedule(SigmaScheduleRequest(schedule=SigmaGeneratorSchedule.GITS))
        set_run_schedule(
            SigmaScheduleRequest(schedule=SigmaGeneratorSchedule.ALIGN_YOUR_STEPS, align_your_steps_model_type="SD1"),
        )
        current = current_run_schedule()
        assert current is not None
        assert current.schedule is SigmaGeneratorSchedule.ALIGN_YOUR_STEPS

    def test_none_clears(self):
        set_run_schedule(SigmaScheduleRequest(schedule=SigmaGeneratorSchedule.GITS))
        set_run_schedule(None)
        assert current_run_schedule() is None


class TestComputedSigmas:
    """The computed schedule against the upstream node that defines it."""

    def test_nothing_is_supplied_when_no_schedule_is_set(self, init_horde: None):
        # The back-compat guarantee: every existing payload keeps resolving its schedule through comfy.
        assert sigmas_for_run(30) is None

    @pytest.mark.parametrize("steps", [1, 8, 10, 30])
    @pytest.mark.parametrize("model_type", ["SD1", "SDXL"])
    def test_align_your_steps_matches_its_node(self, init_horde: None, model_type: str, steps: int):
        from comfy_extras.nodes_align_your_steps import AlignYourStepsScheduler

        set_run_schedule(
            SigmaScheduleRequest(
                schedule=SigmaGeneratorSchedule.ALIGN_YOUR_STEPS,
                align_your_steps_model_type=model_type,
            ),
        )
        ours = sigmas_for_run(steps)
        theirs = AlignYourStepsScheduler().execute(model_type, steps, 1.0).result[0]

        assert ours is not None
        assert len(ours) == steps + 1
        assert ours.tolist() == pytest.approx(theirs.tolist())

    @pytest.mark.parametrize("steps", [2, 10, 20, 30])
    def test_gits_matches_its_node(self, init_horde: None, steps: int):
        from comfy_extras.nodes_gits import GITSScheduler

        set_run_schedule(SigmaScheduleRequest(schedule=SigmaGeneratorSchedule.GITS))
        ours = sigmas_for_run(steps)
        theirs = GITSScheduler().execute(GITS_COEFFICIENT, steps, 1.0).result[0]

        assert ours is not None
        assert len(ours) == steps + 1
        assert ours.tolist() == pytest.approx(theirs.tolist())

    def test_gits_spans_the_steps_asked_for_below_its_node_minimum(self, init_horde: None):
        # The node's own step input starts at 2; a shorter run must still get a schedule of its length.
        set_run_schedule(SigmaScheduleRequest(schedule=SigmaGeneratorSchedule.GITS))
        sigmas = sigmas_for_run(1)
        assert sigmas is not None
        assert len(sigmas) == 2

    @pytest.mark.parametrize("schedule", list(SigmaGeneratorSchedule))
    def test_every_schedule_ends_at_zero(self, init_horde: None, schedule: SigmaGeneratorSchedule):
        set_run_schedule(SigmaScheduleRequest(schedule=schedule, align_your_steps_model_type="SD1"))
        sigmas = sigmas_for_run(12)
        assert sigmas is not None
        assert sigmas[-1] == 0.0
        assert sigmas[0] > sigmas[-2] > 0.0


class TestPatchSeam:
    def test_calculate_sigmas_delegates_when_no_schedule_is_set(self, init_horde: None):
        import comfy.samplers

        from hordelib.execution.comfy_patches import _originals

        model_sampling = _StubModelSampling()
        patched = comfy.samplers.calculate_sigmas(model_sampling, "karras", 12)
        original = _originals["calculate_sigmas"](model_sampling, "karras", 12)
        assert patched.tolist() == original.tolist()

    def test_calculate_sigmas_supplies_the_run_schedule(self, init_horde: None):
        import comfy.samplers

        set_run_schedule(SigmaScheduleRequest(schedule=SigmaGeneratorSchedule.GITS))
        # The schedule name the graph carries is the placeholder, and is overridden whatever it says.
        sigmas = comfy.samplers.calculate_sigmas(_StubModelSampling(), SIGMA_GENERATOR_GRAPH_SCHEDULE, 12)
        assert sigmas.tolist() == pytest.approx(sigmas_for_run(12).tolist())


class _StubModelSampling:
    """The only two attributes ``calculate_sigmas`` reads for the range-based schedules."""

    sigma_min = 0.0291675
    sigma_max = 14.6146412
