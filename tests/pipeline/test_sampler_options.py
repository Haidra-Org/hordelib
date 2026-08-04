"""Per-run solver options: payload mapping, clamping, and per-sampler filtering. No GPU required.

The filtering is the part that must not regress: handing a sampler an option its function does not accept
raises TypeError from inside graph execution, which surfaces as a failed job rather than a rejected
argument. The payload's clamping is the other half, because order 1 raises inside the multistep solvers.
"""

import pytest

from hordelib.execution.sampler_options import (
    clear_run_options,
    current_run_options,
    option_bounds,
    options_for_sampler,
    set_run_options,
)
from hordelib.pipeline.constants import (
    SAMPLERS_MAP,
    SOLVER_OPTION_FALLBACK_BOUNDS,
    SOLVER_TYPES,
    SolverOption,
)
from hordelib.pipeline.payload import ImageGenPayload


@pytest.fixture(autouse=True)
def _no_leftover_options():
    clear_run_options()
    yield
    clear_run_options()


class TestPayloadMapping:
    def test_no_options_by_default(self):
        # The back-compat guarantee: a payload that predates these fields produces no options at all, so
        # the sampler is built exactly as it was.
        assert ImageGenPayload().solver_options() == {}

    def test_eta_and_s_noise_map_to_upstream_names(self):
        payload = ImageGenPayload.from_horde_dict({"sampler_eta": 0.0, "sampler_s_noise": 1.25})
        assert payload.solver_options() == {"eta": 0.0, "s_noise": 1.25}

    def test_eta_zero_survives_rather_than_being_treated_as_unset(self):
        # eta=0 is the meaningful "make this deterministic" value, so it must not be confused with absent.
        assert ImageGenPayload.from_horde_dict({"sampler_eta": 0.0}).solver_options() == {"eta": 0.0}

    def test_order_is_offered_under_both_upstream_spellings(self):
        options = ImageGenPayload.from_horde_dict({"sampler_order": 3}).solver_options()
        assert options == {"max_order": 3, "order": 3}

    def test_out_of_range_values_clamp(self):
        payload = ImageGenPayload.from_horde_dict({"sampler_eta": 999, "sampler_s_churn": -5, "sampler_order": 1})
        options = payload.solver_options()
        assert options["eta"] == 100.0
        assert options["s_churn"] == 0.0
        assert options["max_order"] == 2, "order 1 raises inside deis/ipndm, so 2 is the floor"

    def test_churn_window_maps_to_upstream_names(self):
        payload = ImageGenPayload.from_horde_dict(
            {"sampler_s_churn": 0.3, "sampler_s_tmin": 0.5, "sampler_s_tmax": 12}
        )
        assert payload.solver_options() == {"s_churn": 0.3, "s_tmin": 0.5, "s_tmax": 12.0}

    def test_churn_window_is_absent_unless_asked_for(self):
        assert ImageGenPayload.from_horde_dict({"sampler_s_churn": 0.3}).solver_options() == {"s_churn": 0.3}

    def test_unknown_solver_type_is_dropped(self):
        assert "solver_type" not in ImageGenPayload.from_horde_dict({"sampler_solver_type": "bogus"}).solver_options()

    def test_known_solver_types_survive(self):
        for solver_type in sorted(SOLVER_TYPES):
            options = ImageGenPayload.from_horde_dict({"sampler_solver_type": solver_type}).solver_options()
            assert options["solver_type"] == solver_type


class TestRunScopedState:
    def test_set_and_clear(self):
        set_run_options({"eta": 0.5})
        assert current_run_options() == {"eta": 0.5}
        clear_run_options()
        assert current_run_options() == {}

    def test_setting_replaces_rather_than_merges(self):
        # One run's options must not leak into the next.
        set_run_options({"eta": 0.5, "s_noise": 2.0})
        set_run_options({"eta": 0.1})
        assert current_run_options() == {"eta": 0.1}

    def test_none_clears(self):
        set_run_options({"eta": 0.5})
        set_run_options(None)
        assert current_run_options() == {}


class TestOptionBounds:
    """The seam per-sampler ranges are read through, and the bounding it drives."""

    def test_a_vocabulary_option_has_no_range(self):
        assert option_bounds(str(SolverOption.SOLVER_TYPE), "dpmpp_2m_sde") is None

    def test_an_unknown_option_has_no_range(self):
        assert option_bounds("ge_gamma", "gradient_estimation") is None

    def test_an_option_the_table_does_not_name_falls_back(self):
        # `max_order` is the multistep order under the spelling the shared table does not carry.
        assert option_bounds("max_order", "deis") == SOLVER_OPTION_FALLBACK_BOUNDS[SolverOption.MAX_ORDER]

    @pytest.mark.parametrize(
        ("sampler_name", "expected_bounds"),
        [
            ("lms", (1.0, 100.0)),
            ("dpm_adaptive", (2.0, 3.0)),
        ],
    )
    def test_order_ranges_come_from_the_shared_table(self, sampler_name: str, expected_bounds):
        # The two samplers taking a literal `order` disagree with each other and with the fallback,
        # which is the whole reason the range is per-sampler.
        assert option_bounds("order", sampler_name) == expected_bounds

    def test_a_value_outside_the_range_is_held_rather_than_dropped(self, init_horde: None):
        # Bounding keeps the request serviceable: the sampler still receives the control it asked for.
        set_run_options({"s_churn": 500.0})
        assert options_for_sampler("euler") == {"s_churn": 100.0}

    def test_an_in_range_value_passes_through_untouched(self, init_horde: None):
        set_run_options({"s_churn": 0.25})
        assert options_for_sampler("euler") == {"s_churn": 0.25}

    def test_an_integer_option_stays_an_integer(self, init_horde: None):
        set_run_options({"max_order": 99})
        bounded = options_for_sampler("deis")["max_order"]
        assert bounded == int(SOLVER_OPTION_FALLBACK_BOUNDS[SolverOption.MAX_ORDER][1])
        assert isinstance(bounded, int)

    def test_lms_order_is_held_to_its_own_ceiling(self, init_horde: None):
        set_run_options({"order": 500})
        assert options_for_sampler("lms") == {"order": 100}

    def test_dpm_adaptive_order_is_held_to_its_own_narrow_range(self, init_horde: None):
        set_run_options({"order": 4})
        assert options_for_sampler("dpm_adaptive") == {"order": 3}


class TestSharedTableApplicability:
    """Applicability the shared constraints table decides, which the signature alone gets wrong."""

    def test_every_mapped_sampler_is_carried_by_the_table(self):
        # A sampler missing from the table would fall back to blanket ranges silently, which is the
        # drift this integration exists to prevent.
        from horde_sdk.backend_parsing.image.comfyui.hordelib import ComfyUIBackendValuesMapper
        from horde_sdk.generation_parameters.image.constraints import get_sampler_constraints

        mapper = ComfyUIBackendValuesMapper()
        missing = []
        for comfy_name in sorted(set(SAMPLERS_MAP.values())):
            if not mapper.is_valid_backend_sampler(comfy_name):
                missing.append(comfy_name)
                continue
            get_sampler_constraints(mapper.map_to_sdk_sampler(comfy_name))
        assert missing == [], f"samplers absent from the shared constraints table: {missing}"

    def test_dpm_fast_takes_no_options_despite_its_signature(self, init_horde: None):
        # `ksampler` wraps dpm_fast in a closure that forwards nothing, so an option the signature
        # advertises would raise TypeError inside the graph. The table knows; the signature does not.
        import inspect

        import comfy.k_diffusion.sampling as k_diffusion_sampling

        assert "eta" in inspect.signature(k_diffusion_sampling.sample_dpm_fast).parameters

        set_run_options({"eta": 0.5, "s_noise": 1.2})
        assert options_for_sampler("dpm_fast") == {}

    def test_er_sde_does_not_receive_an_eta_it_cannot_use(self, init_horde: None):
        set_run_options({"eta": 0.5, "s_noise": 1.1})
        assert options_for_sampler("er_sde") == {"s_noise": 1.1}

    def test_solver_type_reaches_only_the_family_whose_vocabulary_it_is(self, init_horde: None):
        set_run_options({"solver_type": "heun"})
        assert options_for_sampler("dpmpp_2m_sde") == {"solver_type": "heun"}
        assert options_for_sampler("dpmpp_2m_sde_heun") == {"solver_type": "heun"}
        # seeds_2 takes a solver_type too, but its vocabulary is phi-based, so midpoint/heun is not
        # a value it implements.
        assert options_for_sampler("seeds_2") == {}


class TestSamplerVocabularyLockstep:
    """hordelib's sampler map against the shared SDK vocabulary, so the two repos cannot drift."""

    LEGACY_ALIASES = {"dpmsolver", "plms"}
    """Horde names kept for backwards compatibility that resolve onto another sampler's backend name.

    They cannot appear in the SDK's convert map, which is one-to-one in both directions.
    """

    def _sdk_pairs(self) -> dict[str, str]:
        from horde_sdk.backend_parsing.image.comfyui.hordelib import ComfyUIBackendValuesMapper

        convert_map = ComfyUIBackendValuesMapper._COMFYUI_SAMPLERS_CONVERT_MAP
        return {str(horde_name).lower(): str(comfy_name) for comfy_name, horde_name in convert_map.items()}

    def test_hordelib_offers_every_sampler_the_sdk_names(self):
        missing = sorted(set(self._sdk_pairs()) - {name.lower() for name in SAMPLERS_MAP})
        assert missing == [], f"the SDK names samplers hordelib does not map: {missing}"

    def test_hordelib_offers_nothing_beyond_the_sdk_vocabulary_but_its_aliases(self):
        extra = {name.lower() for name in SAMPLERS_MAP} - set(self._sdk_pairs())
        assert extra == self.LEGACY_ALIASES, f"hordelib maps samplers the SDK does not name: {sorted(extra)}"

    def test_both_repos_agree_on_every_backend_target(self):
        hordelib_targets = {name.lower(): target for name, target in SAMPLERS_MAP.items()}
        mismatched = {
            horde_name: (comfy_name, hordelib_targets[horde_name])
            for horde_name, comfy_name in self._sdk_pairs().items()
            if hordelib_targets[horde_name] != comfy_name
        }
        assert mismatched == {}, f"the repos disagree on which backend sampler these names mean: {mismatched}"


class TestPerSamplerFiltering:
    def test_no_options_means_no_filtering_work(self):
        assert options_for_sampler("k_euler") == {}

    def test_eta_reaches_a_sampler_that_accepts_it(self, init_horde: None):
        set_run_options({"eta": 0.25})
        assert options_for_sampler("dpmpp_2m_sde") == {"eta": 0.25}

    def test_eta_is_dropped_for_a_sampler_without_it(self, init_horde: None):
        # sample_euler has no eta parameter; passing one would raise TypeError inside the graph.
        set_run_options({"eta": 0.25})
        assert options_for_sampler("euler") == {}

    def test_churn_reaches_the_deterministic_solvers(self, init_horde: None):
        set_run_options({"s_churn": 0.5})
        assert options_for_sampler("euler") == {"s_churn": 0.5}
        assert options_for_sampler("heun") == {"s_churn": 0.5}

    def test_the_churn_window_reaches_the_same_solvers_as_churn(self, init_horde: None):
        set_run_options({"s_tmin": 0.5, "s_tmax": 10.0})
        for churn_capable in ("euler", "heun", "dpm_2", "heunpp2"):
            assert options_for_sampler(churn_capable) == {"s_tmin": 0.5, "s_tmax": 10.0}
        assert options_for_sampler("dpmpp_2m_sde") == {}

    def test_solver_type_only_reaches_the_family_that_takes_it(self, init_horde: None):
        set_run_options({"solver_type": "heun"})
        assert options_for_sampler("dpmpp_2m_sde") == {"solver_type": "heun"}
        assert options_for_sampler("dpmpp_3m_sde") == {}

    def test_mixed_options_are_split_per_sampler(self, init_horde: None):
        set_run_options({"eta": 0.3, "s_churn": 0.4, "solver_type": "midpoint"})
        assert options_for_sampler("dpmpp_2m_sde") == {"eta": 0.3, "solver_type": "midpoint"}
        assert options_for_sampler("euler") == {"s_churn": 0.4}

    def test_unknown_sampler_name_yields_nothing(self, init_horde: None):
        set_run_options({"eta": 0.3})
        assert options_for_sampler("not_a_sampler") == {}

    def test_every_mapped_sampler_can_be_filtered_without_raising(self, init_horde: None):
        set_run_options({"eta": 0.5, "s_noise": 1.1, "s_churn": 0.2, "solver_type": "heun", "max_order": 3})
        for comfy_name in set(SAMPLERS_MAP.values()):
            options_for_sampler(comfy_name)
