"""Unit tests for the bounded ``dpm_adaptive`` sampler (CPU only, no model weights).

The solver is driven with a stub denoiser so the PID controller runs for real without any
accelerator or checkpoint. ``set_system_path`` is what makes the vendored ComfyUI importable
outside a full ``hordelib.initialise()``.
"""

import pytest
import torch
from horde_sdk.generation_parameters.image.consts import KNOWN_IMAGE_SAMPLERS
from horde_sdk.generation_parameters.image.sampler_work import (
    SamplerExecutionContractVersion,
    SamplerWorkUnitCount,
    TrajectoryStepCount,
    maximum_sampler_work,
)

from hordelib.config_path import set_system_path
from hordelib.execution.adaptive_sampler_bound import (
    ADAPTIVE_ITERATION_BUDGET_MULTIPLIER,
    SamplerTruncation,
    begin_run_recording,
    bounded_dpm_adaptive_sampler_function,
    iteration_bound_for,
    take_run_truncations,
)

set_system_path()

k_diffusion_sampling = pytest.importorskip("comfy.k_diffusion.sampling")


class _StubDenoiser:
    """A denoiser whose output never satisfies the solver's tolerance.

    The solver's error estimate compares a low- and a high-order step; a denoiser that returns a
    strongly sigma-dependent field keeps those apart, so the controller keeps proposing steps and
    the loop is the unbounded case the bound exists for.
    """

    def __init__(self, hostile: bool) -> None:
        self.hostile = hostile
        self.calls = 0

    def __call__(self, x: torch.Tensor, sigma: torch.Tensor, **kwargs: object) -> torch.Tensor:
        self.calls += 1
        if self.hostile:
            return torch.sin(x * 40.0) * sigma.reshape(-1, 1, 1, 1) * 10.0
        return x * 0.5


def _sigmas(steps: int) -> torch.Tensor:
    """A descending schedule ending at zero, the shape ComfyUI hands a sampler function."""
    return torch.cat([torch.linspace(8.0, 0.2, steps), torch.zeros(1)])


@pytest.fixture(autouse=True)
def _clear_recording() -> None:
    begin_run_recording()


def test_iteration_bound_is_the_multiplier_rounded_up() -> None:
    assert ADAPTIVE_ITERATION_BUDGET_MULTIPLIER == 1.25
    assert iteration_bound_for(20) == 25
    assert iteration_bound_for(30) == 38
    assert iteration_bound_for(1) == 2
    assert iteration_bound_for(0) == 1


@pytest.mark.parametrize("solver_order", [2, 3])
def test_hostile_solver_terminates_at_the_sdk_bound_with_a_usable_tensor(solver_order: int) -> None:
    steps = 8
    noise = torch.randn(1, 4, 8, 8, generator=torch.Generator().manual_seed(1234))
    model = _StubDenoiser(hostile=True)

    result = bounded_dpm_adaptive_sampler_function(
        model,
        noise,
        _sigmas(steps),
        extra_args={"seed": 1234},
        callback=None,
        disable=True,
        order=solver_order,
    )

    assert result.shape == noise.shape
    assert result.dtype == noise.dtype
    assert result.device == noise.device

    truncations = take_run_truncations()
    assert len(truncations) == 1
    assert truncations[0] == SamplerTruncation(
        sampler="dpm_adaptive",
        nominal_steps=steps,
        iterations=iteration_bound_for(steps),
        capped=True,
    )
    ceiling = maximum_sampler_work(
        sampler=KNOWN_IMAGE_SAMPLERS.k_dpm_adaptive,
        trajectory_steps=TrajectoryStepCount(steps),
        execution_contract_version=SamplerExecutionContractVersion.V1,
        adaptive_work_units_per_iteration=solver_order,
    )
    assert ceiling is not None
    assert ceiling.work_units == SamplerWorkUnitCount(model.calls)


def test_below_the_bound_matches_the_unpatched_sampler() -> None:
    """A solver that converges inside the bound must produce exactly the stock result."""
    steps = 8
    sigmas = _sigmas(steps)
    noise = torch.randn(1, 4, 8, 8, generator=torch.Generator().manual_seed(99))

    bounded = bounded_dpm_adaptive_sampler_function(
        _StubDenoiser(hostile=False),
        noise.clone(),
        sigmas,
        extra_args={"seed": 99},
        callback=None,
        disable=True,
    )

    sigma_min = sigmas[-2]
    stock = k_diffusion_sampling.sample_dpm_adaptive(
        _StubDenoiser(hostile=False),
        noise.clone(),
        sigma_min,
        sigmas[0],
        extra_args={"seed": 99},
        callback=None,
        disable=True,
    )

    assert torch.equal(bounded, stock)
    assert take_run_truncations() == []


def test_no_truncation_is_recorded_when_the_solver_converges() -> None:
    result = bounded_dpm_adaptive_sampler_function(
        _StubDenoiser(hostile=False),
        torch.randn(1, 4, 8, 8, generator=torch.Generator().manual_seed(7)),
        _sigmas(8),
        extra_args={"seed": 7},
        callback=None,
        disable=True,
    )

    assert result is not None
    assert take_run_truncations() == []


def test_degenerate_schedule_returns_the_noise_unchanged() -> None:
    noise = torch.randn(1, 4, 8, 8)

    result = bounded_dpm_adaptive_sampler_function(
        _StubDenoiser(hostile=False),
        noise,
        torch.zeros(1),
        extra_args={},
        callback=None,
        disable=True,
    )

    assert result is noise
    assert take_run_truncations() == []


def test_user_callback_still_receives_every_step_below_the_bound() -> None:
    seen: list[dict] = []

    bounded_dpm_adaptive_sampler_function(
        _StubDenoiser(hostile=False),
        torch.randn(1, 4, 8, 8, generator=torch.Generator().manual_seed(3)),
        _sigmas(8),
        extra_args={"seed": 3},
        callback=seen.append,
        disable=True,
    )

    assert seen
    assert {"sigma", "sigma_hat", "i", "denoised", "x", "steps"} <= set(seen[0])
