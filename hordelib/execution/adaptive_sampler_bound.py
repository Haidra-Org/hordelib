"""An iteration bound for ComfyUI's error-controlled ``dpm_adaptive`` sampler.

``dpm_adaptive`` is the one ComfyUI sampler that chooses its own iteration count: its core
(``comfy.k_diffusion.sampling.DPMSolver.dpm_solver_adaptive``) is an unbounded ``while`` loop
driven by a PID step-size controller, and the nominal step count only sets the sigma range it
integrates over. On some model and payload combinations the controller never satisfies its
tolerance and the loop runs indefinitely while the sample is already essentially converged.

Iterations past the nominal schedule are tolerance polish with approximately zero marginal
quality, so an unbounded (or loosely bounded) solver pays multiples of the schedule's GPU cost
for no practical benefit. This module bounds the loop and delivers the best-effort sample
instead, recording the truncation so the caller can disclose it on the result.

The bound is applied by wrapping the sampler at the point where the nominal step count is
structurally visible. ComfyUI's ``comfy.samplers.ksampler`` builds ``dpm_adaptive``'s sampler
function as a closure that receives the full ``sigmas`` tensor and forwards only
``sigma_min``/``sigma_max`` to ``sample_dpm_adaptive``; patching the factory lets the
replacement read ``len(sigmas) - 1`` from its own arguments, so nothing has to be threaded
through globals or context.

Below the bound the sampler runs ComfyUI's own ``dpm_solver_adaptive`` unmodified, so the order
of operations, the PID controller and the noise handling are stock. This module imports neither
ComfyUI nor torch at module scope: it stays importable before ``hordelib.initialise()`` and it is
re-exported through ``hordelib.api``, whose import must not drag torch into a consumer.
"""

from __future__ import annotations

import threading
import typing
from collections.abc import Callable

from horde_sdk.generation_parameters.image.sampler_work import (
    BOUNDED_DPM_ADAPTIVE_V1,
    SamplerExecutionContractVersion,
    TrajectoryStepCount,
    maximum_adaptive_solver_iterations,
)
from loguru import logger
from pydantic import BaseModel

if typing.TYPE_CHECKING:
    import torch

ADAPTIVE_SAMPLER_NAME: typing.Final[str] = "dpm_adaptive"
"""The ComfyUI sampler name whose iteration count is solver-chosen rather than schedule-chosen."""

ADAPTIVE_ITERATION_BUDGET_MULTIPLIER: typing.Final[float] = (
    BOUNDED_DPM_ADAPTIVE_V1.iteration_multiplier_numerator
    / BOUNDED_DPM_ADAPTIVE_V1.iteration_multiplier_denominator
)
"""How many times the nominal step count the adaptive solver may iterate before it is stopped.

Past the nominal schedule the solver is polishing against its error tolerance, which buys
approximately no perceptible quality; every extra iteration still costs a full model evaluation.
A generous multiplier therefore spends multiples of the schedule's GPU time for nothing, and in
the pathological case (a tolerance the controller never reaches) spends it without terminating.
1.25 leaves a margin for a solver that is genuinely close to converging while capping the worst
case near the cost the schedule advertised. Adaptive runs that today settle around twice the
nominal iteration count are truncated by this bound; that is the intent, as those iterations were
buying polish rather than quality.
"""

SAMPLER_EXECUTION_CONTRACT_VERSION: typing.Final[SamplerExecutionContractVersion] = (
    SamplerExecutionContractVersion.V1
)
"""SDK sampler execution contract this backend guarantees on every render path."""

SAMPLER_TRUNCATION_METADATA_KEY: typing.Final[str] = "sampler_truncation"
"""The ``OutputArtifact.metadata`` key carrying a :class:`SamplerTruncation` for a run."""


class SamplerTruncation(BaseModel):
    """A machine-readable record that a solver-chosen sampler was stopped at its bound."""

    sampler: str
    """The ComfyUI sampler name that was bounded."""
    nominal_steps: int
    """The step count the requested schedule advertised (``len(sigmas) - 1``)."""
    iterations: int
    """The number of solver iterations actually run before the bound stopped the loop."""
    budget_multiplier: float = ADAPTIVE_ITERATION_BUDGET_MULTIPLIER
    """The multiple of the nominal schedule the bound allowed.

    Carried on the record so a consumer can describe the bound accurately without importing (and
    staying in lockstep with) hordelib's constant.
    """
    capped: bool = True
    """Always true on a recorded truncation; present so consumers can branch on the flag."""
    execution_contract_version: SamplerExecutionContractVersion = SAMPLER_EXECUTION_CONTRACT_VERSION
    """The SDK execution contract whose adaptive bound produced this truncation."""


def iteration_bound_for(nominal_steps: int) -> int:
    """Return the maximum solver iterations allowed for a schedule of *nominal_steps* steps."""
    if nominal_steps < 1:
        return 1
    return maximum_adaptive_solver_iterations(
        trajectory_steps=TrajectoryStepCount(nominal_steps),
        execution_policy=BOUNDED_DPM_ADAPTIVE_V1,
    )


class _IterationCapReached(Exception):
    """Raised from the solver's info callback to unwind the loop at the iteration bound.

    The solver reports its working ``x`` on every iteration, so unwinding here returns exactly
    the tensor the stock loop would have returned had it exited at this iteration. Raising is
    what makes the below-bound path stock: nothing about ComfyUI's loop is reimplemented.
    """

    def __init__(self, x: torch.Tensor, iterations: int) -> None:
        super().__init__(f"Adaptive solver reached its iteration bound after {iterations} iterations")
        self.x = x
        self.iterations = iterations


_lock = threading.Lock()
_run_truncations: list[SamplerTruncation] = []


def begin_run_recording() -> None:
    """Discard any recorded truncations so the next pipeline run starts from empty."""
    with _lock:
        _run_truncations.clear()


def take_run_truncations() -> list[SamplerTruncation]:
    """Return and clear the truncations recorded since :func:`begin_run_recording`."""
    with _lock:
        taken = list(_run_truncations)
        _run_truncations.clear()
    return taken


def _record_truncation(truncation: SamplerTruncation) -> None:
    with _lock:
        _run_truncations.append(truncation)


def _sample_dpm_adaptive_bounded(
    model: typing.Any,
    x: torch.Tensor,
    sigma_min: torch.Tensor,
    sigma_max: torch.Tensor,
    *,
    nominal_steps: int,
    extra_args: dict[str, typing.Any] | None = None,
    callback: Callable[[dict[str, typing.Any]], None] | None = None,
    disable: bool | None = None,
    **extra_options: typing.Any,
) -> torch.Tensor:
    """Run ComfyUI's adaptive DPM solver, stopping at the iteration bound for *nominal_steps*.

    Mirrors ``comfy.k_diffusion.sampling.sample_dpm_adaptive`` (same solver construction, same
    progress-bar wiring, same callback shape) and relies on ``dpm_solver_adaptive``'s own
    parameter defaults, which are identical to that wrapper's. The only addition is the
    bound-checking info callback.

    Args:
        model: The wrapped denoiser ComfyUI hands the sampler.
        x: The starting noise tensor.
        sigma_min: The final sigma of the schedule (the last non-zero entry).
        sigma_max: The first sigma of the schedule.
        nominal_steps: The step count the schedule advertised, which sets the bound.
        extra_args: The sampler's extra arguments (conditioning, seed, denoise mask).
        callback: ComfyUI's per-step callback, invoked exactly as the stock wrapper invokes it.
        disable: Whether to suppress the progress bar.
        **extra_options: Solver options forwarded to ``dpm_solver_adaptive``.

    Returns:
        torch.Tensor: The sampled latent, either fully converged or the best effort at the bound.
    """
    import torch
    from comfy.k_diffusion.sampling import DPMSolver
    from tqdm.auto import tqdm

    if sigma_min <= 0 or sigma_max <= 0:
        raise ValueError("sigma_min and sigma_max must not be 0")

    # ``return_info`` would make the solver return a tuple that ComfyUI's KSAMPLER cannot consume,
    # so it is dropped here rather than forwarded into a downstream shape error.
    extra_options.pop("return_info", None)

    max_iterations = iteration_bound_for(nominal_steps)
    truncated_at: int | None = None

    with torch.no_grad(), tqdm(disable=disable) as pbar:
        dpm_solver = DPMSolver(model, extra_args, eps_callback=pbar.update)

        def _info_callback(info: dict[str, typing.Any]) -> None:
            if callback is not None:
                callback(
                    {
                        "sigma": dpm_solver.sigma(info["t"]),
                        "sigma_hat": dpm_solver.sigma(info["t_up"]),
                        **info,
                    },
                )
            if info["steps"] >= max_iterations:
                raise _IterationCapReached(info["x"], info["steps"])

        dpm_solver.info_callback = _info_callback

        try:
            x, _info = dpm_solver.dpm_solver_adaptive(
                x,
                dpm_solver.t(torch.tensor(sigma_max)),
                dpm_solver.t(torch.tensor(sigma_min)),
                **extra_options,
            )
        except _IterationCapReached as reached:
            x = reached.x
            truncated_at = reached.iterations

    if truncated_at is not None:
        logger.warning(
            "Adaptive sampler truncated at its iteration bound; delivering the best-effort sample: "
            "sampler={}, nominal_steps={}, iterations={}",
            ADAPTIVE_SAMPLER_NAME,
            nominal_steps,
            truncated_at,
        )
        _record_truncation(
            SamplerTruncation(
                sampler=ADAPTIVE_SAMPLER_NAME,
                nominal_steps=nominal_steps,
                iterations=truncated_at,
            ),
        )

    return x


def bounded_dpm_adaptive_sampler_function(
    model: typing.Any,
    noise: torch.Tensor,
    sigmas: torch.Tensor,
    extra_args: dict[str, typing.Any] | None = None,
    callback: Callable[[dict[str, typing.Any]], None] | None = None,
    disable: bool | None = None,
    **extra_options: typing.Any,
) -> torch.Tensor:
    """The bounded replacement for the ``dpm_adaptive`` sampler function ``ksampler`` builds.

    Signature and sigma handling match ComfyUI's own ``dpm_adaptive_function`` closure; the
    schedule length it sees is what supplies the iteration bound.
    """
    if len(sigmas) <= 1:
        return noise

    sigma_min = sigmas[-1]
    if sigma_min == 0:
        sigma_min = sigmas[-2]

    return _sample_dpm_adaptive_bounded(
        model,
        noise,
        sigma_min,
        sigmas[0],
        nominal_steps=len(sigmas) - 1,
        extra_args=extra_args,
        callback=callback,
        disable=disable,
        **extra_options,
    )


def bound_adaptive_sampler(sampler: typing.Any, sampler_name: str) -> typing.Any:
    """Swap in the bounded sampler function when *sampler_name* is the adaptive sampler.

    Args:
        sampler: The ``comfy.samplers.KSAMPLER`` the stock factory produced.
        sampler_name: The sampler name the factory was asked for.

    Returns:
        typing.Any: The same sampler object, with its function bounded when it is adaptive.
    """
    if sampler_name == ADAPTIVE_SAMPLER_NAME:
        sampler.sampler_function = bounded_dpm_adaptive_sampler_function
    return sampler
