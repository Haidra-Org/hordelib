"""Per-run sigma schedules for the two ComfyUI ships as nodes, which the node graph cannot express.

ComfyUI resolves a schedule name through ``comfy.samplers.calculate_sigmas``, whose table
(``SCHEDULER_HANDLERS``) holds the schedules computable from the model's sigma range alone. Align Your
Steps and GITS are not in it: both are published, measured schedules carried as literal tables, and
upstream exposes them as scheduler nodes emitting a SIGMAS output. Only the custom-sampler graph shape
consumes such an output, while the graphs this package runs name their schedule on a ``KSampler`` or
``BasicScheduler`` input, so requesting one through the graph is impossible without changing every
graph's shape.

The schedule is therefore carried beside the graph for the duration of one run and applied at
``calculate_sigmas``, which both graph shapes reach: ``KSampler.calculate_sigmas`` calls it as a module
global and ``BasicScheduler`` calls it as a module attribute. The graph's own schedule input keeps a
name ComfyUI recognises (see
:data:`hordelib.pipeline.constants.SIGMA_GENERATOR_GRAPH_SCHEDULE`) and is simply overridden.

The override holds for every schedule computed during the run, which includes a hires-fix second pass
whose node names its own schedule in the packaged graph. That is the intended reading of a request: the
schedule was asked for by the job, not by one node of it.

The state is a module-level value guarded by a lock, mirroring
:mod:`hordelib.execution.sampler_options`: one process renders one image at a time, and both are set
and cleared by the same executor bracket. Nothing here changes behaviour unless a caller sets a
schedule; with none set every schedule resolves exactly as it did before.

The sigma tables and the interpolation are imported from the pinned ComfyUI checkout rather than copied,
so a schedule computed here is the one its node would have produced. Denoise is the one deliberate
difference: the nodes take it as an input and truncate the schedule themselves, while this layer is
called from inside ComfyUI's own denoise handling, which already asks for the inflated step count and
truncates the result. A partial-denoise request therefore behaves as it does for every other schedule.
"""

from __future__ import annotations

import threading
import typing
from dataclasses import dataclass

from loguru import logger

from hordelib.pipeline.constants import SigmaGeneratorSchedule

if typing.TYPE_CHECKING:
    import torch

GITS_COEFFICIENT: typing.Final[float] = 1.20
"""The GITS schedule table to index, at upstream's node default.

The coefficient selects between measured schedule families rather than scaling one, so it is a
vocabulary of its own. Only the default family is offered until a request can name one.
"""


@dataclass(frozen=True)
class SigmaScheduleRequest:
    """A sigma schedule to run, and the facts needed to build it.

    Attributes:
        schedule: The generator to build the schedule with.
        align_your_steps_model_type: The noise-level family for Align Your Steps, as
            ``comfy_extras.nodes_align_your_steps.NOISE_LEVELS`` keys it. Required for that
            schedule and unused by GITS, whose tables are keyed by coefficient and hold for any
            model.
    """

    schedule: SigmaGeneratorSchedule
    align_your_steps_model_type: str | None = None


_lock = threading.Lock()
_run_schedule: list[SigmaScheduleRequest] = []


def set_run_schedule(request: SigmaScheduleRequest | None) -> None:
    """Replace the sigma schedule applied to samplers built during the next pipeline run."""
    with _lock:
        _run_schedule.clear()
        if request is not None:
            _run_schedule.append(request)


def clear_run_schedule() -> None:
    """Drop any schedule so a subsequent run resolves schedules the way ComfyUI does."""
    with _lock:
        _run_schedule.clear()


def current_run_schedule() -> SigmaScheduleRequest | None:
    """Return the schedule currently in force, or None when ComfyUI's own resolution applies."""
    with _lock:
        return _run_schedule[0] if _run_schedule else None


def _align_your_steps_sigmas(model_type: str, steps: int) -> list[float]:
    """Return the Align Your Steps noise levels for *model_type*, interpolated to *steps*.

    Mirrors ``AlignYourStepsScheduler.execute`` at denoise 1.0: the published levels are used as they
    are when they already have one entry per step boundary, and log-linearly interpolated otherwise.
    """
    from comfy_extras.nodes_align_your_steps import NOISE_LEVELS, loglinear_interp

    sigmas = NOISE_LEVELS[model_type][:]
    if (steps + 1) != len(sigmas):
        sigmas = list(loglinear_interp(sigmas, steps + 1))
    return [float(sigma) for sigma in sigmas]


def _gits_sigmas(coefficient: float, steps: int) -> list[float]:
    """Return the GITS schedule for *steps* at *coefficient*.

    Mirrors ``GITSScheduler.execute`` at denoise 1.0: the table holds a measured schedule per step
    count from 2 to 20, and the longest entry is log-linearly interpolated beyond that. The node's own
    step input starts at 2; a single-step run is served by interpolating the shortest entry down, so a
    schedule always spans exactly the steps asked for rather than silently adding one.
    """
    from comfy_extras.nodes_gits import NOISE_LEVELS, loglinear_interp

    table = NOISE_LEVELS[round(coefficient, 2)]
    if 2 <= steps <= 20:
        sigmas = table[steps - 2][:]
    else:
        sigmas = list(loglinear_interp(table[-1][:] if steps > 20 else table[0][:], steps + 1))
    return [float(sigma) for sigma in sigmas]


def sigmas_for_run(steps: int) -> torch.Tensor | None:
    """Return the sigma schedule in force for a *steps*-step run, or None when none is.

    Args:
        steps: The number of sampling steps the schedule must span, as ComfyUI computed it (already
            inflated for partial denoise, and for the samplers that discard a penultimate sigma).

    Returns:
        The schedule as a tensor of ``steps + 1`` sigmas ending at zero, or None when the caller
        should resolve the schedule itself.
    """
    request = current_run_schedule()
    if request is None:
        return None

    import torch

    if request.schedule is SigmaGeneratorSchedule.GITS:
        sigmas = _gits_sigmas(GITS_COEFFICIENT, steps)
    else:
        model_type = request.align_your_steps_model_type
        if model_type is None:
            # Unreachable while requests are built by resolve_sigma_schedule, which refuses a baseline
            # with no published levels rather than passing one through unset.
            logger.error("An align_your_steps run reached the sampler with no model type; using ComfyUI's schedule.")
            return None
        sigmas = _align_your_steps_sigmas(model_type, steps)

    sigmas[-1] = 0.0
    logger.debug(f"Sigma schedule {request.schedule} supplied {len(sigmas)} sigmas for {steps} steps.")
    return torch.FloatTensor(sigmas)
