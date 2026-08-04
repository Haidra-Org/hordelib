"""Per-run solver options for ComfyUI samplers, which the node graph cannot express.

ComfyUI's samplers take tuning arguments beyond the sampler name and the schedule. The ones that matter
are the stochastic controls: ``eta`` scales the reverse-time SDE's noise injection, ``s_noise`` scales the
noise added at each step, and ``s_churn`` adds Karras churn to the otherwise-deterministic solvers, within
the sigma window ``s_tmin`` to ``s_tmax``. They reach the solver through
``comfy.samplers.ksampler(name, extra_options)``, which is the same call upstream's dedicated per-sampler
nodes make: ``SamplerDPMPP_2M_SDE`` and its siblings in ``comfy_extras/nodes_custom_sampler.py`` build a
``KSAMPLER`` object with the options baked in, and ``SamplerCustom`` merely runs a ``SAMPLER`` it is
handed. The stock ``KSampler`` node this package builds graphs around has no inputs for these arguments,
so they sit permanently at their defaults.

Their reach is real rather than cosmetic: at ``eta=0`` an SDE sampler collapses onto its deterministic
counterpart (measured at 0.996 similarity between ``dpmpp_2m_sde`` at eta 0 and ``k_dpmpp_2m``), and at
``eta=2.5`` it departs from it entirely. That makes ``eta`` a continuous variation dial, where changing
the seed is all-or-nothing.

Options are set for the duration of one pipeline run and filtered per sampler, because passing an option a
sampler does not accept raises ``TypeError`` inside the graph. Applicability and per-sampler ranges come
from the constraints table in ``horde_sdk.generation_parameters.image.constraints``, which is the same
data the API validates requests against, so this layer refuses exactly what could never have been
requested rather than holding a second opinion. The sampler function's own signature is checked after it,
as the catch for anything the table has not heard of. The state is a module-level dict guarded by a lock,
mirroring how ``hordelib.execution.adaptive_sampler_bound`` scopes its truncation recording to a run: one
process renders one image at a time, and both are collected by the same executor bracket.

Nothing here changes behaviour unless a caller sets options. With none set the sampler is built exactly as
it was before, so every existing payload renders identically.
"""

from __future__ import annotations

import inspect
import threading
import typing

from horde_sdk.backend_parsing.image.comfyui.hordelib import ComfyUIBackendValuesMapper
from horde_sdk.generation_parameters.image.constraints import (
    SAMPLER_SOLVER_KNOB,
    get_sampler_constraints,
)
from loguru import logger

from hordelib.pipeline.constants import SOLVER_OPTION_FALLBACK_BOUNDS, SolverOption

SamplerOptionValue = float | int | str

_BOUNDED_OPTION_NAMES: typing.Final[frozenset[str]] = frozenset(
    str(option) for option in SOLVER_OPTION_FALLBACK_BOUNDS
)
"""The option names carrying a numeric range, as ComfyUI's sampler functions spell them."""

_SDK_KNOB_NAMES: typing.Final[frozenset[str]] = frozenset(str(knob) for knob in SAMPLER_SOLVER_KNOB)
"""The knobs the shared constraints table carries a per-sampler range for.

``max_order`` is absent: the table names the multistep order by its other upstream spelling, so that
option is bounded from the fallback and left to the signature filter.
"""

_sampler_name_mapper: typing.Final[ComfyUIBackendValuesMapper] = ComfyUIBackendValuesMapper()
"""Translates the backend sampler name back to the horde name the constraints table is keyed by."""

_lock = threading.Lock()
_run_options: dict[str, SamplerOptionValue] = {}


def set_run_options(options: dict[str, SamplerOptionValue] | None) -> None:
    """Replace the options applied to samplers built during the next pipeline run."""
    with _lock:
        _run_options.clear()
        if options:
            _run_options.update(options)


def clear_run_options() -> None:
    """Drop any options so a subsequent run builds samplers at their defaults."""
    with _lock:
        _run_options.clear()


def current_run_options() -> dict[str, SamplerOptionValue]:
    """Return a copy of the options currently in force."""
    with _lock:
        return dict(_run_options)


def _sampler_signature_parameters(sampler_name: str) -> frozenset[str]:
    """Return the keyword arguments the named ComfyUI sampler function accepts.

    ``dpm_fast`` and ``dpm_adaptive`` are built by ``ksampler`` as closures over the schedule rather than
    resolved by name, so their underlying implementations are inspected instead.
    """
    import comfy.k_diffusion.sampling as k_diffusion_sampling

    function = getattr(k_diffusion_sampling, f"sample_{sampler_name}", None)
    if function is None:
        return frozenset()
    return frozenset(inspect.signature(function).parameters)


def _shared_table_bounds(option_name: str, sampler_name: str) -> tuple[float, float] | None:
    """Return the range the shared constraints table gives this option on this sampler.

    None means the table has nothing to say: the option is one it does not name, the sampler is one it
    does not carry, or the sampler's solver function does not take the option at all.
    """
    if option_name not in _SDK_KNOB_NAMES:
        return None

    if not _sampler_name_mapper.is_valid_backend_sampler(sampler_name):
        return None

    constraints = get_sampler_constraints(_sampler_name_mapper.map_to_sdk_sampler(sampler_name))
    knob_range = constraints.numeric_knob_ranges.get(SAMPLER_SOLVER_KNOB(option_name))
    if knob_range is None:
        return None

    return (knob_range.minimum, knob_range.maximum)


def option_bounds(option_name: str, sampler_name: str) -> tuple[float, float] | None:
    """Return the inclusive range *option_name* may take on *sampler_name*, or None when it has none.

    This is the single point at which per-sampler ranges are consulted, so widening a range for one
    solver is a change in the shared constraints table rather than in every place a value is checked.
    That table is the authority, because the same ranges decide what the API accepts: a value this
    returns a narrower range for is one a request could not have carried in the first place. The
    ranges in :data:`hordelib.pipeline.constants.SOLVER_OPTION_FALLBACK_BOUNDS` cover only what the
    table does not name, notably the ``max_order`` spelling of the multistep order.

    Args:
        option_name: The option, spelled as ComfyUI's sampler functions name it.
        sampler_name: The ComfyUI sampler the option is destined for.

    Returns:
        The inclusive lower and upper bound, or None for an option with no numeric range (``solver_type``
        is a vocabulary) or one this layer knows nothing about.
    """
    table_bounds = _shared_table_bounds(option_name, sampler_name)
    if table_bounds is not None:
        return table_bounds

    if option_name not in _BOUNDED_OPTION_NAMES:
        return None
    return SOLVER_OPTION_FALLBACK_BOUNDS[SolverOption(option_name)]


def table_rejects_option(option_name: str, sampler_name: str, value: SamplerOptionValue) -> bool:
    """Return whether the shared constraints table says this option does nothing on this sampler.

    The table is not merely a second opinion on the signature filter. ``ksampler`` builds ``dpm_fast``
    as a closure that accepts no options at all and forwards none, while the ``sample_dpm_fast`` it
    wraps declares several: inspecting the signature therefore says yes where handing the option over
    would raise ``TypeError`` inside graph execution. The table records what the built sampler takes,
    so it is consulted first and the signature stays as the check that catches anything it has not
    heard of.

    ``solver_type`` is judged on its value as well as its name, because several samplers take a
    keyword of that name over vocabularies that share no member: `dpmpp_2m_sde` corrects by
    `midpoint` or `heun`, while `seeds_2` and the `exp_heun_2_x0` pair take `phi_1` or `phi_2`. A
    value from the wrong vocabulary names nothing the solver implements, so it is dropped rather than
    handed over to be compared against branches it can never match.
    """
    if option_name not in _SDK_KNOB_NAMES:
        return False

    if not _sampler_name_mapper.is_valid_backend_sampler(sampler_name):
        return False

    constraints = get_sampler_constraints(_sampler_name_mapper.map_to_sdk_sampler(sampler_name))
    knob = SAMPLER_SOLVER_KNOB(option_name)
    if not constraints.accepts_knob(knob):
        return True

    if knob is SAMPLER_SOLVER_KNOB.solver_type:
        return value not in {str(choice) for choice in constraints.solver_type_choices}

    return False


def _bounded_value(
    option_name: str,
    value: SamplerOptionValue,
    sampler_name: str,
) -> SamplerOptionValue:
    """Return *value* held inside the range :func:`option_bounds` gives it for *sampler_name*."""
    if isinstance(value, str):
        return value

    bounds = option_bounds(option_name, sampler_name)
    if bounds is None:
        return value

    minimum, maximum = bounds
    bounded = min(max(value, minimum), maximum)
    if bounded == value:
        return value

    logger.debug(f"Sampler {sampler_name} bounds {option_name} to {bounds}; {value} was held to {bounded}.")
    return int(bounded) if isinstance(value, int) else bounded


def options_for_sampler(sampler_name: str) -> dict[str, SamplerOptionValue]:
    """Return the in-force options this sampler actually accepts, held inside its own ranges.

    Filtering is not defensive tidiness: ``sample_euler`` has no ``eta``, and handing it one raises
    ``TypeError`` from inside graph execution, which surfaces as a failed job rather than a bad argument.
    An option survives only if the shared constraints table and the sampler function's own signature
    both accept it, in that order (see :func:`table_rejects_option` for why the signature alone is not
    enough). A value the sampler does take but at a narrower range is bounded rather than dropped, so a
    request stays serviceable instead of losing the control it asked for.
    """
    options = current_run_options()
    if not options:
        return {}

    accepted = _sampler_signature_parameters(sampler_name)
    if not accepted:
        return {}

    applicable = {
        key: _bounded_value(key, value, sampler_name)
        for key, value in options.items()
        if key in accepted and not table_rejects_option(key, sampler_name, value)
    }
    ignored = sorted(set(options) - set(applicable))
    if ignored:
        logger.debug(f"Sampler {sampler_name} does not accept {ignored}; those options were dropped.")
    return applicable
