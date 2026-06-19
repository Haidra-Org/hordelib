"""Pure-Python mirrors of ComfyUI's VRAM accounting, usable before (or without) torch/ComfyUI.

The worker's orchestrator decides whether a model will fit a device *before* dispatching it to a
torch child, and it must do so without importing torch (a ~500MB RSS cost it cannot pay; see the
torch-free-orchestrator invariant in horde-worker-reGen). ComfyUI's decision to keep a model's
weights resident or stream them from host RAM hinges on one quantity computed in
``comfy/model_management.py``: ``minimum_inference_memory()``, the free VRAM it keeps for activations.
A model whose weights leave less than that free at load time is offloaded and streamed during
sampling, which collapses the step rate on a memory-constrained device. This module reproduces that
formula exactly from the device's total VRAM and platform so the worker can predict streaming and gate
accordingly.

Keep these constants in lockstep with the vendored ``comfy/model_management.py``. For runtime-exact
agreement (across ComfyUI version skew) prefer the live value a torch child reports; this module is
the torch-free estimate the orchestrator uses for admission and the no-boot benchmark planner.

Comfy-free and torch-free: safe to import in the orchestrator and the benchmark planner.
"""

from __future__ import annotations

import sys

# Mirror of ``comfy/model_management.py`` module-level constants. ComfyUI reserves a base amount of
# VRAM for other applications, raised on Windows (shared-VRAM driver behaviour) and again on cards
# above 15GB. An explicit ``--reserve-vram`` overrides all of it.
_BASE_EXTRA_RESERVED_MB = 400
_WINDOWS_EXTRA_RESERVED_MB = 600
_WINDOWS_LARGE_CARD_BONUS_MB = 100
_LARGE_CARD_THRESHOLD_MB = 15 * 1024
# The fixed activation working set ComfyUI keeps free on top of the reserve (0.8 GB).
_INFERENCE_WORKING_SET_MB = 0.8 * 1024
# ComfyUI scales total VRAM by this before subtracting the reserve to get the static weight budget.
_WEIGHT_BUDGET_FRACTION = 0.88


def _resolve_is_windows(is_windows: bool | None) -> bool:
    """Return whether to apply ComfyUI's Windows reserve rules (autodetect when not forced)."""
    return sys.platform.startswith("win") if is_windows is None else is_windows


def compute_extra_reserved_mb(
    total_vram_mb: int,
    *,
    is_windows: bool | None = None,
    reserve_vram_gb: float | None = None,
) -> int:
    """Return ComfyUI's ``EXTRA_RESERVED_VRAM`` (MB) for a device of ``total_vram_mb``.

    Mirrors ``comfy/model_management.py``: a base reserve, higher on Windows, with a bonus on cards
    above 15GB. An explicit ``--reserve-vram`` (``reserve_vram_gb``) overrides all of it.
    """
    if reserve_vram_gb is not None:
        return round(reserve_vram_gb * 1024)
    if _resolve_is_windows(is_windows):
        reserved = _WINDOWS_EXTRA_RESERVED_MB
        if total_vram_mb > _LARGE_CARD_THRESHOLD_MB:
            reserved += _WINDOWS_LARGE_CARD_BONUS_MB
        return reserved
    return _BASE_EXTRA_RESERVED_MB


def compute_inference_reserve_mb(
    total_vram_mb: int,
    *,
    is_windows: bool | None = None,
    reserve_vram_gb: float | None = None,
) -> int:
    """Return ComfyUI's ``minimum_inference_memory()`` (MB): the free VRAM it keeps for activations.

    A model whose weights leave less than this free at load time is partially offloaded to host RAM
    and streamed during sampling (the slow path). Equal to the 0.8GB working set plus the extra
    reserve, so it is the headroom the worker must preserve to avoid weight streaming.
    """
    extra = compute_extra_reserved_mb(total_vram_mb, is_windows=is_windows, reserve_vram_gb=reserve_vram_gb)
    return round(_INFERENCE_WORKING_SET_MB + extra)


def compute_weight_budget_mb(
    total_vram_mb: int,
    *,
    is_windows: bool | None = None,
    reserve_vram_gb: float | None = None,
) -> int:
    """Return ComfyUI's static ``maximum_vram_for_weights()`` (MB): the most weights can occupy.

    ``total * 0.88 - inference_reserve``. A model whose weights exceed this budget cannot be held
    resident even alone and will always stream; the runtime per-load budget can be lower still when
    other models are resident, which is why the worker also reasons about reclaimable siblings.
    """
    reserve = compute_inference_reserve_mb(total_vram_mb, is_windows=is_windows, reserve_vram_gb=reserve_vram_gb)
    return round(total_vram_mb * _WEIGHT_BUDGET_FRACTION) - reserve
