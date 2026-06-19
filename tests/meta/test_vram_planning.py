"""The torch-free VRAM-planning helpers must mirror ComfyUI's reserve/budget formulas exactly.

The worker's orchestrator uses these (without loading torch) to forecast whether a model will stream its
weights from host RAM. They must reproduce ``comfy/model_management.py``'s ``minimum_inference_memory()``
and ``maximum_vram_for_weights()`` so the forecast lines up with the backend's real split point. These are
pure-formula assertions (no torch/comfy needed); the worker-side integration validates against a live
device.
"""

from __future__ import annotations

from hordelib.vram_planning import (
    compute_extra_reserved_mb,
    compute_inference_reserve_mb,
    compute_weight_budget_mb,
)

# A 16GB Windows card (the live 4070 Ti SUPER): total > 15GB, so ComfyUI's EXTRA_RESERVED_VRAM is
# 600 + 100 = 700MB, and minimum_inference_memory() is 0.8*1024 + 700 = 1519MB.
_TOTAL_16GB = 16375


def test_extra_reserved_mirrors_comfy_platform_rules() -> None:
    """The extra reserve follows ComfyUI: 400 base, 600 on Windows, +100 over 15GB, override wins."""
    assert compute_extra_reserved_mb(_TOTAL_16GB, is_windows=False) == 400
    assert compute_extra_reserved_mb(8000, is_windows=True) == 600
    assert compute_extra_reserved_mb(_TOTAL_16GB, is_windows=True) == 700
    # An explicit --reserve-vram (GB) overrides the platform rules entirely.
    assert compute_extra_reserved_mb(_TOTAL_16GB, is_windows=True, reserve_vram_gb=2.0) == 2048


def test_inference_reserve_matches_comfy_minimum_inference_memory() -> None:
    """minimum_inference_memory() == 0.8GB working set + extra reserve; 1519MB on the live 16GB box."""
    assert compute_inference_reserve_mb(_TOTAL_16GB, is_windows=True) == 1519
    assert compute_inference_reserve_mb(8000, is_windows=False) == round(0.8 * 1024) + 400


def test_weight_budget_matches_comfy_maximum_vram_for_weights() -> None:
    """maximum_vram_for_weights() == total*0.88 - inference_reserve."""
    expected = round(_TOTAL_16GB * 0.88) - 1519
    assert compute_weight_budget_mb(_TOTAL_16GB, is_windows=True) == expected
    # The budget must be strictly below total so weights alone can never claim the entire device.
    assert compute_weight_budget_mb(_TOTAL_16GB, is_windows=True) < _TOTAL_16GB
