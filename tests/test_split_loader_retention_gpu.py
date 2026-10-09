"""Check on a real GPU that a retained split-files model keeps one text encoder and one VAE loaded across jobs.

With the diffusion model retained (``defer_vram_unload``), no unload runs between jobs. ComfyUI tracks loaded
models by patcher identity, so a stock ``CLIPLoader``/``VAELoader`` building a fresh copy every job left each
earlier job's text encoder and VAE loaded beside the new ones. Serving both through the component cache keeps
one patcher per file, so ComfyUI finds it already loaded.

The run with the two loader hijacks disabled is printed and carries no assertion. On a card that cannot hold
the model's components together, ComfyUI frees the older copies itself to make room, so the duplication only
shows where everything fits. Anima fits a 16 GB card and Z-Image Turbo does not.

Marked ``slow`` plus each model's marker. Run manually and serially, for example::

    uv run --no-sync pytest tests/test_split_loader_retention_gpu.py -m slow
"""

from __future__ import annotations

from collections import Counter

import pytest

from hordelib.horde import HordeLib

_SPLIT_LOADER_PATCHES = ["clip_loader_load_clip", "vae_loader_load_vae"]


def _split_model_job(model_name: str, seed: int) -> dict:
    return {
        "sampler_name": "k_euler",
        "cfg_scale": 1.0,
        "denoising_strength": 1.0,
        "seed": seed,
        "height": 512,
        "width": 512,
        "karras": False,
        "tiling": False,
        "hires_fix": False,
        "clip_skip": 1,
        "prompt": "a lighthouse on a cliff at dusk, illustration",
        "ddim_steps": 4,
        "n_iter": 1,
        "model": model_name,
    }


def _loaded_model_classes() -> Counter[str]:
    import comfy.model_management

    return Counter(
        type(loaded.model.model).__name__
        for loaded in comfy.model_management.current_loaded_models
        if loaded.model is not None
    )


def _run_retained_jobs(hordelib_instance: HordeLib, model_name: str, job_count: int) -> Counter[str]:
    for seed in range(job_count):
        results = hordelib_instance.basic_inference(_split_model_job(model_name, seed), defer_vram_unload=True)
        assert results
        assert not results[0].faults
    return _loaded_model_classes()


@pytest.mark.slow
@pytest.mark.parametrize(
    "model_fixture_name",
    [
        pytest.param("anima_turbo_base_model_name", marks=pytest.mark.default_anima_model),
        pytest.param("z_image_turbo_base_model_name", marks=pytest.mark.default_z_image_turbo_model),
    ],
)
def test_retained_split_model_keeps_one_copy_of_each_component_loaded(
    hordelib_instance: HordeLib,
    model_fixture_name: str,
    request: pytest.FixtureRequest,
) -> None:
    from hordelib.comfy_horde import unload_all_models_ram
    from hordelib.execution.comfy_patches import temporary_monkeypatch_state
    from hordelib.execution.component_cache import ComponentSlotKind, held_components

    model_name: str = request.getfixturevalue(model_fixture_name)

    unload_all_models_ram()
    with temporary_monkeypatch_state(enable=False, patch_names=_SPLIT_LOADER_PATCHES):
        without_cache = _run_retained_jobs(hordelib_instance, model_name, job_count=3)
    unload_all_models_ram()

    with_cache = _run_retained_jobs(hordelib_instance, model_name, job_count=3)
    held_kinds = Counter(snapshot.kind for snapshot in held_components())
    unretained_results = hordelib_instance.basic_inference(_split_model_job(model_name, 99), defer_vram_unload=False)
    held_kinds_after_unretained = Counter(snapshot.kind for snapshot in held_components())
    unload_all_models_ram()

    assert unretained_results
    assert held_kinds_after_unretained[ComponentSlotKind.CLIP] == 0
    assert held_kinds_after_unretained[ComponentSlotKind.VAE] == 0

    print(f"loaded models without the split-loader cache: {dict(without_cache)}")
    print(f"loaded models with the split-loader cache: {dict(with_cache)}")
    # A model whose components do not fit the card together leaves nothing loaded at the end, which the
    # duplicate check cannot distinguish from a fix. The held entries show the loaders were served from the
    # cache, since the stock loaders put nothing there.
    assert all(count == 1 for count in with_cache.values()), with_cache
    assert held_kinds[ComponentSlotKind.CLIP] == 1
    assert held_kinds[ComponentSlotKind.VAE] == 1
