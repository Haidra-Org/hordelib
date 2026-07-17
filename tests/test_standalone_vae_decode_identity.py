"""Real-GPU identity check for the standalone-VAE decode path.

The standalone path must be a pure performance optimisation: decoding a latent through the VAE loaded from a
checkpoint's pre-extracted standalone file must produce output byte-identical to decoding through the VAE
subset-loaded from the monolithic checkpoint, because the extraction copies the VAE tensors byte-for-byte.
This module proves that in a single process, and separately proves that seamless tiling requested through the
standalone path actually retunes the VAE's Conv2d padding to circular.

These are real-GPU tests, marked ``slow`` plus the checkpoint's model marker (matching
``tests/test_stage_disaggregation.py``), and are deselected by the CI default ``-m "not slow"``. Run manually
and serially, for example::

    uv run --no-sync pytest tests/test_standalone_vae_decode_identity.py -m slow
"""

from __future__ import annotations

import pytest
import torch

from hordelib.shared_model_manager import SharedModelManager


def _resolve_checkpoint_name(shared_model_manager: type[SharedModelManager], model_name: str) -> str:
    """Return the on-disk checkpoint file name for *model_name* (the loader's ``file_type is None`` file)."""
    compvis = shared_model_manager.manager.compvis
    assert compvis is not None
    entries = compvis.get_model_filenames(model_name)
    assert entries, f"no files for {model_name}"
    file_path = entries[0]["file_path"]
    return str(file_path) if file_path.is_absolute() else file_path.name


def _fixed_latent() -> torch.Tensor:
    """Return a deterministic SD-latent-shaped tensor (1x4x64x64); the VAE decode is architecture-shared."""
    generator = torch.Generator().manual_seed(1234)
    return torch.randn(1, 4, 64, 64, generator=generator)


@pytest.mark.slow
@pytest.mark.default_sdxl_model
class TestStandaloneVaeDecodeIdentity:
    def test_standalone_and_subset_decode_are_identical(
        self,
        init_horde,
        shared_model_manager: type[SharedModelManager],
        sdxl_refined_model_name: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import folder_paths

        from hordelib.nodes.node_model_loader import HordeCheckpointLoader

        model_name = sdxl_refined_model_name
        compvis = shared_model_manager.manager.compvis
        assert compvis is not None
        assert compvis.download_model(model_name)

        # The download hook builds the sidecar and extracts the VAE; make it explicit so the test is
        # self-contained even if the model was already present before this run.
        compvis._ensure_sidecars_for_model(model_name)
        ckpt_name = _resolve_checkpoint_name(shared_model_manager, model_name)
        if not ckpt_name.endswith(".safetensors"):
            pytest.skip(f"{model_name} resolves to {ckpt_name}: a non-safetensors container cannot carry a sidecar")
        assert folder_paths.get_full_path("checkpoints", ckpt_name) is not None

        loader = HordeCheckpointLoader()

        # Subset path: force the prior behaviour with the kill-switch so the VAE is sliced from the monolith.
        monkeypatch.setenv("HORDE_DISABLE_STANDALONE_VAE_PATH", "1")
        SharedModelManager.manager._models_in_ram.evict_all()
        subset_result = loader.load_checkpoint(
            will_load_loras=False,
            seamless_tiling_enabled=False,
            horde_model_name=model_name,
            file_type=None,
            output_model=False,
            output_vae=True,
            output_clip=False,
        )
        subset_vae = subset_result[2]
        assert subset_vae is not None

        # Standalone path: the extracted file, keyed by content hash.
        monkeypatch.delenv("HORDE_DISABLE_STANDALONE_VAE_PATH", raising=False)
        SharedModelManager.manager._models_in_ram.evict_all()
        standalone_result = loader.load_checkpoint(
            will_load_loras=False,
            seamless_tiling_enabled=False,
            horde_model_name=model_name,
            file_type=None,
            output_model=False,
            output_vae=True,
            output_clip=False,
        )
        standalone_vae = standalone_result[2]
        assert standalone_vae is not None
        # The standalone serve must have cached by content identity, not the model name.
        assert any(
            snapshot.identity.startswith("vae@")
            for snapshot in SharedModelManager.manager._models_in_ram.held_report()
        )

        latent = _fixed_latent()
        subset_image = subset_vae.decode(latent.clone())
        standalone_image = standalone_vae.decode(latent.clone())

        assert torch.equal(subset_image, standalone_image), (
            "standalone-extracted VAE decode diverged from the subset-loaded VAE decode; the extraction is "
            "supposed to be byte-identical VAE weights"
        )

    def test_seamless_tiling_applies_through_standalone_path(
        self,
        init_horde,
        shared_model_manager: type[SharedModelManager],
        sdxl_refined_model_name: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from hordelib.nodes.node_model_loader import HordeCheckpointLoader

        model_name = sdxl_refined_model_name
        compvis = shared_model_manager.manager.compvis
        assert compvis is not None
        assert compvis.download_model(model_name)
        compvis._ensure_sidecars_for_model(model_name)

        loader = HordeCheckpointLoader()
        monkeypatch.delenv("HORDE_DISABLE_STANDALONE_VAE_PATH", raising=False)
        SharedModelManager.manager._models_in_ram.evict_all()
        result = loader.load_checkpoint(
            will_load_loras=False,
            seamless_tiling_enabled=True,
            horde_model_name=model_name,
            file_type=None,
            output_model=False,
            output_vae=True,
            output_clip=False,
        )
        vae = result[2]
        assert vae is not None

        circular_padding_present = any(
            isinstance(module, torch.nn.Conv2d) and module.padding_mode == "circular"
            for module in vae.first_stage_model.modules()
        )
        assert circular_padding_present, "seamless tiling did not retune the standalone VAE's Conv2d padding"
