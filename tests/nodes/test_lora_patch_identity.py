"""Unit tests for the content-derived LoRA patch identity in ``node_lora_loader``.

Comfy rerolls ``ModelPatcher.patches_uuid`` on every ``add_patches`` call, so an identical repeat LoRA
stack cannot be recognised as the one already baked into the resident weights. ``_stable_patch_identity``
replaces that call-scoped uuid with one derived from the incoming patcher, the lora file, and the strength.
The safety direction is one-sided: every ingredient that could change what gets applied must change the
uuid, and an unreadable file must yield no identity at all.

``hordelib.nodes.node_lora_loader`` imports ``comfy`` and ``folder_paths`` at module scope, so these depend
on the session ``init_horde`` fixture and import locally rather than at collection time.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

_PARENT = uuid.UUID("11111111-1111-1111-1111-111111111111")
_OTHER_PARENT = uuid.UUID("22222222-2222-2222-2222-222222222222")


@pytest.fixture
def lora_file(tmp_path: Path) -> Path:
    """A stand-in lora file with stable contents; only its stat metadata matters here."""
    path = tmp_path / "some_lora.safetensors"
    path.write_bytes(b"lora-bytes")
    return path


@pytest.mark.usefixtures("init_horde")
class TestStablePatchIdentity:
    def test_identical_ingredients_produce_the_same_identity(self, lora_file: Path) -> None:
        from hordelib.nodes.node_lora_loader import _stable_patch_identity

        first = _stable_patch_identity(_PARENT, str(lora_file), 1.0)
        second = _stable_patch_identity(_PARENT, str(lora_file), 1.0)

        assert first is not None
        assert first == second, "the derivation is not deterministic, so no repeat can ever match"

    def test_a_different_parent_uuid_changes_the_identity(self, lora_file: Path) -> None:
        from hordelib.nodes.node_lora_loader import _stable_patch_identity

        assert _stable_patch_identity(_PARENT, str(lora_file), 1.0) != _stable_patch_identity(
            _OTHER_PARENT,
            str(lora_file),
            1.0,
        )

    def test_a_different_path_changes_the_identity(self, tmp_path: Path, lora_file: Path) -> None:
        from hordelib.nodes.node_lora_loader import _stable_patch_identity

        other = tmp_path / "other_lora.safetensors"
        other.write_bytes(lora_file.read_bytes())
        os.utime(other, ns=(os.stat(lora_file).st_atime_ns, os.stat(lora_file).st_mtime_ns))

        assert _stable_patch_identity(_PARENT, str(lora_file), 1.0) != _stable_patch_identity(
            _PARENT,
            str(other),
            1.0,
        )

    def test_a_changed_file_size_changes_the_identity(self, lora_file: Path) -> None:
        from hordelib.nodes.node_lora_loader import _stable_patch_identity

        before = _stable_patch_identity(_PARENT, str(lora_file), 1.0)
        original_stat = os.stat(lora_file)
        lora_file.write_bytes(b"lora-bytes-but-longer")
        os.utime(lora_file, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))

        assert before != _stable_patch_identity(_PARENT, str(lora_file), 1.0)

    def test_a_changed_mtime_changes_the_identity(self, lora_file: Path) -> None:
        from hordelib.nodes.node_lora_loader import _stable_patch_identity

        before = _stable_patch_identity(_PARENT, str(lora_file), 1.0)
        stat_result = os.stat(lora_file)
        os.utime(lora_file, ns=(stat_result.st_atime_ns, stat_result.st_mtime_ns + 1_000_000_000))

        assert before != _stable_patch_identity(_PARENT, str(lora_file), 1.0), (
            "a rewritten lora of identical size would keep its identity, so stale weights could be served"
        )

    def test_a_different_strength_changes_the_identity(self, lora_file: Path) -> None:
        from hordelib.nodes.node_lora_loader import _stable_patch_identity

        assert _stable_patch_identity(_PARENT, str(lora_file), 1.0) != _stable_patch_identity(
            _PARENT,
            str(lora_file),
            0.75,
        )

    def test_a_missing_file_yields_no_identity(self, tmp_path: Path) -> None:
        from hordelib.nodes.node_lora_loader import _stable_patch_identity

        assert _stable_patch_identity(_PARENT, str(tmp_path / "absent.safetensors"), 1.0) is None


def _tiny_cpu_patcher() -> Any:
    """A ``ModelPatcher`` around a two-linear-layer CPU module, enough to carry real patch identity."""
    import comfy.model_patcher
    import torch

    model = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Linear(8, 8))
    cpu = torch.device("cpu")
    return comfy.model_patcher.ModelPatcher(model, load_device=cpu, offload_device=cpu)


@pytest.mark.usefixtures("init_horde")
class TestPatchIdentityAssignment:
    """The seam the node uses: overwrite a freshly-patched clone's uuid, or leave comfy's in place."""

    def test_two_equivalent_applications_end_up_with_equal_identities(self, lora_file: Path) -> None:
        import torch

        from hordelib.nodes.node_lora_loader import _assign_stable_patch_identity

        base = _tiny_cpu_patcher()
        patch_payload = {"0.weight": torch.zeros(8, 8)}

        clone_one = base.clone()
        clone_one.add_patches(patch_payload, 1.0)
        clone_two = base.clone()
        clone_two.add_patches(patch_payload, 1.0)
        assert clone_one.patches_uuid != clone_two.patches_uuid, "comfy no longer rerolls the uuid per call"

        _assign_stable_patch_identity(base, clone_one, str(lora_file), 1.0, "model")
        _assign_stable_patch_identity(base, clone_two, str(lora_file), 1.0, "model")

        assert clone_one.patches_uuid == clone_two.patches_uuid
        assert clone_one.patches_uuid != base.patches_uuid, "the patched clone must not claim the base identity"

    def test_a_different_strength_yields_a_different_clone_identity(self, lora_file: Path) -> None:
        import torch

        from hordelib.nodes.node_lora_loader import _assign_stable_patch_identity

        base = _tiny_cpu_patcher()
        patch_payload = {"0.weight": torch.zeros(8, 8)}

        clone_one = base.clone()
        clone_one.add_patches(patch_payload, 1.0)
        clone_two = base.clone()
        clone_two.add_patches(patch_payload, 0.5)

        _assign_stable_patch_identity(base, clone_one, str(lora_file), 1.0, "model")
        _assign_stable_patch_identity(base, clone_two, str(lora_file), 0.5, "model")

        assert clone_one.patches_uuid != clone_two.patches_uuid

    def test_chained_applications_compose_the_incoming_identity(self, lora_file: Path) -> None:
        """A second loader node folds the first's identity in, so stack order and membership both count."""
        import torch

        from hordelib.nodes.node_lora_loader import _assign_stable_patch_identity

        from_base = _tiny_cpu_patcher()
        patch_payload = {"0.weight": torch.zeros(8, 8)}

        first_link = from_base.clone()
        first_link.add_patches(patch_payload, 1.0)
        _assign_stable_patch_identity(from_base, first_link, str(lora_file), 1.0, "model")

        second_link = first_link.clone()
        second_link.add_patches(patch_payload, 1.0)
        _assign_stable_patch_identity(first_link, second_link, str(lora_file), 1.0, "model")

        assert second_link.patches_uuid != first_link.patches_uuid, (
            "applying a second lora produced the single-lora identity, so a shorter stack would falsely match"
        )

    def test_a_stat_failure_leaves_the_comfy_assigned_uuid(self, tmp_path: Path) -> None:
        from hordelib.nodes.node_lora_loader import _assign_stable_patch_identity

        parent = SimpleNamespace(patches_uuid=_PARENT)
        clone_uuid = uuid.uuid4()
        clone = SimpleNamespace(patches_uuid=clone_uuid)

        _assign_stable_patch_identity(parent, clone, str(tmp_path / "absent.safetensors"), 1.0, "model")

        assert clone.patches_uuid == clone_uuid

    def test_a_side_without_patches_uuid_is_skipped(self, lora_file: Path) -> None:
        from hordelib.nodes.node_lora_loader import _assign_stable_patch_identity

        clone = SimpleNamespace()

        # No exception is the assertion: a comfy object without the attribute must not fault the job.
        _assign_stable_patch_identity(SimpleNamespace(patches_uuid=_PARENT), clone, str(lora_file), 1.0, "clip")
        _assign_stable_patch_identity(None, None, str(lora_file), 1.0, "clip")

        assert not hasattr(clone, "patches_uuid")
