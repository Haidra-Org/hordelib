from pathlib import Path

import pytest

from hordelib.comfy_horde import Comfy_Horde
from hordelib.horde import HordeLib
from hordelib.initialisation import _validate_comfyui_import_state


def _managed_package_paths(comfy_path: Path) -> frozenset[Path]:
    return frozenset({(comfy_path / "comfy").resolve()})


def test_checkout_sync_guard_rejects_preloaded_comfy_even_at_the_pinned_revision() -> None:
    pinned = "a" * 40
    comfy_path = Path("ComfyUI")

    with pytest.raises(RuntimeError, match="imported before hordelib.initialise"):
        _validate_comfyui_import_state(
            comfy_path,
            pinned,
            loaded=("comfy", "comfy.sync_guard_probe"),
            package_paths=_managed_package_paths(comfy_path),
            current_ref=pinned,
        )


def test_checkout_sync_guard_allows_an_unloaded_runtime() -> None:
    _validate_comfyui_import_state(
        Path("ComfyUI"),
        "a" * 40,
        loaded=(),
        package_paths=frozenset(),
        current_ref=None,
    )


def test_checkout_sync_guard_rejects_mixed_comfyui_revisions() -> None:
    old = "a" * 40
    pinned = "b" * 40
    comfy_path = Path("ComfyUI")
    with pytest.raises(RuntimeError, match="after ComfyUI has been imported") as exc_info:
        _validate_comfyui_import_state(
            comfy_path,
            pinned,
            loaded=("comfy", "comfy.sync_guard_probe"),
            package_paths=_managed_package_paths(comfy_path),
            current_ref=old,
        )

    assert old in str(exc_info.value)
    assert pinned in str(exc_info.value)
    assert "comfy.sync_guard_probe" in str(exc_info.value)


def test_checkout_sync_guard_rejects_a_plain_comfy_stub() -> None:
    pinned = "a" * 40

    with pytest.raises(RuntimeError, match="incompatible module named 'comfy'"):
        _validate_comfyui_import_state(
            Path("ComfyUI"),
            pinned,
            loaded=("comfy",),
            package_paths=frozenset(),
            current_ref=pinned,
        )


def test_find_comfyui(init_horde):
    import execution

    assert hasattr(execution, "get_input_data")


def test_instantiation(hordelib_instance: HordeLib):
    assert isinstance(hordelib_instance.backend.comfy_horde, Comfy_Horde)


def test_path():  # XXX
    from hordelib.config_path import set_system_path

    set_system_path()
