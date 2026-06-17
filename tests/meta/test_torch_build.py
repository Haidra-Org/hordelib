"""Guards for the torch / torchvision / torchaudio build configuration.

These caught (and now prevent) a class of bug where torch was routed to a per-CUDA wheel index but
torchvision/torchaudio leaked from generic PyPI, so an ``--extra cu132`` install ended up with a
CUDA-13.2 torch and a CUDA-13.0 torchaudio (which refuses to import). torchaudio is now an optional,
non-default package; torch and torchvision must always share one build.
"""

from __future__ import annotations

import sys
import tomllib
import types
from pathlib import Path

import pytest

from hordelib.utils import torch_build

_REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT_PATH = _REPO_ROOT / "pyproject.toml"
UV_LOCK_PATH = _REPO_ROOT / "uv.lock"

# build extra -> the wheel index every routed torch package must resolve from for that build.
BUILD_INDEX = {
    "cu126": "https://download.pytorch.org/whl/cu126",
    "cu130": "https://download.pytorch.org/whl/cu130",
    "cu132": "https://download.pytorch.org/whl/cu132",
    "cpu": "https://download.pytorch.org/whl/cpu",
}
# The packages that must stay on the same build. torchaudio is intentionally excluded (optional).
ROUTED_PACKAGES = ("torch", "torchvision")


def _load_pyproject() -> dict:
    with open(PYPROJECT_PATH, "rb") as f:
        return tomllib.load(f)


def _load_lock() -> dict:
    with open(UV_LOCK_PATH, "rb") as f:
        return tomllib.load(f)


def _local_build_tag(version: str) -> str | None:
    return version.split("+", 1)[1] if "+" in version else None


def _dep_name(spec: str) -> str:
    """Strip version/extra/marker decoration from a dependency spec, leaving the bare package name."""
    name = spec.split(";", 1)[0].strip()
    for sep in ("[", "=", ">", "<", "~", "!", " "):
        name = name.split(sep, 1)[0]
    return name.strip()


# --- pyproject.toml routing -------------------------------------------------------------------


def test_torchaudio_is_not_a_default_dependency() -> None:
    """torchaudio must not be a default dependency (it has no cu132 wheel; audio is unsupported)."""
    deps = _load_pyproject()["project"]["dependencies"]
    offenders = [d for d in deps if _dep_name(d) == "torchaudio"]
    assert not offenders, f"torchaudio must not be in [project.dependencies]; found {offenders}"


def test_torch_and_torchvision_routed_to_every_build_index() -> None:
    """Both torch and torchvision must be routed to the matching wheel index for every build."""
    sources = _load_pyproject()["tool"]["uv"]["sources"]
    for package in ROUTED_PACKAGES:
        routes = {(entry["extra"], entry["index"]) for entry in sources[package]}
        for build in BUILD_INDEX:
            assert (build, f"pytorch-{build}") in routes, f"'{package}' not routed to pytorch-{build}"


def test_build_extras_list_routed_packages() -> None:
    """Each build extra must list torch and torchvision so their per-build source routing applies."""
    extras = _load_pyproject()["project"]["optional-dependencies"]
    for build in BUILD_INDEX:
        names = {_dep_name(d) for d in extras[build]}
        for package in ROUTED_PACKAGES:
            assert package in names, f"build extra '{build}' must list '{package}'"


# --- uv.lock consistency ----------------------------------------------------------------------


def _packages_by_name(lock: dict) -> dict[str, list[dict]]:
    by_name: dict[str, list[dict]] = {}
    for package in lock["package"]:
        by_name.setdefault(package["name"], []).append(package)
    return by_name


def test_lock_has_no_torchaudio() -> None:
    """torchaudio must not appear in the lock at all (it is no longer a dependency)."""
    by_name = _packages_by_name(_load_lock())
    assert "torchaudio" not in by_name, "torchaudio is still locked; it should have been removed"


def test_lock_pairs_torch_and_torchvision_per_build() -> None:
    """For every build, torch and torchvision must resolve from the matching index with the matching tag.

    A torchvision left on generic PyPI (the original bug) would have no ``+cuXXX`` entry sourced from the
    build's index, so this asserts the consistent pair exists for each build.
    """
    by_name = _packages_by_name(_load_lock())
    for build, index_url in BUILD_INDEX.items():
        for package in ROUTED_PACKAGES:
            matches = [
                p
                for p in by_name.get(package, [])
                if p.get("source", {}).get("registry") == index_url and _local_build_tag(p["version"]) == build
            ]
            assert matches, (
                f"no {package} entry in uv.lock with tag '+{build}' sourced from {index_url}; "
                "torch and torchvision have drifted apart for this build"
            )


# --- runtime preflight ------------------------------------------------------------------------


def test_local_build_tag() -> None:
    assert _local_build_tag("2.12.0+cu132") == "cu132"
    assert _local_build_tag("0.27.0+cpu") == "cpu"
    assert _local_build_tag("2.12.0") is None


def test_preflight_passes_when_tags_match(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_torch = types.SimpleNamespace(__version__="2.12.0+cu132")
    fake_torchvision = types.SimpleNamespace(__version__="0.27.0+cu132")
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torchvision", fake_torchvision)
    monkeypatch.setattr(torch_build, "_torchaudio_installed", lambda: False)
    torch_build.verify_torch_build_consistency()  # must not raise


def test_preflight_raises_when_tags_differ(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_torch = types.SimpleNamespace(__version__="2.12.0+cu132")
    fake_torchvision = types.SimpleNamespace(__version__="0.27.0+cu130")
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torchvision", fake_torchvision)
    monkeypatch.setattr(torch_build, "_torchaudio_installed", lambda: False)
    with pytest.raises(RuntimeError, match="different backends"):
        torch_build.verify_torch_build_consistency()


# --- torchaudio stub --------------------------------------------------------------------------


@pytest.fixture
def _restore_torchaudio_modules():
    saved = {name: mod for name, mod in sys.modules.items() if name == "torchaudio" or name.startswith("torchaudio.")}
    for name in list(saved):
        del sys.modules[name]
    try:
        yield
    finally:
        for name in [n for n in sys.modules if n == "torchaudio" or n.startswith("torchaudio.")]:
            del sys.modules[name]
        sys.modules.update(saved)


def test_stub_installed_when_absent(monkeypatch: pytest.MonkeyPatch, _restore_torchaudio_modules: None) -> None:
    monkeypatch.setattr(torch_build, "_torchaudio_installed", lambda: False)

    assert torch_build.ensure_torchaudio_importable() is True

    import torchaudio  # the stub

    # Import-time references comfy makes must succeed...
    resample = torchaudio.functional.resample
    from torchaudio.transforms import MelSpectrogram

    # ...but actually using audio must raise the actionable error.
    with pytest.raises(RuntimeError, match="Audio support is not installed"):
        resample(object(), 16000, 44100)
    with pytest.raises(RuntimeError, match="Audio support is not installed"):
        MelSpectrogram(sample_rate=44100)


def test_stub_not_installed_when_present(monkeypatch: pytest.MonkeyPatch, _restore_torchaudio_modules: None) -> None:
    monkeypatch.setattr(torch_build, "_torchaudio_installed", lambda: True)
    assert torch_build.ensure_torchaudio_importable() is False
    assert "torchaudio" not in sys.modules


def test_stub_is_discoverable_via_find_spec(
    monkeypatch: pytest.MonkeyPatch,
    _restore_torchaudio_modules: None,
) -> None:
    """The stub must be resolvable via importlib.util.find_spec, not just `import`/attribute access.

    transformers' is_torchaudio_available() (evaluated eagerly at `import transformers`, which ComfyUI
    pulls in) calls importlib.util.find_spec('torchaudio'). A stub built from a bare types.ModuleType
    leaves __spec__ = None, which makes find_spec raise 'ValueError: torchaudio.__spec__ is None' and
    crashes the worker's inference process at comfy import. This guards that regression.
    """
    import importlib.util

    monkeypatch.setattr(torch_build, "_torchaudio_installed", lambda: False)
    assert torch_build.ensure_torchaudio_importable() is True

    for name in ("torchaudio", "torchaudio.functional", "torchaudio.transforms"):
        spec = importlib.util.find_spec(name)  # must not raise ValueError
        assert spec is not None, f"find_spec({name!r}) returned None"
