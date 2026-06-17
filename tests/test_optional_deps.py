"""Unit tests for the optional-dependency gate (``hordelib.utils.optional_deps``)."""

import pytest

from hordelib.utils.optional_deps import (
    MissingOptionalDependency,
    import_optional,
    require,
)


def test_require_passes_for_present_package() -> None:
    # A stdlib package is always importable, so this must not raise.
    require("importlib", extra="cpu", feature="unit test")


def test_require_raises_actionable_error_for_absent_package() -> None:
    with pytest.raises(MissingOptionalDependency) as exc_info:
        require("definitely_not_installed_zzz", extra="rembg", feature="strip_background")
    message = str(exc_info.value)
    # The hint must name the installable distribution + extra, not the import name.
    assert "horde-engine[rembg]" in message
    assert "strip_background" in message
    assert exc_info.value.package == "definitely_not_installed_zzz"
    assert exc_info.value.extra == "rembg"


def test_import_optional_returns_module_when_present() -> None:
    module = import_optional("json", extra="cpu", feature="unit test")
    assert module.__name__ == "json"


def test_import_optional_probes_top_level_package() -> None:
    # A dotted path whose top-level package is absent should report that top-level package.
    with pytest.raises(MissingOptionalDependency) as exc_info:
        import_optional(
            "definitely_not_installed_zzz.submodule",
            extra="layerdiffuse",
            feature="layer diffusion",
        )
    assert exc_info.value.package == "definitely_not_installed_zzz"
    assert "horde-engine[layerdiffuse]" in str(exc_info.value)
