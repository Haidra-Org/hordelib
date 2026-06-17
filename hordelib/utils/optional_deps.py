"""Lazy optional-dependency gating for hordelib.

hordelib aims to stay close to base ComfyUI: heavy or platform-constrained features are
guarded so that the packages they need are only required when the feature is actually used.
A call site asks for an optional package via :func:`require` or :func:`import_optional`; if
the package is absent the caller gets an actionable "install this extra" error instead of a
bare ``ModuleNotFoundError`` surfacing from deep in the call stack.

This module intentionally only DETECTS dependencies (via ``importlib.util.find_spec``); it
never installs anything at runtime. Extras are resolved at lock/sync time by the packaging
tool, which keeps environments reproducible and auditable (the opposite of runtime
``pip install`` machinery).
"""

import importlib
import importlib.util
from types import ModuleType

__all__ = ["MissingOptionalDependency", "import_optional", "require"]

# The PyPI distribution name differs from the import name (`hordelib`); install hints must use
# the distribution name so `pip install ...[extra]` actually works.
_DISTRIBUTION_NAME = "horde-engine"


class MissingOptionalDependency(ImportError):
    """Raised when an optional feature is used but its backing package is not installed."""

    def __init__(self, package: str, *, extra: str, feature: str) -> None:
        self.package = package
        self.extra = extra
        self.feature = feature
        super().__init__(
            f"{feature} requires the optional package '{package}', which is not installed. "
            f"Install it with: pip install {_DISTRIBUTION_NAME}[{extra}]",
        )


def require(package: str, *, extra: str, feature: str) -> None:
    """Ensure an optional dependency is importable, or raise an actionable error.

    Args:
        package: The top-level import name to probe (e.g. ``"rembg"``).
        extra: The ``horde-engine`` extra that provides it (e.g. ``"rembg"``).
        feature: Human-readable name of the feature needing it, used in the error message.

    Raises:
        MissingOptionalDependency: If ``package`` cannot be found in the environment.
    """
    if importlib.util.find_spec(package) is None:
        raise MissingOptionalDependency(package, extra=extra, feature=feature)


def import_optional(module: str, *, extra: str, feature: str) -> ModuleType:
    """Import and return an optional module, raising an actionable error if it is absent.

    The presence probe uses the top-level package of ``module`` so that, for example,
    ``import_optional("horde_image_utilities.rembg", ...)`` reports the installable package
    rather than the dotted submodule.

    Args:
        module: The (possibly dotted) module path to import.
        extra: The ``horde-engine`` extra that provides it.
        feature: Human-readable name of the feature needing it, used in the error message.

    Returns:
        The imported module.

    Raises:
        MissingOptionalDependency: If the module's top-level package is not installed.
    """
    top_level = module.split(".", 1)[0]
    require(top_level, extra=extra, feature=feature)
    return importlib.import_module(module)
