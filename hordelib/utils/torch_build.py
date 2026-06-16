"""Guards for the torch / torchvision / torchaudio build trio.

PyTorch publishes a separate wheel per CUDA (or CPU) build, and the three packages MUST all come
from the same build: torchaudio refuses to import when its compiled CUDA version differs from
torch's (``_check_cuda_version``), and torchvision's compiled ops misbehave the same way. The worker
installs torch and torchvision from a per-build wheel index (``--extra cu126``/``cu130``/``cu132``/
``cpu``); routing keeps their CUDA tags in lockstep.

torchaudio is the odd one out: it has no ``+cu132`` wheel at all, and audio is currently unsupported,
so it is intentionally NOT a default dependency. Two helpers here keep that arrangement honest:

* :func:`verify_torch_build_consistency` fails fast with an actionable message when the installed
  torch and torchvision (or a hand-installed torchaudio) disagree on their build, instead of letting
  the mismatch surface as a cryptic traceback deep inside ComfyUI.
* :func:`ensure_torchaudio_importable` registers a lazy stub ``torchaudio`` when the real package is
  absent, so ComfyUI's several eager ``import torchaudio`` statements still load for image/video work;
  the stub only raises (with guidance) if audio functionality is actually exercised.
"""

import importlib.util
import sys
import types

from loguru import logger

_AUDIO_UNAVAILABLE_MSG = (
    "Audio support is not installed: torchaudio was deliberately left out of the default install "
    "(it has no CUDA 13.2 wheel, and audio generation is currently unsupported). The image/video "
    "pipeline does not need it. To enable audio, install a torchaudio build matching your torch "
    "CUDA version, e.g. `uv pip install torchaudio --index-url https://download.pytorch.org/whl/<build>` "
    "(cu126/cu130/cpu; there is no cu132 wheel). Triggered by use of `{attr}`."
)


def _torchaudio_installed() -> bool:
    """Return True when a real torchaudio is importable, without importing/executing it.

    ``find_spec`` avoids running torchaudio's import-time CUDA check (the very thing we want
    :func:`verify_torch_build_consistency` to own), and ignores any stub already in ``sys.modules``.
    """
    try:
        return importlib.util.find_spec("torchaudio") is not None
    except (ImportError, ValueError):
        return False


class _RaiseOnUse:
    """Placeholder returned by the torchaudio stub; raises only when actually called.

    Attribute access returns another placeholder so chained lookups (e.g.
    ``torchaudio.functional.resample``) resolve, and the error is deferred until the call site that
    truly needs audio. Dunder lookups raise ``AttributeError`` so the object still behaves sanely
    (e.g. is not mistaken for something with a ``__len__``).
    """

    def __init__(self, attr_path: str) -> None:
        self._attr_path = attr_path

    def __call__(self, *args: object, **kwargs: object) -> object:
        raise RuntimeError(_AUDIO_UNAVAILABLE_MSG.format(attr=self._attr_path))

    def __getattr__(self, item: str) -> "_RaiseOnUse":
        if item.startswith("__") and item.endswith("__"):
            raise AttributeError(item)
        return _RaiseOnUse(f"{self._attr_path}.{item}")


def _make_lazy_module(name: str) -> types.ModuleType:
    """Build a module whose unknown attributes resolve to lazy :class:`_RaiseOnUse` placeholders."""
    module = types.ModuleType(name)

    def __getattr__(item: str) -> _RaiseOnUse:
        if item.startswith("__") and item.endswith("__"):
            raise AttributeError(item)
        return _RaiseOnUse(f"{name}.{item}")

    module.__getattr__ = __getattr__  # type: ignore[method-assign]
    return module


def ensure_torchaudio_importable() -> bool:
    """Register a lazy stub ``torchaudio`` when the real package is absent.

    ComfyUI imports torchaudio eagerly in several modules that load at node-registration time
    (``comfy_extras.nodes_audio`` / ``nodes_lt`` / ``nodes_wandancer``, ``comfy.audio_encoders.*``,
    ``comfy.ldm.lightricks.vae.audio_vae``). With audio unsupported and torchaudio omitted, those
    imports would otherwise crash startup, so we satisfy them with a stub that defers failure to the
    point audio is actually used. Call this BEFORE ComfyUI is imported.

    Returns:
        True if a stub was installed; False if a real (or already-registered) torchaudio is present.
    """
    if "torchaudio" in sys.modules:
        return False
    if _torchaudio_installed():
        return False

    torchaudio = _make_lazy_module("torchaudio")
    functional = _make_lazy_module("torchaudio.functional")
    transforms = _make_lazy_module("torchaudio.transforms")
    # Bind the submodules as real attributes so both `import torchaudio; torchaudio.functional.x`
    # and `from torchaudio.transforms import X` resolve without hitting the module __getattr__.
    torchaudio.functional = functional  # type: ignore[attr-defined]
    torchaudio.transforms = transforms  # type: ignore[attr-defined]
    torchaudio.__version__ = "0.0.0+horde-stub"  # type: ignore[attr-defined]
    sys.modules["torchaudio"] = torchaudio
    sys.modules["torchaudio.functional"] = functional
    sys.modules["torchaudio.transforms"] = transforms

    logger.info(
        "torchaudio is not installed; registered a lazy stub so ComfyUI imports for image/video "
        "work. Audio operations will raise if actually used.",
    )
    return True


def _local_build_tag(version: str) -> str | None:
    """Return a wheel's local build tag (``2.12.0+cu132`` -> ``cu132``), or None when untagged."""
    return version.split("+", 1)[1] if "+" in version else None


def verify_torch_build_consistency() -> None:
    """Fail fast when the installed torch trio disagrees on its CUDA/CPU build.

    torch and torchvision wheels from a PyTorch index carry a matching local build tag (e.g.
    ``+cu132``); a mismatch means one leaked from generic PyPI (a different CUDA build) and the pair
    will misbehave. torchaudio carries no reliable tag, so when one is installed we let its own
    import-time check run and re-raise any failure with guidance. Raises ``RuntimeError`` on any
    mismatch.
    """
    import torch

    try:
        import torchvision  # type: ignore[import-untyped]
    except ImportError:
        # torchvision is always expected, but a missing one is a different (clearer) failure that
        # the normal import path will report; don't mask it here.
        return

    torch_tag = _local_build_tag(torch.__version__)
    torchvision_tag = _local_build_tag(torchvision.__version__)
    if torch_tag != torchvision_tag:
        raise RuntimeError(
            "PyTorch and torchvision were built for different backends "
            f"(torch={torch.__version__!r}, torchvision={torchvision.__version__!r}). They must come "
            "from the same wheel index. Reinstall with the matching build extra, e.g. "
            "`uv sync --locked --extra <build>` where <build> is one of cu126/cu130/cu132/cpu.",
        )

    if _torchaudio_installed():
        try:
            # Importing torchaudio runs its _check_cuda_version, which validates its build against torch's.
            importlib.import_module("torchaudio")
        except RuntimeError as exc:
            raise RuntimeError(
                "An installed torchaudio was built against a different CUDA version than PyTorch: "
                f"{exc} torchaudio is optional and omitted from the default install; either remove it "
                "or install a build matching your torch CUDA version "
                "(`uv pip install torchaudio --index-url https://download.pytorch.org/whl/<build>`; "
                "there is no cu132 wheel).",
            ) from exc
