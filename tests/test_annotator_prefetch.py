"""GPU-free tests for the controlnet-annotator prefetch hook.

The prefetch pre-places annotator checkpoints through the unified gated-mirror-then-origin download engine so
``custom_hf_download`` finds them present and skips its own HuggingFace download. The model-reference catalog and
the engine's record-free fetch helper live in ``horde_model_reference``; the installed copy in this venv may be
older and lack them (the prefetch then degrades to a quiet no-op), so these tests inject fakes for the catalog
and the engine to exercise the hook's own logic without a network or a specific hmr version.
"""

from __future__ import annotations

import horde_model_reference

# Imported before any package-attribute patching so it binds the *real* download_engine at its module top,
# independent of the fake we later bind onto the horde_model_reference package for the prefetch's own import.
import hordelib.model_manager.base  # noqa: F401
import hordelib.preload as preload


class _FakeFile:
    """A structural stand-in for ``horde_model_reference.annotator_catalog.AnnotatorFile``."""

    def __init__(
        self,
        filename: str,
        *,
        sha256: str | None = None,
        preprocessors: tuple[str, ...] = (),
        control_types: tuple[str, ...] = (),
    ) -> None:
        self.filename = filename
        self.sha256 = sha256
        self.preprocessors = preprocessors
        self.control_types = control_types

    @property
    def relative_path(self) -> str:
        return f"lllyasviel/Annotators/{self.filename}"

    @property
    def origin_url(self) -> str:
        return f"https://huggingface.co/lllyasviel/Annotators/resolve/main/{self.filename}"


def _inject_catalog_and_engine(monkeypatch, files, download):
    """Bind a fake catalog (``ANNOTATOR_FILES``) and engine (``download_addressed_file``) onto the hmr package."""
    import types

    fake_catalog = types.SimpleNamespace(ANNOTATOR_FILES=tuple(files))
    fake_engine = types.SimpleNamespace(download_addressed_file=download)
    monkeypatch.setattr(horde_model_reference, "annotator_catalog", fake_catalog, raising=False)
    monkeypatch.setattr(horde_model_reference, "download_engine", fake_engine, raising=False)


def test_filter_keeps_all_when_controlnet_feature_available(monkeypatch):
    """With the controlnet extra present, every catalog file is prefetched (nothing is gated away)."""
    monkeypatch.setattr("hordelib.feature_requirements.feature_available", lambda kind: True)
    files = [
        _FakeFile("a.pth", preprocessors=("HEDPreprocessor",)),
        _FakeFile("b.pth", preprocessors=("OpenposePreprocessor",)),
    ]

    kept = {entry.filename for entry in preload._annotator_files_to_prefetch(files)}

    assert kept == {"a.pth", "b.pth"}


def test_filter_keeps_all_when_no_preprocessor_is_gated(monkeypatch):
    """With no preprocessor gated behind a blocker dep, the filter branch keeps every catalog file.

    Even forcing the controlnet feature probe to report unavailable, the lean-install filter drops a file
    only when every preprocessor that loads it is gated away; the gated set is empty, so nothing is.
    """
    monkeypatch.setattr("hordelib.feature_requirements.feature_available", lambda kind: False)
    hed = _FakeFile("ControlNetHED.pth", preprocessors=("HEDPreprocessor", "FakeScribblePreprocessor"))
    openpose = _FakeFile("body_pose_model.pth", preprocessors=("OpenposePreprocessor",))

    kept = {entry.filename for entry in preload._annotator_files_to_prefetch([hed, openpose])}

    assert kept == {"ControlNetHED.pth", "body_pose_model.pth"}


def test_prefetch_places_missing_and_skips_present(monkeypatch, tmp_path):
    """A missing file is fetched to its exact custom_hf_download path; an already-present file is left alone."""
    monkeypatch.setattr("hordelib.feature_requirements.feature_available", lambda kind: True)
    calls: list[str] = []

    def _fake_download(origin_url, destination, *, sha256, gateway_base_url, apikey):
        import types as _types

        calls.append(origin_url)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"weights")
        return _types.SimpleNamespace(success=True)

    missing = _FakeFile("missing.pth", preprocessors=("HEDPreprocessor",))
    present = _FakeFile("present.pth", preprocessors=("HEDPreprocessor",))
    present_path = tmp_path / "lllyasviel" / "Annotators" / "present.pth"
    present_path.parent.mkdir(parents=True, exist_ok=True)
    present_path.write_bytes(b"already here")
    _inject_catalog_and_engine(monkeypatch, [missing, present], _fake_download)

    preload._prefetch_annotator_files(tmp_path)

    assert calls == [missing.origin_url]  # only the missing file was fetched
    landed = tmp_path / "lllyasviel" / "Annotators" / "missing.pth"
    assert landed.read_bytes() == b"weights"
    assert present_path.read_bytes() == b"already here"  # untouched


def test_prefetch_is_exception_safe(monkeypatch, tmp_path):
    """A failing download must never raise into the preload; the detector falls back to its own fetch."""
    monkeypatch.setattr("hordelib.feature_requirements.feature_available", lambda kind: True)

    def _boom(origin_url, destination, *, sha256, gateway_base_url, apikey):
        raise OSError("network exploded")

    _inject_catalog_and_engine(monkeypatch, [_FakeFile("x.pth", preprocessors=("HEDPreprocessor",))], _boom)

    preload._prefetch_annotator_files(tmp_path)  # must not raise

    assert not (tmp_path / "lllyasviel" / "Annotators" / "x.pth").exists()


def test_prefetch_orders_legacy_control_types_first():
    """Classic-type weights sort ahead of extended ones so a fresh install serves legacy jobs early."""
    files = [
        _FakeFile("sk_model.pth", control_types=("lineart",)),
        _FakeFile("ControlNetHED.pth", control_types=("hed", "fakescribbles")),
        _FakeFile("7_model.pth", control_types=("teed",)),
        _FakeFile("mlsd_large_512_fp32.pth", control_types=("mlsd",)),
    ]

    ordered = [entry.filename for entry in preload._ordered_prefetch_entries(files)]

    assert ordered == ["ControlNetHED.pth", "mlsd_large_512_fp32.pth", "sk_model.pth", "7_model.pth"]
