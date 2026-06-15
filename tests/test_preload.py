"""GPU-free regression guard for controlnet-annotator preload ordering.

The worker's ``download_models`` flow runs ``hordelib.initialise()`` and then
``SharedModelManager.preload_annotators()`` *without* ever constructing a ``HordeLib``.
Custom nodes (the ``comfyui_controlnet_aux`` package that registers ``AIO_Preprocessor``)
only register when the ``Comfy_Horde`` backend is built, so the preload routine must
construct a ``HordeLib`` itself. The rest of the suite always has a session-scoped
``HordeLib`` already built before it reaches preload, which is exactly why this gap went
unnoticed — so this test exercises the cold path with everything comfy/GPU monkeypatched
out.
"""

import huggingface_hub.constants as hf_constants
import pytest

import hordelib.comfy_horde
import hordelib.horde
import hordelib.preload as preload


def _install_fake_backend(monkeypatch, *, on_execute=None):
    """Wire up GPU-free fakes for HordeLib + node lookup, returning the construction log."""
    constructed: list[object] = []

    class _FakeNode:
        def execute(self, *args, **kwargs):
            if on_execute is not None:
                on_execute()
            return

    class _FakeHordeLib:
        CONTROLNET_IMAGE_PREPROCESSOR_MAP = {"normal": "MiDaS-NormalMapPreprocessor"}

        def __init__(self):
            constructed.append(self)

    monkeypatch.setattr(hordelib.horde, "HordeLib", _FakeHordeLib)
    monkeypatch.setattr(hordelib.comfy_horde, "get_node_class", lambda class_type: _FakeNode)
    monkeypatch.setattr(preload, "_preload_completed", False)
    # Disable the persistent skip by default so these tests exercise the run path regardless of
    # any marker on the developer's machine. The marker round-trip is covered separately below.
    monkeypatch.setattr(preload, "_pinned_annotator_ref", lambda: None)
    return constructed


def test_preload_forces_offline_when_midas_cached(monkeypatch):
    """When the MiDaS checkpoint is already cached, the preload runs with the Hub offline."""
    offline_during_execute: list[bool] = []
    _install_fake_backend(
        monkeypatch, on_execute=lambda: offline_during_execute.append(hf_constants.is_offline_mode())
    )
    monkeypatch.setattr(preload, "_midas_already_cached", lambda: True)

    was_offline_before = hf_constants.HF_HUB_OFFLINE

    assert preload.download_all_controlnet_annotators()
    assert offline_during_execute and all(offline_during_execute), "preload should run offline when MiDaS is cached"
    assert hf_constants.HF_HUB_OFFLINE == was_offline_before, "offline flag must be restored afterwards"


def test_preload_stays_online_when_midas_not_cached(monkeypatch):
    """A cold cache must keep the Hub reachable so first-time downloads can proceed."""
    offline_during_execute: list[bool] = []
    _install_fake_backend(
        monkeypatch, on_execute=lambda: offline_during_execute.append(hf_constants.is_offline_mode())
    )
    monkeypatch.setattr(preload, "_midas_already_cached", lambda: False)

    assert preload.download_all_controlnet_annotators()
    assert offline_during_execute and not any(offline_during_execute), "cold preload must stay online"


def test_preload_retries_online_when_offline_run_fails(monkeypatch):
    """If a forced-offline run fails (a checkpoint is missing), retry once with the Hub reachable."""
    runs: list[bool] = []

    def _record_and_maybe_fail():
        offline = hf_constants.is_offline_mode()
        runs.append(offline)
        if offline:
            raise OSError("checkpoint missing from cache")

    constructed = _install_fake_backend(monkeypatch, on_execute=_record_and_maybe_fail)
    monkeypatch.setattr(preload, "_midas_already_cached", lambda: True)

    assert preload.download_all_controlnet_annotators()
    assert runs == [True, False], "expected a failed offline run followed by an online retry"
    assert len(constructed) == 2, "the retry should rebuild the backend"


def test_preload_skips_when_already_verified(monkeypatch):
    """A matching marker for the pinned ref short-circuits the whole load-and-run."""
    constructed = _install_fake_backend(monkeypatch, on_execute=lambda: pytest.fail("preload ran despite marker"))
    monkeypatch.setattr(preload, "_pinned_annotator_ref", lambda: "deadbeef")
    monkeypatch.setattr(preload, "_annotators_already_verified", lambda ref: ref == "deadbeef")
    # If it wrongly tried to run, _midas_already_cached would be consulted; make that loud too.
    monkeypatch.setattr(preload, "_midas_already_cached", lambda: pytest.fail("preload ran despite marker"))

    assert preload.download_all_controlnet_annotators()
    assert not constructed, "a valid marker must skip constructing the backend entirely"


def test_preload_records_marker_and_skips_next_time(monkeypatch, tmp_path):
    """A successful run writes the marker keyed to the pinned ref; the next run then skips."""
    runs: list[int] = []
    monkeypatch.setenv("AUX_ANNOTATOR_CKPTS_PATH", str(tmp_path))
    monkeypatch.setattr(preload, "_midas_already_cached", lambda: False)
    _install_fake_backend(monkeypatch, on_execute=lambda: runs.append(1))
    # Override the helper's default (which disables persistence) so the marker is exercised.
    monkeypatch.setattr(preload, "_pinned_annotator_ref", lambda: "ref-abc123")

    assert preload.download_all_controlnet_annotators()
    marker = tmp_path / preload._PRELOAD_MARKER_NAME
    assert marker.is_file() and marker.read_text(encoding="utf-8").strip() == "ref-abc123"
    assert runs == [1], "first call should run the preload once"

    # A fresh process (reset in-process guard) with the marker present must skip.
    monkeypatch.setattr(preload, "_preload_completed", False)
    assert preload.download_all_controlnet_annotators()
    assert runs == [1], "second call must skip the run thanks to the marker"


def test_preload_constructs_hordelib_before_node_lookup(monkeypatch):
    """``download_all_controlnet_annotators`` must build a HordeLib before looking up nodes."""
    constructed: list[object] = []

    class _FakeNode:
        def execute(self, *args, **kwargs):
            return None

    class _FakeHordeLib:
        CONTROLNET_IMAGE_PREPROCESSOR_MAP = {"canny": "CannyEdgePreprocessor"}

        def __init__(self):
            constructed.append(self)

    def _fake_get_node_class(class_type: str) -> type:
        assert constructed, (
            "preload looked up node class "
            f"{class_type!r} before constructing HordeLib; AIO_Preprocessor would be "
            "unregistered (the worker download_models regression)"
        )
        return _FakeNode

    monkeypatch.setattr(hordelib.horde, "HordeLib", _FakeHordeLib)
    monkeypatch.setattr(hordelib.comfy_horde, "get_node_class", _fake_get_node_class)
    monkeypatch.setattr(preload, "_preload_completed", False)
    # Disable the persistent marker skip and pin the offline decision so the cold run path is
    # exercised regardless of an earlier real preload (which sets AUX_ANNOTATOR_CKPTS_PATH and
    # writes the marker) having run in the same session.
    monkeypatch.setattr(preload, "_pinned_annotator_ref", lambda: None)
    monkeypatch.setattr(preload, "_midas_already_cached", lambda: False)

    assert preload.download_all_controlnet_annotators()
    assert constructed, "preload did not construct a HordeLib instance"
