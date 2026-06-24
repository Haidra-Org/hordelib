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


def test_annotators_present_true_when_marker_matches(monkeypatch, tmp_path):
    """``controlnet_annotators_present`` reads the on-disk marker the same way the preload skip does."""
    monkeypatch.setenv("AUX_ANNOTATOR_CKPTS_PATH", str(tmp_path))
    monkeypatch.setattr(preload, "_pinned_annotator_ref", lambda: "ref-xyz")
    (tmp_path / preload._PRELOAD_MARKER_NAME).write_text("ref-xyz\n", encoding="utf-8")

    assert preload.controlnet_annotators_present() is True


def test_annotators_present_false_when_marker_absent_or_stale(monkeypatch, tmp_path):
    """A missing marker (or one keyed to a different pin) reads as a pending download, not unknown."""
    monkeypatch.setenv("AUX_ANNOTATOR_CKPTS_PATH", str(tmp_path))
    monkeypatch.setattr(preload, "_pinned_annotator_ref", lambda: "ref-new")

    assert preload.controlnet_annotators_present() is False

    (tmp_path / preload._PRELOAD_MARKER_NAME).write_text("ref-old\n", encoding="utf-8")
    assert preload.controlnet_annotators_present() is False


def test_annotators_present_unknown_when_ref_undeterminable(monkeypatch, tmp_path):
    """An unreadable pinned ref yields None (unknown) so callers do not claim a false "missing"."""
    monkeypatch.setenv("AUX_ANNOTATOR_CKPTS_PATH", str(tmp_path))
    monkeypatch.setattr(preload, "_pinned_annotator_ref", lambda: None)

    assert preload.controlnet_annotators_present() is None


def _make_annotator_files(ckpts_dir, control_types):
    """Create the flat ``<repo>/<subfolder>/<filename>`` checkpoints a set of control types needs."""
    from horde_model_reference import annotator_catalog

    wanted = set(control_types)
    for entry in annotator_catalog.ANNOTATOR_FILES:
        if wanted.intersection(entry.control_types):
            destination = ckpts_dir / entry.relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(b"x")


@pytest.fixture
def _no_hub_cache(monkeypatch):
    """Force the HuggingFace hub-cache lookup to miss, so resolution depends only on on-disk files."""
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", lambda **kwargs: None)


def test_annotators_resolvable_true_when_files_present(monkeypatch, tmp_path):
    """Files on disk in the flat layout read resolvable, regardless of the preload marker."""
    monkeypatch.setenv("AUX_ANNOTATOR_CKPTS_PATH", str(tmp_path))
    _make_annotator_files(tmp_path, ["depth"])
    assert preload.annotators_resolvable(["depth"]) is True


def test_annotators_resolvable_false_when_absent(monkeypatch, tmp_path, _no_hub_cache):
    """A control type whose files are not on disk (and not hub-cached) reads as not resolvable."""
    monkeypatch.setenv("AUX_ANNOTATOR_CKPTS_PATH", str(tmp_path))
    assert preload.annotators_resolvable(["openpose"]) is False


def test_annotators_resolvable_false_when_partial(monkeypatch, tmp_path, _no_hub_cache):
    """Resolution requires every needed file: a partially-present selection is not resolvable."""
    monkeypatch.setenv("AUX_ANNOTATOR_CKPTS_PATH", str(tmp_path))
    _make_annotator_files(tmp_path, ["depth"])
    assert preload.annotators_resolvable(["depth", "openpose"]) is False


def test_annotators_resolvable_vacuous_for_weightless_unknown_empty(monkeypatch, tmp_path):
    """Weightless (canny), unknown, and empty selections need no files and are vacuously resolvable."""
    monkeypatch.setenv("AUX_ANNOTATOR_CKPTS_PATH", str(tmp_path))
    assert preload.annotators_resolvable(["canny"]) is True
    assert preload.annotators_resolvable(["definitely-not-a-control-type"]) is True
    assert preload.annotators_resolvable([]) is True


def test_annotators_resolvable_unknown_when_ckpts_dir_undeterminable(monkeypatch):
    """When the checkpoints directory cannot be derived, presence is unknown (None), never a false missing."""
    monkeypatch.delenv("AUX_ANNOTATOR_CKPTS_PATH", raising=False)
    monkeypatch.setattr(preload, "_annotator_ckpts_dir", lambda: None)
    assert preload.annotators_resolvable(["depth"]) is None


def test_annotators_resolvable_independent_of_marker(monkeypatch, tmp_path):
    """The original bug: present files must read resolvable even when the pin-keyed marker is absent."""
    monkeypatch.setenv("AUX_ANNOTATOR_CKPTS_PATH", str(tmp_path))
    monkeypatch.setattr(preload, "_pinned_annotator_ref", lambda: "ref-never-written")
    _make_annotator_files(tmp_path, ["depth"])
    assert preload.controlnet_annotators_present() is False  # marker absent -> the stale "missing"
    assert preload.annotators_resolvable(["depth"]) is True  # but the files are genuinely there


def test_preload_import_is_torch_free():
    """Importing ``hordelib.preload`` (the torch-free presence surface) must not drag torch in."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-c", "import hordelib.preload, sys; assert 'torch' not in sys.modules"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


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
