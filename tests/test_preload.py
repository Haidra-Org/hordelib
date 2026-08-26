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

import os

import huggingface_hub.constants as hf_constants
import pytest

import hordelib.comfy_horde
import hordelib.horde
import hordelib.preload as preload


def _install_fake_backend(monkeypatch, *, on_execute=None, preprocessors=("MiDaS-NormalMapPreprocessor",)):
    """Wire up GPU-free fakes for HordeLib + node lookup, returning the construction log.

    ``on_execute`` receives the preprocessor name being run.
    """
    constructed: list[object] = []

    class _FakeNode:
        def execute(self, preprocessor, *args, **kwargs):
            if on_execute is not None:
                on_execute(preprocessor)
            return

    class _FakeHordeLib:
        CONTROLNET_IMAGE_PREPROCESSOR_MAP = {name: name for name in preprocessors}

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
        monkeypatch, on_execute=lambda _name: offline_during_execute.append(hf_constants.is_offline_mode())
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
        monkeypatch, on_execute=lambda _name: offline_during_execute.append(hf_constants.is_offline_mode())
    )
    monkeypatch.setattr(preload, "_midas_already_cached", lambda: False)

    assert preload.download_all_controlnet_annotators()
    assert offline_during_execute and not any(offline_during_execute), "cold preload must stay online"


def test_preload_retries_online_when_offline_run_fails(monkeypatch):
    """A preprocessor that fails offline (its files are not cached) is retried alone with the Hub reachable."""
    runs: list[bool] = []

    def _record_and_maybe_fail(_name):
        offline = hf_constants.is_offline_mode()
        runs.append(offline)
        if offline:
            raise OSError("checkpoint missing from cache")

    constructed = _install_fake_backend(monkeypatch, on_execute=_record_and_maybe_fail)
    monkeypatch.setattr(preload, "_midas_already_cached", lambda: True)

    assert preload.download_all_controlnet_annotators()
    assert runs == [True, False], "expected a failed offline attempt followed by an online retry"
    assert len(constructed) == 1, "the retry runs inside the same pass; the backend is not rebuilt"


def test_preload_online_retry_covers_only_the_failed_preprocessor(monkeypatch):
    """Verified preprocessors are not re-run when a later one needs the Hub."""
    runs: list[tuple[str, bool]] = []

    def _record_and_fail_second_offline(name):
        offline = hf_constants.is_offline_mode()
        runs.append((name, offline))
        if offline and name == "second":
            raise OSError("checkpoint missing from cache")

    _install_fake_backend(
        monkeypatch,
        on_execute=_record_and_fail_second_offline,
        preprocessors=("first", "second", "third"),
    )
    monkeypatch.setattr(preload, "_midas_already_cached", lambda: True)
    was_offline_before = hf_constants.HF_HUB_OFFLINE

    assert preload.download_all_controlnet_annotators()
    assert runs == [("first", True), ("second", True), ("second", False), ("third", True)]
    assert hf_constants.HF_HUB_OFFLINE == was_offline_before, "offline flag must be restored afterwards"


def test_preload_online_failure_fails_the_pass(monkeypatch):
    """A preprocessor that also fails with the Hub reachable fails the verify; nothing is retried twice."""
    runs: list[bool] = []

    def _always_fail(_name):
        runs.append(hf_constants.is_offline_mode())
        raise OSError("detector cannot load")

    _install_fake_backend(monkeypatch, on_execute=_always_fail)
    monkeypatch.setattr(preload, "_midas_already_cached", lambda: True)

    assert not preload.download_all_controlnet_annotators()
    assert runs == [True, False]


def test_preload_skips_when_already_verified(monkeypatch):
    """A matching marker for the pinned ref short-circuits the whole load-and-run."""
    constructed = _install_fake_backend(monkeypatch, on_execute=lambda: pytest.fail("preload ran despite marker"))
    monkeypatch.setattr(preload, "_pinned_annotator_ref", lambda: "deadbeef")
    monkeypatch.setattr(preload, "_annotators_already_verified", lambda ref: ref == "deadbeef")
    # If it wrongly tried to run, _midas_already_cached would be consulted; make that loud too.
    monkeypatch.setattr(preload, "_midas_already_cached", lambda: pytest.fail("preload ran despite marker"))

    assert preload.download_all_controlnet_annotators()
    assert not constructed, "a valid marker must skip constructing the backend entirely"


def test_annotators_present_true_when_marker_matches(monkeypatch, tmp_path):
    """``controlnet_annotators_present`` reads the on-disk marker the same way the preload skip does."""
    monkeypatch.setenv("AUX_ANNOTATOR_CKPTS_PATH", str(tmp_path))
    monkeypatch.setattr(preload, "_pinned_annotator_ref", lambda: "ref-xyz")
    (tmp_path / preload._PRELOAD_MARKER_NAME).write_text(preload._marker_key("ref-xyz") + "\n", encoding="utf-8")

    assert preload.controlnet_annotators_present() is True


def test_annotators_present_false_when_marker_absent_or_stale(monkeypatch, tmp_path):
    """A missing marker (or one keyed to a different pin) reads as a pending download, not unknown."""
    monkeypatch.setenv("AUX_ANNOTATOR_CKPTS_PATH", str(tmp_path))
    monkeypatch.setattr(preload, "_pinned_annotator_ref", lambda: "ref-new")

    assert preload.controlnet_annotators_present() is False

    (tmp_path / preload._PRELOAD_MARKER_NAME).write_text(preload._marker_key("ref-old") + "\n", encoding="utf-8")
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


def test_marker_is_stale_when_the_hub_cache_moves(monkeypatch, tmp_path):
    """A marker written against one hub cache does not vouch for another location."""
    monkeypatch.setenv("AUX_ANNOTATOR_CKPTS_PATH", str(tmp_path))
    monkeypatch.setattr(preload, "_pinned_annotator_ref", lambda: "ref-xyz")
    monkeypatch.setattr(preload, "hub_cache_dir", lambda: str(tmp_path / "hub-a"))
    preload._record_annotators_verified("ref-xyz")
    assert preload._annotators_already_verified("ref-xyz")

    monkeypatch.setattr(preload, "hub_cache_dir", lambda: str(tmp_path / "hub-b"))

    assert not preload._annotators_already_verified("ref-xyz")


def test_marker_from_before_the_location_key_is_stale(monkeypatch, tmp_path):
    """A marker holding only the pin (the earlier format) re-verifies once."""
    monkeypatch.setenv("AUX_ANNOTATOR_CKPTS_PATH", str(tmp_path))
    monkeypatch.setattr(preload, "_pinned_annotator_ref", lambda: "ref-xyz")
    (tmp_path / preload._PRELOAD_MARKER_NAME).write_text("ref-xyz\n", encoding="utf-8")

    assert not preload._annotators_already_verified("ref-xyz")


def test_pin_only_marker_predates_the_location_key(monkeypatch, tmp_path):
    """A marker holding only the pin is recognised as pre-location; the current format and no marker are not."""
    monkeypatch.setenv("AUX_ANNOTATOR_CKPTS_PATH", str(tmp_path))
    assert not preload.annotator_verify_marker_predates_location_key()

    (tmp_path / preload._PRELOAD_MARKER_NAME).write_text("ref-xyz\n", encoding="utf-8")
    assert preload.annotator_verify_marker_predates_location_key()

    monkeypatch.setattr(preload, "hub_cache_dir", lambda: str(tmp_path / "hub"))
    preload._record_annotators_verified("ref-xyz")
    assert not preload.annotator_verify_marker_predates_location_key()


def test_marker_reports_the_hub_cache_it_was_written_against(monkeypatch, tmp_path):
    """A current-format marker names its hub cache; a pin-only or missing marker names none."""
    monkeypatch.setenv("AUX_ANNOTATOR_CKPTS_PATH", str(tmp_path))
    assert preload.annotator_verify_marker_hub_cache_dir() is None

    (tmp_path / preload._PRELOAD_MARKER_NAME).write_text("ref-xyz\n", encoding="utf-8")
    assert preload.annotator_verify_marker_hub_cache_dir() is None

    monkeypatch.setattr(preload, "hub_cache_dir", lambda: str(tmp_path / "hub-a"))
    preload._record_annotators_verified("ref-xyz")
    assert preload.annotator_verify_marker_hub_cache_dir() == str(tmp_path / "hub-a")


class TestHuggingFaceCacheIsolation:
    """The hub cache must resolve under ``AIWORKER_CACHE_HOME`` so every consumer of that root shares it."""

    def test_noop_without_a_cache_root(self, monkeypatch):
        """With no cache root declared there is nothing to isolate to, so the hub keeps its own defaults."""
        monkeypatch.delenv("AIWORKER_CACHE_HOME", raising=False)
        monkeypatch.setenv("HF_HOME", "/somewhere/ambient")
        before = dict(os.environ)
        preload.apply_huggingface_cache_isolation()
        assert os.environ == before

    def test_points_the_hub_at_the_cache_root(self, monkeypatch, tmp_path):
        monkeypatch.setenv("AIWORKER_CACHE_HOME", str(tmp_path))
        monkeypatch.delenv("HF_HOME", raising=False)
        preload.apply_huggingface_cache_isolation()
        assert os.environ["HF_HOME"] == str(tmp_path / preload.HUGGINGFACE_HOME_DIRNAME)

    def test_ambient_hub_cache_variables_are_replaced(self, monkeypatch, tmp_path):
        """``HF_HUB_CACHE``/``HUGGINGFACE_HUB_CACHE`` outrank ``HF_HOME``, so setting it alone is not enough."""
        monkeypatch.setenv("AIWORKER_CACHE_HOME", str(tmp_path))
        monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "elsewhere" / "hub"))
        monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(tmp_path / "elsewhere" / "hub"))
        preload.apply_huggingface_cache_isolation()
        assert "HF_HUB_CACHE" not in os.environ
        assert "HUGGINGFACE_HUB_CACHE" not in os.environ
        assert os.environ["HF_HOME"] == str(tmp_path / preload.HUGGINGFACE_HOME_DIRNAME)

    def test_records_the_ambient_locations_it_displaced(self, monkeypatch, tmp_path):
        """The displaced directories are the only record of where a pre-isolation install left its entries."""
        monkeypatch.setenv("AIWORKER_CACHE_HOME", str(tmp_path))
        monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "ambient" / "hub"))
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
        monkeypatch.delenv("HF_HOME", raising=False)
        monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)
        preload.apply_huggingface_cache_isolation()
        recorded = os.environ[preload.LEGACY_HUB_CACHES_ENV_VAR].split(os.pathsep)
        assert str(tmp_path / "ambient" / "hub") in recorded
        assert str(tmp_path / "xdg" / "huggingface" / "hub") in recorded

    def test_repeat_application_never_records_its_own_target(self, monkeypatch, tmp_path):
        """A second call must not read the isolated location as a legacy one and copy entries onto themselves."""
        monkeypatch.setenv("AIWORKER_CACHE_HOME", str(tmp_path))
        monkeypatch.delenv("HF_HOME", raising=False)
        preload.apply_huggingface_cache_isolation()
        preload.apply_huggingface_cache_isolation()
        target_hub = os.path.join(str(tmp_path / preload.HUGGINGFACE_HOME_DIRNAME), "hub")
        recorded = [entry for entry in os.environ[preload.LEGACY_HUB_CACHES_ENV_VAR].split(os.pathsep) if entry]
        assert os.environ["HF_HOME"] == str(tmp_path / preload.HUGGINGFACE_HOME_DIRNAME)
        assert target_hub not in recorded

    def test_the_marker_key_follows_the_isolated_location(self, monkeypatch, tmp_path):
        """The verify marker is what two processes disagree over when their hub caches differ."""
        monkeypatch.setenv("AUX_ANNOTATOR_CKPTS_PATH", str(tmp_path / "annotators"))
        (tmp_path / "annotators").mkdir()
        monkeypatch.setattr(preload, "hub_cache_dir", lambda: str(tmp_path / "isolated" / "hub"))
        preload._record_annotators_verified("ref-xyz")
        assert preload._annotators_already_verified("ref-xyz")

        monkeypatch.setattr(preload, "hub_cache_dir", lambda: str(tmp_path / "ambient" / "hub"))
        assert not preload._annotators_already_verified("ref-xyz")
