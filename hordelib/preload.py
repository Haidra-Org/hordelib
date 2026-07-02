"""Preloading of auxiliary models (controlnet annotators).

ControlNet preprocessing is provided by the ``comfyui_controlnet_aux`` custom node package
(pinned via ``hordelib/installation/manifest.json``). Its detectors download their checkpoint
files from the HuggingFace hub on first use, into the directory named by the
``AUX_ANNOTATOR_CKPTS_PATH`` environment variable (set during ``hordelib.initialise()``).

Preloading exercises each supported preprocessor once on a tiny image, which both triggers the
downloads and verifies that each detector actually runs ahead of any real generation. The
detectors keep nothing resident (each node wrapper loads its weights, runs, then ``del``s the
model), so the run is purely a download-and-verify step: it provides no warmup for later jobs.

Two optimisations keep this cheap after the first time:

* Once everything has been downloaded and verified for the pinned ``comfyui_controlnet_aux``
  commit, a marker is written into ``AUX_ANNOTATOR_CKPTS_PATH``. Subsequent processes (e.g. every
  worker restart) see the marker and skip the whole load-and-run entirely, deferring all annotator
  loading to first real use. A pin bump re-verifies once; a deleted checkpoint is simply
  re-downloaded lazily at first use.
* When a run is needed but the MiDaS checkpoint is already cached, the run is forced offline. MiDaS
  (the ``normal`` control type) is the one detector that loads via HuggingFace ``transformers``
  rather than the existence-guarded ``custom_hf_download``; its repo ships only a
  ``pytorch_model.bin``, so with the Hub reachable ``transformers`` issues a safetensors
  auto-conversion lookup (several API calls) on every load even when nothing needs downloading.
"""

import contextlib
import os
import threading
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Protocol

from loguru import logger


class _AnnotatorFileLike(Protocol):
    """The shape the prefetch reads from a ``horde_model_reference.annotator_catalog.AnnotatorFile``.

    Described structurally so this module type-checks even against an older installed ``horde_model_reference``
    that does not yet ship the catalog (the concrete type is imported lazily and may be absent; see
    :func:`_prefetch_annotator_files`).
    """

    repo: str
    subfolder: str
    filename: str
    sha256: str | None
    preprocessors: tuple[str, ...]

    @property
    def relative_path(self) -> str:
        """The file's path relative to the annotator checkpoints directory (``<repo>/<subfolder>/<filename>``)."""
        ...

    @property
    def origin_url(self) -> str:
        """The HuggingFace origin URL the file is fetched from when the gated mirror does not serve it."""
        ...


_preload_mutex = threading.Lock()
_preload_completed = False

# The only preprocessor whose model is loaded through ``transformers`` rather than the
# existence-guarded ``custom_hf_download``. See module docstring for why this matters.
_MIDAS_REPO_ID = "Intel/dpt-hybrid-midas"
_MIDAS_REQUIRED_FILES = ("config.json", "preprocessor_config.json", "pytorch_model.bin")

_ANNOTATOR_NODE_NAME = "comfyui_controlnet_aux"
# Records, inside the annotator checkpoint directory, that a full preload completed for a given
# pinned annotator commit so later processes can skip it. See module docstring.
_PRELOAD_MARKER_NAME = ".hordelib_preload_complete"

# comfyui_controlnet_aux preprocessors that cannot import without an optional `controlnet`-extra
# package, so they must be dropped from the preload on a lean base install (their node_wrapper is
# guarded and simply does not register). Openpose's DWPose detector is the only horde-exposed
# preprocessor that needs onnxruntime; running it without the extra would abort the whole preload.
_ONNXRUNTIME_GATED_PREPROCESSORS = frozenset({"OpenposePreprocessor"})


def _annotator_files_to_prefetch(all_files: Iterable[_AnnotatorFileLike]) -> list[_AnnotatorFileLike]:
    """Filter the annotator catalog to the files whose preprocessor will actually run on this install.

    On a lean base install (no ``controlnet`` extra) the onnxruntime-gated preprocessors never register and the
    preload skips them (see :func:`_run_preload`), so prefetching their large weights would be wasted. A file is
    kept when at least one preprocessor that loads it is not gated away. Best-effort: if the feature check itself
    cannot be evaluated, everything is kept (the worst case is a redundant download, never a missing one).
    """
    try:
        from hordelib.feature_impact import FEATURE_KIND
        from hordelib.feature_requirements import feature_available

        if feature_available(FEATURE_KIND.controlnet):
            return list(all_files)
    except Exception as e:
        logger.debug("Could not evaluate controlnet feature for annotator prefetch; keeping all: error={}", e)
        return list(all_files)

    return [
        entry
        for entry in all_files
        if any(preprocessor not in _ONNXRUNTIME_GATED_PREPROCESSORS for preprocessor in entry.preprocessors)
    ]


def _prefetch_annotator_files(ckpts_dir: Path) -> None:
    """Pre-place known annotator checkpoints via the unified gated-mirror-then-origin download engine.

    For each catalog file not already on disk, fetch it to the exact path ``custom_hf_download`` expects, so the
    detector finds it present and skips its own HuggingFace download. The gated R2 mirror (when configured and
    the file's hash is known) is tried first and falls back to the HuggingFace origin on any failure, so this
    only ever adds a faster, free path; it never blocks a download. Best-effort throughout: any failure simply
    leaves the file for the detector to fetch itself, exactly as before this hook existed.
    """
    try:
        from horde_model_reference import annotator_catalog, download_engine
    except Exception as e:
        # An older installed horde_model_reference without the catalog (or its record-free fetch helper): the
        # detectors download from HuggingFace themselves, exactly as before. Degrade quietly.
        logger.debug("Annotator catalog unavailable (older horde_model_reference?); skipping prefetch: {}", e)
        return

    gateway_base_url: str | None = None
    apikey: str | None = None
    try:
        from hordelib.model_manager.base import _resolve_horde_api_key, _resolve_r2_gateway_url

        gateway_base_url = _resolve_r2_gateway_url()
        apikey = _resolve_horde_api_key()
    except Exception as e:
        logger.debug("Could not resolve gated-mirror settings; annotator prefetch uses origin only: error={}", e)

    for entry in _annotator_files_to_prefetch(annotator_catalog.ANNOTATOR_FILES):
        destination = ckpts_dir / Path(entry.relative_path)
        if destination.is_file():
            continue
        try:
            outcome = download_engine.download_addressed_file(
                entry.origin_url,
                destination,
                sha256=entry.sha256,
                gateway_base_url=gateway_base_url,
                apikey=apikey,
            )
            if not outcome.success:
                logger.info(
                    "Annotator prefetch did not complete; the detector will fetch it at first use: file={}",
                    entry.filename,
                )
        except Exception as e:
            logger.warning(
                "Annotator prefetch errored; the detector will fetch it at first use: file={} error={}",
                entry.filename,
                e,
            )


def _midas_already_cached() -> bool:
    """Return whether the transformers-based MiDaS annotator is fully in the HF cache.

    When it is, the entire preload can run with the Hub forced offline, which suppresses
    transformers' per-load safetensors auto-conversion lookup against ``Intel/dpt-hybrid-midas``
    without affecting the other annotators: those are guarded by ``custom_hf_download`` and never
    touch the network once their checkpoints exist on disk.
    """
    try:
        from huggingface_hub import try_to_load_from_cache
    except Exception:
        return False

    return all(
        isinstance(try_to_load_from_cache(repo_id=_MIDAS_REPO_ID, filename=filename), str)
        for filename in _MIDAS_REQUIRED_FILES
    )


def _pinned_annotator_ref() -> str | None:
    """Return the pinned ``comfyui_controlnet_aux`` commit, or None if it can't be determined.

    The marker is keyed to this commit so a manifest bump (new annotator version) re-verifies
    once. If the ref can't be read, the persistent skip is disabled and the preload runs normally.
    """
    try:
        from hordelib.installation import load_packaged_manifest

        for node in load_packaged_manifest().custom_nodes:
            if node.name == _ANNOTATOR_NODE_NAME:
                return node.ref
    except Exception as e:
        logger.warning("Could not read pinned annotator ref; preload will not be cached: error={}", e)
    return None


def _annotator_ckpts_dir() -> Path | None:
    """Return the directory annotator checkpoints live in, or None if it cannot be determined.

    ``hordelib.initialise()`` exports ``AUX_ANNOTATOR_CKPTS_PATH`` and is the authority once a run is
    underway. When it is unset (e.g. a no-boot caller that only wants to *check* presence) the same value
    is derived from :class:`~hordelib.settings.UserSettings`, which needs neither ComfyUI nor a GPU, so
    presence can be read before (or without) initialisation.
    """
    ckpts_dir = os.environ.get("AUX_ANNOTATOR_CKPTS_PATH")
    if ckpts_dir:
        return Path(ckpts_dir)
    try:
        from hordelib.settings import UserSettings

        return UserSettings.get_model_directory() / "controlnet" / "annotators"
    except Exception as e:
        logger.debug("Could not derive the annotator checkpoint directory: error={}", e)
        return None


def _preload_marker_path() -> Path | None:
    """Return the on-disk marker path, or None if the annotator directory is unknown."""
    ckpts_dir = _annotator_ckpts_dir()
    if ckpts_dir is None:
        return None
    return ckpts_dir / _PRELOAD_MARKER_NAME


def _annotators_already_verified(ref: str) -> bool:
    """Whether a previous process already downloaded and verified annotators for this ``ref``."""
    marker = _preload_marker_path()
    if marker is None:
        return False
    try:
        return marker.is_file() and marker.read_text(encoding="utf-8").strip() == ref
    except OSError:
        return False


def controlnet_annotators_present() -> bool | None:
    """Whether the controlnet annotators are already downloaded and verified on this machine.

    Reads the on-disk preload marker (keyed to the pinned ``comfyui_controlnet_aux`` commit) exactly the
    way :func:`download_all_controlnet_annotators` reads it for its fast-path skip, so a caller can decide
    whether a (slow, one-time) annotator download is still pending *before* paying for it. Import-safe:
    needs neither :func:`hordelib.initialise` nor a GPU, because the directory is derived from
    :class:`~hordelib.settings.UserSettings` when the runtime env var is unset.

    Returns:
        ``True`` when the marker matches the pinned ref (a full set was downloaded and verified),
        ``False`` when it is absent or stale (a download is still pending), or ``None`` when presence
        cannot be determined (the pinned ref or the annotator directory is unknown).
    """
    try:
        ref = _pinned_annotator_ref()
        if ref is None:
            return None
        if _preload_marker_path() is None:
            return None
        return _annotators_already_verified(ref)
    except Exception as e:  # pragma: no cover - presence is best-effort; never raise into a caller
        logger.debug("Could not determine controlnet annotator presence: error={}", e)
        return None


def _annotator_file_resolvable(entry: _AnnotatorFileLike, ckpts_dir: Path | None) -> bool:
    """Whether one annotator file is already where the engine will find it at first use (no download needed).

    Existence-based, unlike the preload marker: a detector loads a file it finds on disk and skips its own
    HuggingFace fetch, so a file present in the checkpoints dir (the flat ``<repo>/<subfolder>/<filename>``
    layout ``custom_hf_download`` writes) -- or already in the HuggingFace hub cache, which a fetch resolves
    without touching the network -- will not be re-downloaded. The cheap on-disk path is checked first.
    """
    if ckpts_dir is not None and (ckpts_dir / Path(entry.relative_path)).is_file():
        return True
    try:
        from huggingface_hub import try_to_load_from_cache

        in_repo = "/".join(part for part in (entry.subfolder, entry.filename) if part)
        if isinstance(try_to_load_from_cache(repo_id=entry.repo, filename=in_repo), str):
            return True
        if ckpts_dir is not None and isinstance(
            try_to_load_from_cache(repo_id=entry.repo, filename=in_repo, cache_dir=str(ckpts_dir)),
            str,
        ):
            return True
    except Exception as e:  # pragma: no cover - a hub-cache probe failure just means "not via the cache"
        logger.debug("Hub-cache annotator lookup failed (treating this file as absent): error={}", e)
    return False


def annotators_resolvable(control_types: Iterable[str]) -> bool | None:
    """Whether the controlnet annotators for *control_types* are on disk or cached (existence, not the marker).

    The authoritative answer to "will a controlnet job for these control types need a slow annotator download
    before it can run", as distinct from :func:`controlnet_annotators_present`, which reads the preload
    *marker* (a stricter, pin-keyed fast-path that is ``False`` whenever a full verify has not run for the
    current pin, even when every file is already present on disk). A surface deciding whether to prompt a
    pre-download should use this; the marker stays only the preload's own skip optimisation.

    Weightless control types (``canny``, ``scribble``) and types unknown to the catalog need no files and are
    vacuously resolvable. Returns ``None`` only when the file catalog or the checkpoint directory cannot be
    determined, so a caller treats the unknown as "do not claim missing" rather than nagging spuriously.
    """
    try:
        from horde_model_reference import annotator_catalog
    except Exception as e:
        logger.debug("Annotator catalog unavailable; cannot resolve annotator presence: error={}", e)
        return None
    # Prefer the catalog's own control-type selector; fall back to filtering the file list directly so an
    # older horde_model_reference that ships ANNOTATOR_FILES but not the helper still resolves correctly.
    selector = getattr(annotator_catalog, "annotators_for_control_types", None)
    if selector is not None:
        needed = selector(control_types)
    else:
        wanted = set(control_types)
        needed = tuple(
            entry for entry in annotator_catalog.ANNOTATOR_FILES if wanted.intersection(entry.control_types)
        )
    if not needed:
        return True
    ckpts_dir = _annotator_ckpts_dir()
    if ckpts_dir is None:
        return None
    return all(_annotator_file_resolvable(entry, ckpts_dir) for entry in needed)


def _record_annotators_verified(ref: str) -> None:
    """Persist that annotators for ``ref`` are downloaded and verified on this machine.

    Best-effort: a failure here just means the next process re-verifies, so it must not fail
    the preload itself.
    """
    marker = _preload_marker_path()
    if marker is None:
        return
    try:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(ref + "\n", encoding="utf-8")
    except OSError as e:
        logger.warning("Could not write annotator preload marker: error={}", e)


@contextlib.contextmanager
def _hub_offline(enabled: bool) -> Iterator[None]:
    """Force ``huggingface_hub``/``transformers`` into offline mode for the duration if enabled.

    The offline flag is captured from the environment into ``huggingface_hub.constants.HF_HUB_OFFLINE``
    at import time, but consulted dynamically via ``is_offline_mode()`` on every request. Since the
    hub is already imported by the time preload runs, toggling the live module attribute (not just
    the env var, which has already been read) is what actually takes effect.
    """
    if not enabled:
        yield
        return

    import huggingface_hub.constants as hf_constants

    previous_attr = hf_constants.HF_HUB_OFFLINE
    previous_env = os.environ.get("HF_HUB_OFFLINE")
    hf_constants.HF_HUB_OFFLINE = True
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        yield
    finally:
        hf_constants.HF_HUB_OFFLINE = previous_attr
        if previous_env is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = previous_env


def _run_preload(*, force_offline: bool) -> bool:
    """Construct the backend and exercise every supported preprocessor once.

    Returns:
        bool: True if all annotators are available and runnable, False otherwise.
    """
    try:
        import torch

        from hordelib.comfy_horde import get_node_class
        from hordelib.horde import HordeLib

        # Constructing the HordeLib singleton starts the ComfyUI backend, which loads
        # the custom nodes that register AIO_Preprocessor. Idempotent: an existing
        # instance is reused. Without this, a caller that only ran initialise() (e.g.
        # the worker's download_models flow) would find AIO_Preprocessor unregistered.
        HordeLib()

        aio_preprocessor_class = get_node_class("AIO_Preprocessor")

        preprocessors = sorted(set(HordeLib.CONTROLNET_IMAGE_PREPROCESSOR_MAP.values()))
        # On a lean base install (no `controlnet` extra) the onnxruntime-backed detectors never
        # registered; preloading them would raise and abort the whole run. Drop them and preload the
        # rest, which are pure-torch / transformers and work without the extra.
        from hordelib.feature_impact import FEATURE_KIND
        from hordelib.feature_requirements import feature_available

        if not feature_available(FEATURE_KIND.controlnet):
            skipped = sorted(_ONNXRUNTIME_GATED_PREPROCESSORS.intersection(preprocessors))
            if skipped:
                logger.info(
                    "Skipping controlnet annotators that need the 'controlnet' extra (onnxruntime "
                    "absent): preprocessors={}",
                    skipped,
                )
                preprocessors = [p for p in preprocessors if p not in _ONNXRUNTIME_GATED_PREPROCESSORS]

        # A tiny gray test card; enough for every detector to run its model once
        test_image = torch.full((1, 64, 64, 3), 0.5)

        with _hub_offline(force_offline):
            for i, preprocessor in enumerate(preprocessors):
                logger.info(
                    "Preloading controlnet annotator",
                    preprocessor=preprocessor,
                    current=i + 1,
                    total=len(preprocessors),
                )
                aio_preprocessor_class().execute(preprocessor, test_image, resolution=64)

        return True
    except Exception as e:
        logger.exception("Failed to preload controlnet annotators: error={}", e)
        return False


def download_all_controlnet_annotators() -> bool:
    """Download (and verify by running) all controlnet annotators hordelib supports.

    Requires ``hordelib.initialise()`` to have completed. A ``HordeLib`` instance is
    constructed here if one does not already exist, since custom nodes (the
    ``comfyui_controlnet_aux`` package that registers ``AIO_Preprocessor``) are only
    loaded when the ``Comfy_Horde`` backend is built; ``initialise()`` alone does not
    register them.

    If a previous process already downloaded and verified the annotators for the pinned
    annotator version, this returns immediately and leaves the actual loading to first use.

    Returns:
        bool: True if all annotators are available and runnable, False otherwise.
    """
    global _preload_completed
    with _preload_mutex:
        if _preload_completed:
            return True

        ref = _pinned_annotator_ref()
        if ref is not None and _annotators_already_verified(ref):
            logger.debug(
                "Controlnet annotators already downloaded and verified (ref={}); skipping preload "
                "and deferring annotator loading to first use",
                ref[:8],
            )
            _preload_completed = True
            return True

        # Pre-place the annotator checkpoints through the unified (gated R2 -> HuggingFace) download engine
        # *before* the detectors run, so custom_hf_download finds them present and skips its own HF fetch. Runs
        # with the Hub online (the offline window is inside _run_preload). Best-effort; never blocks the preload.
        ckpts_dir = _annotator_ckpts_dir()
        if ckpts_dir is not None:
            _prefetch_annotator_files(ckpts_dir)

        force_offline = _midas_already_cached()
        if force_offline:
            logger.debug(
                "MiDaS annotator already cached; running preload with the HuggingFace Hub forced "
                "offline to avoid redundant safetensors auto-conversion lookups",
            )

        succeeded = _run_preload(force_offline=force_offline)
        if not succeeded and force_offline:
            # A forced-offline run can only fail because a checkpoint is unexpectedly missing from
            # the cache (offline blocks the download). Retry once with the Hub reachable.
            logger.warning("Offline annotator preload failed; retrying with the HuggingFace Hub reachable")
            succeeded = _run_preload(force_offline=False)

        if succeeded:
            if ref is not None:
                _record_annotators_verified(ref)
            _preload_completed = True
            return True

        return False


__all__ = [
    "download_all_controlnet_annotators",
]
