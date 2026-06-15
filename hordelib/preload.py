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
from collections.abc import Iterator
from pathlib import Path

from loguru import logger

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


def _preload_marker_path() -> Path | None:
    """Return the on-disk marker path, or None if the annotator directory is unknown."""
    ckpts_dir = os.environ.get("AUX_ANNOTATOR_CKPTS_PATH")
    if not ckpts_dir:
        return None
    return Path(ckpts_dir) / _PRELOAD_MARKER_NAME


def _annotators_already_verified(ref: str) -> bool:
    """Whether a previous process already downloaded and verified annotators for this ``ref``."""
    marker = _preload_marker_path()
    if marker is None:
        return False
    try:
        return marker.is_file() and marker.read_text(encoding="utf-8").strip() == ref
    except OSError:
        return False


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
