"""Preloading of auxiliary models (controlnet annotators).

ControlNet preprocessing is provided by the ``comfyui_controlnet_aux`` custom node package
(pinned via ``hordelib/installation/manifest.json``). Its detectors download their checkpoint
files from the HuggingFace hub on first use, into the directory named by the
``AUX_ANNOTATOR_CKPTS_PATH`` environment variable (set during ``hordelib.initialise()``).

Preloading simply exercises each supported preprocessor once on a tiny image, which triggers
(and therefore verifies) the downloads ahead of any real generation.
"""

import threading

from loguru import logger

_preload_mutex = threading.Lock()
_preload_completed = False


def download_all_controlnet_annotators() -> bool:
    """Download (and verify by running) all controlnet annotators hordelib supports.

    Requires ``hordelib.initialise()`` to have completed and custom nodes to be loaded
    (i.e. a ``Comfy_Horde``/``HordeLib`` instance must have been constructed).

    Returns:
        bool: True if all annotators are available and runnable, False otherwise.
    """
    global _preload_completed
    with _preload_mutex:
        if _preload_completed:
            return True

        try:
            import torch

            from hordelib.comfy_horde import get_node_class
            from hordelib.horde import HordeLib

            aio_preprocessor_class = get_node_class("AIO_Preprocessor")

            preprocessors = sorted(set(HordeLib.CONTROLNET_IMAGE_PREPROCESSOR_MAP.values()))
            # A tiny gray test card; enough for every detector to run its model once
            test_image = torch.full((1, 64, 64, 3), 0.5)

            for i, preprocessor in enumerate(preprocessors):
                logger.info(
                    "Preloading controlnet annotator",
                    preprocessor=preprocessor,
                    current=i + 1,
                    total=len(preprocessors),
                )
                aio_preprocessor_class().execute(preprocessor, test_image, resolution=64)

            _preload_completed = True
            return True
        except Exception as e:
            logger.exception("Failed to preload controlnet annotators: error={}", e)
            return False


__all__ = [
    "download_all_controlnet_annotators",
]
