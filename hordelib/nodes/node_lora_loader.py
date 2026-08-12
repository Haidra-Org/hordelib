import os
import uuid
from typing import Any

import comfy.utils
import folder_paths  # type: ignore
import logfire
from loguru import logger

_PATCH_IDENTITY_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "hordelib/lora-patch-identity")
"""Stable namespace for content-derived ``ModelPatcher.patches_uuid`` values."""


def _stable_patch_identity(parent_uuid: object, lora_path: str, strength: float) -> uuid.UUID | None:
    """Derive a deterministic patch identity from the incoming patcher, the lora file, and the strength.

    Comfy identifies a patch set by the *call* that produced it: ``ModelPatcher.add_patches`` rerolls
    ``patches_uuid`` with a fresh uuid4 and never looks at the patch content. A job that re-applies the
    exact lora stack already baked into the resident weights therefore mismatches the shared module's
    ``current_weight_patches_uuid``, and ``partially_load`` responds by unpatching the weights down to the
    offload device and re-uploading them. Deriving the identity from content instead lets comfy's own
    zero-cost fast return recognise the repeat. This only pays off while the weights stay resident between
    jobs; when they do not, the identity is simply unused.

    Folding in ``parent_uuid`` makes chained loader nodes compose: each link's identity covers everything
    applied before it, so two differently-ordered or differently-populated stacks cannot collide.

    Any uncertainty must produce a *different* identity rather than a matching one, so the file's size and
    mtime are part of the input and a failed stat yields ``None`` (leaving comfy's random uuid in place).
    The worst outcome is then a redundant re-bake, never serving stale weights.

    Args:
        parent_uuid: The ``patches_uuid`` of the patcher this lora is being applied to.
        lora_path: Resolved path of the lora file on disk.
        strength: The strength this lora is applied at.

    Returns:
        The derived uuid, or ``None`` if the file could not be stat'd.
    """
    try:
        stat_result = os.stat(lora_path)
    except OSError:
        return None

    return uuid.uuid5(
        _PATCH_IDENTITY_NAMESPACE,
        f"{parent_uuid}|{lora_path}|{stat_result.st_size}|{stat_result.st_mtime_ns}|{strength!r}",
    )


def _assign_stable_patch_identity(parent: Any, clone: Any, lora_path: str, strength: float, label: str) -> None:
    """Overwrite ``clone.patches_uuid`` with a content-derived identity, or leave comfy's uuid alone.

    See :func:`_stable_patch_identity` for why. Failures here must never fail the job: an unrecognised
    repeat costs a re-bake, which is exactly what the unmodified behavior costs anyway.
    """
    if parent is None or clone is None:
        return
    if not hasattr(parent, "patches_uuid") or not hasattr(clone, "patches_uuid"):
        logger.debug("lora.patch_identity_skipped: side={}, reason=no_patches_uuid", label)
        return

    identity = _stable_patch_identity(parent.patches_uuid, lora_path, strength)
    if identity is None:
        logger.debug("lora.patch_identity_skipped: side={}, reason=stat_failed, lora_path={}", label, lora_path)
        return

    clone.patches_uuid = identity


class HordeLoraLoader:
    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "lora_name": ("STRING", {"default": ""}),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01}),
                "strength_clip": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "load_lora"

    CATEGORY = "loaders"

    @logfire.instrument("lora.load_node")
    def load_lora(self, model, clip, lora_name, strength_model, strength_clip):
        from hordelib.comfy_horde import log_free_ram

        log_free_ram()
        logger.info(
            "lora.load_requested: lora_name={}, strength_model={}, strength_clip={}",
            lora_name,
            strength_model,
            strength_clip,
        )

        _test_exception = os.getenv("FAILURE_TEST", False)
        if _test_exception:
            raise Exception("This tests exceptions being thrown from within the pipeline")

        logger.debug("Loading lora through custom node: lora_name={}", lora_name)

        if strength_model == 0 and strength_clip == 0:
            logger.debug("Strengths are 0, skipping lora loading")
            logger.info("lora.load_skipped: reason=zero_strength")
            return (model, clip)

        if lora_name is None or lora_name == "" or lora_name == "None":
            logger.warning("No lora name provided, skipping lora loading")
            logger.warning("lora.load_skipped: reason=no_name")
            return (model, clip)

        if not os.path.exists(folder_paths.get_full_path("loras", lora_name)):
            logger.warning("Lora file does not exist, skipping: lora_name={}", lora_name)
            logger.warning("lora.load_failed: reason=file_not_found, lora_name={}", lora_name)
            return (model, clip)

        loras_on_disk = folder_paths.get_filename_list("loras")

        if "loras" in folder_paths.filename_list_cache:
            del folder_paths.filename_list_cache["loras"]

        if lora_name not in loras_on_disk:
            logger.warning("Lora file does not exist, skipping: lora_name={}", lora_name)
            return (model, clip)

        lora_path = folder_paths.get_full_path("loras", lora_name)
        lora = None
        if self.loaded_lora is not None:
            if self.loaded_lora[0] == lora_path:
                lora = self.loaded_lora[1]
            else:
                temp = self.loaded_lora
                self.loaded_lora = None
                del temp

        try:
            with logger.catch(reraise=True):
                if lora is None:
                    with logfire.span("lora.load_from_disk", lora_path=lora_path):
                        lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
                        self.loaded_lora = (lora_path, lora)

                with logfire.span("lora.apply_to_models"):
                    model_lora, clip_lora = comfy.sd.load_lora_for_models(
                        model,
                        clip,
                        lora,
                        strength_model,
                        strength_clip,
                    )
                try:
                    _assign_stable_patch_identity(model, model_lora, lora_path, strength_model, "model")
                    clip_patcher = clip.patcher if hasattr(clip, "patcher") else None
                    clip_lora_patcher = clip_lora.patcher if hasattr(clip_lora, "patcher") else None
                    _assign_stable_patch_identity(
                        clip_patcher,
                        clip_lora_patcher,
                        lora_path,
                        strength_clip,
                        "clip",
                    )
                except Exception as identity_error:
                    logger.debug("lora.patch_identity_failed: error={}", identity_error)

                log_free_ram()
                logger.info("lora.loaded_successfully: lora_name={}", lora_name)
                return (model_lora, clip_lora)
        except Exception as e:
            logger.bind(lora_name=lora_name).exception("Error loading lora")
            logger.error("lora.load_exception: lora_name={}, error={}", lora_name, str(e))
            return (model, clip)


NODE_CLASS_MAPPINGS = {"HordeLoraLoader": HordeLoraLoader}
