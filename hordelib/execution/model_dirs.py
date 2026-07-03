"""Registers hordelib's model directories with ComfyUI through folder_paths' public API.

ComfyUI resolves model files per category ("checkpoints", "loras", ...) through
``folder_paths``. hordelib keeps its models under :meth:`UserSettings.get_model_directory`
in horde-named subdirectories; this module is the single place that mapping is declared
(:data:`MODEL_CATEGORY_DIRS`) and applied (:func:`register_horde_model_paths`).

Registration uses ``folder_paths.add_model_folder_path``, which appends to a category's
search list (ComfyUI's own default directories stay registered but are empty in horde
deployments). Adding a model category for a new modality (e.g. audio encoders) is one new
row in the table.

ComfyUI must already be imported (``hordelib.initialise()``); the comfy import is deferred
to call time so this module stays importable at any point.
"""

from collections.abc import Mapping
from enum import StrEnum

from loguru import logger

from hordelib.config_path import get_hordelib_path
from hordelib.settings import UserSettings


class ModelCategory(StrEnum):
    """The ComfyUI folder_paths categories hordelib registers directories for.

    Values are ComfyUI's canonical category names (post ``map_legacy``: "unet" and "clip"
    are addressed as ``diffusion_models`` and ``text_encoders``).
    """

    CHECKPOINTS = "checkpoints"
    DIFFUSION_MODELS = "diffusion_models"
    LORAS = "loras"
    EMBEDDINGS = "embeddings"
    VAE = "vae"
    TEXT_ENCODERS = "text_encoders"
    UPSCALE_MODELS = "upscale_models"
    FACERESTORE_MODELS = "facerestore_models"
    CONTROLNET = "controlnet"
    CUSTOM_NODES = "custom_nodes"


MODEL_CATEGORY_DIRS: Mapping[ModelCategory, tuple[str, ...]] = {
    ModelCategory.CHECKPOINTS: ("compvis",),
    ModelCategory.DIFFUSION_MODELS: ("compvis",),
    ModelCategory.LORAS: ("lora",),
    ModelCategory.EMBEDDINGS: ("ti",),
    ModelCategory.VAE: ("vae",),
    ModelCategory.TEXT_ENCODERS: ("text_encoders",),
    ModelCategory.UPSCALE_MODELS: ("esrgan", "gfpgan", "codeformer"),
    ModelCategory.FACERESTORE_MODELS: ("gfpgan", "codeformer"),
    ModelCategory.CONTROLNET: ("controlnet",),
}
"""Per category, the subdirectories of the horde model directory to register.

:data:`ModelCategory.CUSTOM_NODES` is intentionally absent: it is registered from the
hordelib package itself (``hordelib/nodes``), not from the model directory.
"""


def register_horde_model_paths() -> None:
    """Register every horde model directory (and the custom-node path) with ComfyUI.

    Idempotent: ``add_model_folder_path`` ignores directories already in a category's
    search list, so repeated bridge construction is safe.

    Side Effects:
        Mutates ComfyUI's ``folder_paths`` registry for every category in
        :data:`MODEL_CATEGORY_DIRS`, plus ``custom_nodes``.
    """
    import folder_paths

    model_directory = UserSettings.get_model_directory()

    for category, subdirectories in MODEL_CATEGORY_DIRS.items():
        for subdirectory in subdirectories:
            folder_paths.add_model_folder_path(category.value, str(model_directory / subdirectory))

        # A category folder_paths did not already know (facerestore_models, from the vendored
        # facerestore_cf nodes) is created with an empty extension set, which would list every
        # file in the directory. Restrict it to model-file extensions, matching both the
        # historical bridge behavior and every built-in model category. This is the one
        # sanctioned direct touch of the registry dict; add_model_folder_path offers no way
        # to set extensions.
        registered_paths, registered_extensions = folder_paths.folder_names_and_paths[category.value]
        if not registered_extensions:
            folder_paths.folder_names_and_paths[category.value] = (
                registered_paths,
                set(folder_paths.supported_pt_extensions),
            )

    folder_paths.add_model_folder_path(ModelCategory.CUSTOM_NODES.value, str(get_hordelib_path() / "nodes"))

    logger.debug(
        "Registered horde model directories with ComfyUI",
        model_directory=str(model_directory),
        categories=[category.value for category in MODEL_CATEGORY_DIRS],
    )


def invalidate_filename_cache(category: ModelCategory) -> None:
    """Drop ComfyUI's cached file listing for a category so the next lookup rescans disk.

    Needed when files appear in a registered directory mid-process (e.g. a textual
    inversion downloaded just before a job). folder_paths exposes no public invalidation
    for its module-level ``filename_list_cache`` (``cache_helper`` is a separate,
    context-scoped cache), so the key removal is confined here and the dict's shape is
    pinned by ``tests/test_comfy_contract_drift.py``.

    Args:
        category: The category whose listing cache to drop.
    """
    import folder_paths

    folder_paths.filename_list_cache.pop(category.value, None)


def registered_paths(category: ModelCategory) -> list[str]:
    """Return the directories ComfyUI currently searches for a category.

    Args:
        category: The category to inspect.

    Returns:
        list[str]: The search paths, in precedence order.
    """
    import folder_paths

    search_paths: list[str] = folder_paths.get_folder_paths(category.value)
    return search_paths
