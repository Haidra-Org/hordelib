"""The horde model-directory registration against a live ComfyUI folder_paths registry."""

from pathlib import Path

from hordelib.comfy_horde import Comfy_Horde
from hordelib.execution.model_dirs import (
    MODEL_CATEGORY_DIRS,
    ModelCategory,
    invalidate_filename_cache,
    registered_paths,
)
from hordelib.settings import UserSettings


def test_category_table_is_well_formed() -> None:
    assert ModelCategory.CUSTOM_NODES not in MODEL_CATEGORY_DIRS, (
        "custom_nodes is registered from the hordelib package, not the model directory"
    )
    for category, subdirectories in MODEL_CATEGORY_DIRS.items():
        assert isinstance(category, ModelCategory)
        assert subdirectories, f"category {category} declares no subdirectories"
        assert all(subdirectory for subdirectory in subdirectories)


class TestLiveRegistration:
    def test_every_category_resolves_its_horde_directories(self, init_horde: None) -> None:
        Comfy_Horde()

        model_directory = UserSettings.get_model_directory()
        for category, subdirectories in MODEL_CATEGORY_DIRS.items():
            search_paths = registered_paths(category)
            for subdirectory in subdirectories:
                expected = str(model_directory / subdirectory)
                assert expected in search_paths, (
                    f"{category.value} does not search {expected}; got {search_paths}"
                )

    def test_comfy_defaults_keep_precedence(self, init_horde: None) -> None:
        # Registration appends; ComfyUI's own (empty in horde deployments) default directory
        # stays first, matching the historical bridge behavior.
        Comfy_Horde()

        model_directory = UserSettings.get_model_directory()
        checkpoint_paths = registered_paths(ModelCategory.CHECKPOINTS)
        assert checkpoint_paths[0] != str(model_directory / "compvis")

    def test_custom_nodes_path_registered(self, init_horde: None) -> None:
        from hordelib.config_path import get_hordelib_path

        Comfy_Horde()

        assert str(get_hordelib_path() / "nodes") in registered_paths(ModelCategory.CUSTOM_NODES)

    def test_facerestore_category_has_model_extensions(self, init_horde: None) -> None:
        # facerestore_models is created by hordelib (not a ComfyUI built-in); it must filter
        # to model-file extensions rather than the empty everything-matches set.
        import folder_paths

        Comfy_Horde()

        _, extensions = folder_paths.folder_names_and_paths[ModelCategory.FACERESTORE_MODELS.value]
        assert set(extensions) == set(folder_paths.supported_pt_extensions)

    def test_repeated_construction_does_not_duplicate_paths(self, init_horde: None) -> None:
        Comfy_Horde()
        paths_after_first = registered_paths(ModelCategory.LORAS)
        Comfy_Horde()
        paths_after_second = registered_paths(ModelCategory.LORAS)

        assert paths_after_first == paths_after_second
        assert len(set(paths_after_second)) == len(paths_after_second)

    def test_invalidate_filename_cache_drops_the_listing(self, init_horde: None) -> None:
        import folder_paths

        Comfy_Horde()

        folder_paths.get_filename_list(ModelCategory.EMBEDDINGS.value)
        assert ModelCategory.EMBEDDINGS.value in folder_paths.filename_list_cache

        invalidate_filename_cache(ModelCategory.EMBEDDINGS)

        assert ModelCategory.EMBEDDINGS.value not in folder_paths.filename_list_cache

    def test_registered_directories_are_the_horde_layout(self, init_horde: None) -> None:
        # The subdirectory names are the on-disk layout contract shared with the model
        # managers; a rename there must show up here deliberately.
        expected_layout = {
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
        assert dict(MODEL_CATEGORY_DIRS) == expected_layout

    def test_model_directory_is_a_real_path(self, init_horde: None) -> None:
        assert isinstance(UserSettings.get_model_directory(), Path)
