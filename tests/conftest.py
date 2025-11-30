import json
import os
from collections.abc import Generator
from pathlib import Path

import PIL.Image
import pytest

os.environ["TESTS_ONGOING"] = "1"


from hordelib.comfy_horde import Comfy_Horde
from hordelib.horde import HordeLib
from hordelib.model_manager.hyper import ALL_MODEL_MANAGER_TYPES
from hordelib.shared_model_manager import SharedModelManager

from .testing_shared_classes import ResolutionTestCase


@pytest.fixture(scope="function", autouse=True)
def line_break():
    print()


@pytest.fixture(scope="session")
def init_horde(
    custom_model_info_for_testing: tuple[str, str, str, str],
    default_custom_model_json_path: str,
    default_custom_model_json: dict[str, dict],
):
    """This fixture initialises HordeLib and sets the VRAM to leave free to 90%.
    You must call this fixture if your test uses a module which imports `hordelib.comfy_horde`. You will usually
    see a characteristic RuntimeError exception if you forget to call this fixture, but you may also see an
    import error from within comfy if your code does not instantiate the `Comfy_Horde` class."""
    assert os.getenv("TESTS_ONGOING")
    assert os.getenv("CIVIT_API_TOKEN")

    examples_path = Path(__file__).parent.parent / "images_expected"
    assert (
        examples_path.exists() and examples_path.is_dir()
    ), "The `images_expected` directory must exist. You can find in in the github repo."

    HORDELIB_CUSTOM_MODELS = os.getenv("HORDELIB_CUSTOM_MODELS", None)
    print(f"HORDELIB_CUSTOM_MODELS: {HORDELIB_CUSTOM_MODELS}")

    if HORDELIB_CUSTOM_MODELS is not None:
        assert os.path.exists(
            HORDELIB_CUSTOM_MODELS,
        ), f"Custom models directory {HORDELIB_CUSTOM_MODELS} does not exist."
    else:
        if not os.path.exists(default_custom_model_json_path):
            os.makedirs(os.path.dirname(default_custom_model_json_path), exist_ok=True)
            with open(default_custom_model_json_path, "w") as f:
                json.dump(default_custom_model_json, f, indent=4)

        os.environ["HORDELIB_CUSTOM_MODELS"] = default_custom_model_json_path

    HORDELIB_CUSTOM_MODELS = os.getenv("HORDELIB_CUSTOM_MODELS", None)

    # assert HORDELIB_CUSTOM_MODELS is not None

    # Load the custom models json and confirm the model is on disk
    custom_models = None
    if HORDELIB_CUSTOM_MODELS is not None:
        with open(HORDELIB_CUSTOM_MODELS) as f:
            custom_models = json.load(f)

    assert custom_models is not None

    custom_model_name, _, custom_model_filename, custom_model_url = custom_model_info_for_testing

    assert custom_model_name in custom_models
    assert "config" in custom_models[custom_model_name]
    assert "files" in custom_models[custom_model_name]["config"]
    assert "path" in custom_models[custom_model_name]["config"]["files"][0]
    assert custom_model_filename in custom_models[custom_model_name]["config"]["files"][0]["path"]

    custom_model_in_json_path = custom_models[custom_model_name]["config"]["files"][0]["path"]

    print(f"Custom model path: {custom_model_in_json_path}")
    # If the custom model is not on disk, download it
    if not os.path.exists(custom_model_in_json_path):
        import requests

        response = requests.get(custom_model_url)
        response.raise_for_status()

        with open(custom_model_in_json_path, "wb") as f:
            f.write(response.content)

    import hordelib

    extra_comfyui_args = []
    extra_comfyui_args.extend(["--reserve-vram", "1.4"])

    hordelib.initialise(
        setup_logging=True,
        logging_verbosity=5,
        disable_smart_memory=True,
        force_normal_vram_mode=True,
        do_not_load_model_mangers=True,
        extra_comfyui_args=extra_comfyui_args,
        # models_not_to_force_load=[
        #     "sdxl",
        #     "cascade",
        #     "flux",
        # ],
    )
    from hordelib.settings import UserSettings

    UserSettings.set_ram_to_leave_free_mb("100%")
    UserSettings.set_vram_to_leave_free_mb("90%")


@pytest.fixture(scope="session")
def hordelib_instance(init_horde) -> HordeLib:
    return HordeLib()


@pytest.fixture(scope="class")
def isolated_comfy_horde_instance(init_horde) -> Comfy_Horde:
    return Comfy_Horde()


_testing_model_name = "Deliberate"
_sdxl_1_0_model_name = "SDXL 1.0"
_sdxl_refined_model_name = "AlbedoBase XL (SDXL)"
_stable_cascade_base_model_name = "Stable Cascade 1.0"
_flux1_schnell_fp8_base_model_name = "Flux.1-Schnell fp8 (Compact)"
_qwen_fp8_base_model_name = "Qwen-Image_fp8"
_z_image_turbo_base_model_name = "Z-Image-Turbo"
_am_pony_xl_model_name = "AMPonyXL"
_rev_animated_model_name = "Rev Animated"

_all_model_names = [
    _testing_model_name,
    _sdxl_1_0_model_name,
    _sdxl_refined_model_name,
    _stable_cascade_base_model_name,
    _flux1_schnell_fp8_base_model_name,
    _qwen_fp8_base_model_name,
    _z_image_turbo_base_model_name,
    _am_pony_xl_model_name,
    _rev_animated_model_name,
]

# !!!!
# If you're adding a model name, follow the pattern and **add it to `_all_model_names`**
# !!!!


@pytest.fixture(scope="session")
def stable_diffusion_model_name_for_testing(shared_model_manager: type[SharedModelManager]) -> str:
    """The default stable diffusion 1.5 model name used for testing."""
    return _testing_model_name


@pytest.fixture(scope="session")
def sdxl_1_0_base_model_name(shared_model_manager: type[SharedModelManager]) -> str:
    """The default SDXL 1.0 model name used for testing."""
    return _sdxl_1_0_model_name


@pytest.fixture(scope="session")
def sdxl_refined_model_name(shared_model_manager: type[SharedModelManager]) -> str:
    """The default SDXL finetune model name used for testing."""
    return _sdxl_refined_model_name


@pytest.fixture(scope="session")
def stable_cascade_base_model_name(shared_model_manager: type[SharedModelManager]) -> str:
    """The default stable cascade 1.0 model name used for testing."""
    return _stable_cascade_base_model_name


@pytest.fixture(scope="session")
def flux1_schnell_fp8_base_model_name(shared_model_manager: type[SharedModelManager]) -> str:
    """The default flux1-schnell fp8 model name used for testing."""
    return _flux1_schnell_fp8_base_model_name


@pytest.fixture(scope="session")
def qwen_image_fp8_base_model_name(shared_model_manager: type[SharedModelManager]) -> str:
    """The default qwen-iimage fp8 model name used for testing."""
    return _qwen_fp8_base_model_name


@pytest.fixture(scope="session")
def z_image_turbo_base_model_name(shared_model_manager: type[SharedModelManager]) -> str:
    """The default z-image turbo model name used for testing."""
    return _z_image_turbo_base_model_name


@pytest.fixture(scope="session")
def am_pony_xl_model_name(shared_model_manager: type[SharedModelManager]) -> str:
    """The default AMPonyXL model name used for testing."""
    return _am_pony_xl_model_name


@pytest.fixture(scope="session")
def rev_animated_model_name(shared_model_manager: type[SharedModelManager]) -> str:
    """The default Rev Animated model name used for testing."""
    return _rev_animated_model_name


# !!!!
# If you're adding a model name, follow the pattern and **add it to `_all_model_names`**
# !!!!


@pytest.fixture(scope="session")
def shared_model_manager(
    custom_model_info_for_testing: tuple[str, str, str, str],
    hordelib_instance: HordeLib,
) -> Generator[type[SharedModelManager], None, None]:
    assert hordelib_instance
    SharedModelManager(do_not_load_model_mangers=True)
    SharedModelManager.load_model_managers(ALL_MODEL_MANAGER_TYPES)

    assert SharedModelManager()._instance is not None
    assert SharedModelManager.manager is not None
    assert SharedModelManager.manager.codeformer is not None
    assert SharedModelManager.manager.codeformer.download_all_models()
    assert SharedModelManager.manager.miscellaneous is not None
    assert SharedModelManager.manager.miscellaneous.download_all_models()
    assert SharedModelManager.manager.compvis is not None

    for model_name in _all_model_names:
        assert SharedModelManager.manager.compvis.download_model(model_name)
        assert SharedModelManager.manager.compvis.validate_model(model_name)

    custom_model_name, _, _, _ = custom_model_info_for_testing
    assert custom_model_name in SharedModelManager.manager.compvis.available_models

    assert SharedModelManager.manager.controlnet is not None
    assert SharedModelManager.manager.controlnet.download_all_models()
    assert SharedModelManager.preload_annotators()
    assert SharedModelManager.manager.esrgan is not None
    assert SharedModelManager.manager.esrgan.download_all_models()
    assert SharedModelManager.manager.gfpgan is not None
    assert SharedModelManager.manager.gfpgan.download_all_models()
    assert SharedModelManager.manager.safety_checker is not None
    assert SharedModelManager.manager.safety_checker.download_all_models()
    assert SharedModelManager.manager.lora is not None
    assert SharedModelManager.manager.ti is not None

    yield SharedModelManager

    SharedModelManager._instance = None  # type: ignore
    SharedModelManager.manager = None  # type: ignore


@pytest.fixture(scope="session")
def custom_model_info_for_testing() -> tuple[str, str, str, str]:
    """Returns a tuple of the custom model name, its baseline, the on-disk file name and the download url."""
    # https://civitai.com/models/338712/pvc-style-modelmovable-figure-model-xl?modelVersionId=413807
    return (
        "Movable figure model XL",
        "stable_diffusion_xl",
        "PVCStyleModelMovable_beta25Realistic.safetensors",
        "https://huggingface.co/mirroring/horde_models/resolve/main/PVCStyleModelMovable_beta25Realistic.safetensors?download=true",
    )


@pytest.fixture(scope="session")
def default_custom_model_directory_name() -> str:
    return "custom"


@pytest.fixture(scope="session")
def default_custom_model_json_path(default_custom_model_directory_name) -> str:
    AIWORKER_CACHE_HOME = os.getenv("AIWORKER_CACHE_HOME", "models")
    return os.path.join(AIWORKER_CACHE_HOME, default_custom_model_directory_name, "custom_models.json")


@pytest.fixture(scope="session")
def default_custom_model_json(
    custom_model_info_for_testing: tuple[str, str, str, str],
    default_custom_model_directory_name,
) -> dict[str, dict]:
    model_name, baseline, filename, _ = custom_model_info_for_testing
    AIWORKER_CACHE_HOME = os.getenv("AIWORKER_CACHE_HOME", "models")
    return {
        model_name: {
            "name": model_name,
            "baseline": baseline,
            "type": "ckpt",
            "config": {
                "files": [{"path": os.path.join(AIWORKER_CACHE_HOME, default_custom_model_directory_name, filename)}],
            },
        },
    }


@pytest.fixture(scope="session")
def db0_test_image() -> PIL.Image.Image:
    return PIL.Image.open("images/test_db0.jpg")


@pytest.fixture(scope="session")
def real_image() -> PIL.Image.Image:
    return PIL.Image.open("images/test_annotator.jpg")


@pytest.fixture(scope="session")
def lora_GlowingRunesAI(shared_model_manager: type[SharedModelManager]) -> str:
    assert shared_model_manager.manager.lora
    if shared_model_manager.manager.lora.is_model_available("51686"):
        return "51686"
    name = shared_model_manager.manager.lora.fetch_adhoc_lora("51686")
    assert name is not None
    assert shared_model_manager.manager.lora.is_model_available(name)

    return name


@pytest.fixture(scope="session")
def default_min_steps() -> int:
    return 6


@pytest.fixture(scope="session")
def default_first_pass_steps() -> int:
    return 30


@pytest.fixture(scope="session")
def default_hires_fix_denoise_strength() -> float:
    return 0.65


@pytest.fixture(scope="session")
def sdxl_hires_test_cases(
    default_first_pass_steps: int,
    default_hires_fix_denoise_strength: int,
    default_min_steps: int,
) -> list[ResolutionTestCase]:
    sdxl_model_native_resolution = 1024
    sdxl_default_second_pass_steps = int(default_first_pass_steps / 2.25)
    sdxl_high_second_pass_steps = default_first_pass_steps * 0.6
    return [
        ResolutionTestCase(
            width=1024,
            height=1024,
            ddim_steps=default_first_pass_steps,
            hires_fix_denoise_strength=default_hires_fix_denoise_strength,
            model_native_resolution=sdxl_model_native_resolution,
            max_expected_steps=default_first_pass_steps / 2,
            min_expected_steps=default_min_steps,
        ),
        ResolutionTestCase(
            width=1280,
            height=1280,
            ddim_steps=default_first_pass_steps,
            hires_fix_denoise_strength=default_hires_fix_denoise_strength,
            model_native_resolution=sdxl_model_native_resolution,
            max_expected_steps=sdxl_high_second_pass_steps + default_min_steps,
            min_expected_steps=default_min_steps,
        ),
        ResolutionTestCase(
            width=1156,
            height=1480,
            ddim_steps=default_first_pass_steps,
            hires_fix_denoise_strength=default_hires_fix_denoise_strength,
            model_native_resolution=sdxl_model_native_resolution,
            max_expected_steps=sdxl_high_second_pass_steps + default_min_steps,
            min_expected_steps=sdxl_default_second_pass_steps,
        ),
        ResolutionTestCase(
            width=2048,
            height=2048,
            ddim_steps=default_first_pass_steps,
            hires_fix_denoise_strength=default_hires_fix_denoise_strength,
            model_native_resolution=sdxl_model_native_resolution,
            max_expected_steps=default_first_pass_steps,
            min_expected_steps=default_first_pass_steps - default_min_steps,
        ),
        ResolutionTestCase(
            width=1600,
            height=1600,
            ddim_steps=default_first_pass_steps,
            hires_fix_denoise_strength=default_hires_fix_denoise_strength,
            model_native_resolution=sdxl_model_native_resolution,
            max_expected_steps=default_first_pass_steps,
            min_expected_steps=sdxl_high_second_pass_steps - default_min_steps,
        ),
        ResolutionTestCase(
            width=1664,
            height=1152,
            ddim_steps=default_first_pass_steps,
            hires_fix_denoise_strength=default_hires_fix_denoise_strength,
            model_native_resolution=sdxl_model_native_resolution,
            max_expected_steps=default_first_pass_steps,
            min_expected_steps=sdxl_high_second_pass_steps - default_min_steps,
        ),
        ResolutionTestCase(
            height=1664,
            width=1152,
            ddim_steps=12,
            hires_fix_denoise_strength=0.65,
            model_native_resolution=1024,
            max_expected_steps=12,
            min_expected_steps=9,
        ),
    ]


@pytest.fixture(scope="session")
def sd15_hires_test_cases(
    default_first_pass_steps: int,
    default_hires_fix_denoise_strength: int,
    default_min_steps: int,
) -> list[ResolutionTestCase]:
    sd15_model_native_resolution = 512
    sd15_high_second_pass_steps = default_first_pass_steps * 0.67
    return [
        ResolutionTestCase(
            width=512,
            height=512,
            ddim_steps=default_first_pass_steps,
            hires_fix_denoise_strength=default_hires_fix_denoise_strength,
            model_native_resolution=sd15_model_native_resolution,
            max_expected_steps=sd15_high_second_pass_steps,
            min_expected_steps=default_min_steps,
        ),
        ResolutionTestCase(
            width=640,
            height=640,
            ddim_steps=default_first_pass_steps,
            hires_fix_denoise_strength=default_hires_fix_denoise_strength,
            model_native_resolution=sd15_model_native_resolution,
            max_expected_steps=sd15_high_second_pass_steps + default_min_steps,
            min_expected_steps=default_min_steps,
        ),
        ResolutionTestCase(
            width=578,
            height=740,
            ddim_steps=default_first_pass_steps,
            hires_fix_denoise_strength=default_hires_fix_denoise_strength,
            model_native_resolution=sd15_model_native_resolution,
            max_expected_steps=default_min_steps * 2.25,
            min_expected_steps=default_min_steps * 1.25,
        ),
        ResolutionTestCase(
            width=1024,
            height=1024,
            ddim_steps=default_first_pass_steps,
            hires_fix_denoise_strength=default_hires_fix_denoise_strength,
            model_native_resolution=sd15_model_native_resolution,
            max_expected_steps=18,
            min_expected_steps=default_min_steps * 2,
        ),
        ResolutionTestCase(
            width=1536,
            height=1024,
            ddim_steps=default_first_pass_steps,
            hires_fix_denoise_strength=default_hires_fix_denoise_strength,
            model_native_resolution=sd15_model_native_resolution,
            max_expected_steps=default_first_pass_steps,
            min_expected_steps=default_first_pass_steps - default_min_steps,
        ),
        ResolutionTestCase(
            width=800,
            height=800,
            ddim_steps=default_first_pass_steps,
            hires_fix_denoise_strength=default_hires_fix_denoise_strength,
            model_native_resolution=sd15_model_native_resolution,
            max_expected_steps=default_min_steps * 2.5,
            min_expected_steps=default_min_steps * 1.5,
        ),
        ResolutionTestCase(
            width=2048,
            height=2048,
            ddim_steps=default_first_pass_steps,
            hires_fix_denoise_strength=default_hires_fix_denoise_strength,
            model_native_resolution=sd15_model_native_resolution,
            max_expected_steps=default_first_pass_steps,
            min_expected_steps=default_first_pass_steps,
        ),
    ]


@pytest.fixture(scope="session")
def all_hires_test_cases(
    sdxl_hires_test_cases: list[ResolutionTestCase],
    sd15_hires_test_cases: list[ResolutionTestCase],
) -> list[ResolutionTestCase]:
    return sdxl_hires_test_cases + sd15_hires_test_cases


def pytest_collection_modifyitems(items):
    """Modifies test items to ensure test modules run in a given order."""
    MODULES_TO_RUN_FIRST = [
        "tests.meta.test_packaging_errors",
        "tests.test_internal_comfyui_failures",
        "tests.test_initialisation",
        "tests.test_cuda",
        "tests.test_utils",
        "tests.test_comfy_install",
        "tests.test_comfy",
        "tests.test_payload_mapping",
        "tests.model_managers.test_shared_model_manager",
        "tests.test_mm_lora",
        "tests.test_mm_ti",
        "tests.test_inference",
    ]
    MODULES_TO_RUN_LAST = [
        "tests.test_horde_inference",
        "tests.test_horde_inference_img2img",
        "tests.test_horde_inference_qrcode",
        "tests.test_horde_samplers",
        "tests.test_horde_ti",
        "tests.test_horde_lcm",
        "tests.test_horde_lora",
        "tests.test_horde_inference_controlnet",
        "tests.test_horde_inference_painting",
        "tests.test_horde_inference_cascade",
    ]
    module_mapping = {item: item.module.__name__ for item in items}

    sorted_items = []

    PYTEST_MARK_LAST = [
        "default_sdxl_model",
        "refined_sdxl_model",
    ]

    for module in MODULES_TO_RUN_FIRST:
        sorted_module = [item for item in items if module_mapping[item] == module]
        # Any items with a mark in PYTEST_MARK_LAST will be moved to the end (in the order of the list)
        for mark in PYTEST_MARK_LAST:
            marked_items = [
                item for item in sorted_module if any(own_marker.name == mark for own_marker in item.own_markers)
            ]
            sorted_module = [item for item in sorted_module if item not in marked_items] + marked_items

        sorted_items.extend(sorted_module)

    sorted_items.extend(
        [item for item in items if module_mapping[item] not in MODULES_TO_RUN_FIRST + MODULES_TO_RUN_LAST],
    )

    for module in MODULES_TO_RUN_LAST:
        sorted_module = [item for item in items if module_mapping[item] == module]
        # Any items with a mark in PYTEST_MARK_LAST will be moved to the end (in the order of the list)
        for mark in PYTEST_MARK_LAST:
            marked_items = [
                item for item in sorted_module if any(own_marker.name == mark for own_marker in item.own_markers)
            ]
            sorted_module = [item for item in sorted_module if item not in marked_items] + marked_items

        sorted_items.extend(sorted_module)

    # Any items with a mark in PYTEST_MARK_LAST will be moved to the end (in the order of the list)
    # for mark in PYTEST_MARK_LAST:
    #     marked_items = [
    #         item for item in sorted_items if any(own_marker.name == mark for own_marker in item.own_markers)
    #     ]
    #     sorted_items = [item for item in sorted_items if item not in marked_items] + marked_items

    items[:] = sorted_items
