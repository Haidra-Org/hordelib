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

    assert HORDELIB_CUSTOM_MODELS is not None

    # Load the custom models json and confirm the model is on disk
    custom_models = None
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

    hordelib.initialise(setup_logging=True, logging_verbosity=5, disable_smart_memory=True)
    from hordelib.settings import UserSettings

    UserSettings.set_ram_to_leave_free_mb("100%")
    UserSettings.set_vram_to_leave_free_mb("90%")


@pytest.fixture(scope="session")
def hordelib_instance(init_horde) -> HordeLib:
    return HordeLib()


@pytest.fixture(scope="class")
def isolated_comfy_horde_instance(init_horde) -> Comfy_Horde:
    return Comfy_Horde()


@pytest.fixture(scope="session")
def shared_model_manager(
    custom_model_info_for_testing: tuple[str, str, str, str],
    hordelib_instance: HordeLib,
) -> Generator[type[SharedModelManager], None, None]:
    SharedModelManager()
    SharedModelManager.load_model_managers(ALL_MODEL_MANAGER_TYPES)

    assert SharedModelManager()._instance is not None
    assert SharedModelManager.manager is not None
    assert SharedModelManager.manager.codeformer is not None
    assert SharedModelManager.manager.codeformer.download_all_models()
    assert SharedModelManager.manager.miscellaneous is not None
    assert SharedModelManager.manager.miscellaneous.download_all_models()
    assert SharedModelManager.manager.compvis is not None

    assert SharedModelManager.manager.download_model("Deliberate")
    assert SharedModelManager.manager.validate_model("Deliberate")
    assert SharedModelManager.manager.download_model("SDXL 1.0")
    assert SharedModelManager.manager.validate_model("SDXL 1.0")
    assert SharedModelManager.manager.download_model("AlbedoBase XL (SDXL)")
    assert SharedModelManager.manager.validate_model("AlbedoBase XL (SDXL)")
    assert SharedModelManager.manager.download_model("Rev Animated")
    assert SharedModelManager.manager.validate_model("Rev Animated")

    assert SharedModelManager.manager.download_model("Stable Cascade 1.0")
    assert SharedModelManager.manager.validate_model("Stable Cascade 1.0")

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


_testing_model_name = "Deliberate"


@pytest.fixture(scope="session")
def stable_diffusion_model_name_for_testing(shared_model_manager: type[SharedModelManager]) -> str:
    return _testing_model_name


@pytest.fixture(scope="session")
def sdxl_1_0_base_model_name(shared_model_manager: type[SharedModelManager]) -> str:
    return "SDXL 1.0"


@pytest.fixture(scope="session")
def sdxl_refined_model_name(shared_model_manager: type[SharedModelManager]) -> str:
    return "AlbedoBase XL (SDXL)"


@pytest.fixture(scope="session")
def stable_cascade_base_model_name(shared_model_manager: type[SharedModelManager]) -> str:
    return "Stable Cascade 1.0"


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
        "tests.test_horde_inference_cascade",
        "tests.test_horde_samplers",
        "tests.test_horde_ti",
        "tests.test_horde_lcm",
        "tests.test_horde_lora",
        "tests.test_horde_inference_controlnet",
        "tests.test_horde_inference_painting",
    ]
    module_mapping = {item: item.module.__name__ for item in items}

    sorted_items = []

    for module in MODULES_TO_RUN_FIRST:
        sorted_items.extend([item for item in items if module_mapping[item] == module])

    sorted_items.extend(
        [item for item in items if module_mapping[item] not in MODULES_TO_RUN_FIRST + MODULES_TO_RUN_LAST],
    )

    for module in MODULES_TO_RUN_LAST:
        sorted_items.extend([item for item in items if module_mapping[item] == module])

    items[:] = sorted_items
