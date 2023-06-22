import PIL.Image
import pytest

from hordelib.comfy_horde import Comfy_Horde
from hordelib.horde import HordeLib
from hordelib.model_manager.compvis import CompVisModelManager
from hordelib.shared_model_manager import ALL_MODEL_MANAGER_TYPES, SharedModelManager


@pytest.fixture(scope="session")
def init_horde():
    """This fixture initialises HordeLib and sets the VRAM to leave free to 90%.
    You must call this fixture if your test uses a module which imports `hordelib.comfy_horde`. You will usually
    see a characteristic RuntimeError exception if you forget to call this fixture, but you may also see an
    import error from within comfy if your code does not instantiate the `Comfy_Horde` class."""
    import hordelib

    hordelib.initialise()
    from hordelib.settings import UserSettings

    UserSettings.set_vram_to_leave_free_mb("90%")


@pytest.fixture(scope="session")
def hordelib_instance(init_horde) -> HordeLib:
    return HordeLib()


@pytest.fixture(scope="class")
def isolated_comfy_horde_instance(init_horde) -> Comfy_Horde:
    return Comfy_Horde()


@pytest.fixture(scope="class")
def shared_model_manager(hordelib_instance: HordeLib) -> type[SharedModelManager]:
    SharedModelManager()
    SharedModelManager.load_model_managers(ALL_MODEL_MANAGER_TYPES)

    assert SharedModelManager._instance is not None
    assert SharedModelManager.manager is not None
    assert SharedModelManager.manager.codeformer is not None
    assert SharedModelManager.manager.compvis is not None
    assert SharedModelManager.manager.controlnet is not None
    assert SharedModelManager.manager.esrgan is not None
    assert SharedModelManager.manager.gfpgan is not None
    assert SharedModelManager.manager.safety_checker is not None
    assert SharedModelManager.manager.lora is not None
    assert SharedModelManager.manager.blip is not None
    assert SharedModelManager.manager.clip is not None

    yield SharedModelManager

    SharedModelManager._instance = None
    SharedModelManager.manager = None


@pytest.fixture(scope="class")
def stable_diffusion_modelname_for_testing(shared_model_manager: type[SharedModelManager]) -> str:
    """Loads the stable diffusion model for testing. This model is used by many tests.
    This fixture returns the model name as string."""
    shared_model_manager.load_model_managers([CompVisModelManager])
    model_name = "Deliberate"
    assert shared_model_manager.manager.load(model_name)
    return model_name


@pytest.fixture(scope="session")
def db0_test_image() -> PIL.Image.Image:
    return PIL.Image.open("images/test_db0.jpg")


@pytest.fixture(scope="session")
def real_image() -> PIL.Image.Image:
    return PIL.Image.open("images/test_annotator.jpg")


# TODO
# @pytest.fixture(scope="function")
#    yield
#    assert SharedModelManager.manager.loaded_models


def pytest_collection_modifyitems(items):
    """Modifies test items to ensure test modules run in a given order."""
    MODULES_TO_RUN_FIRST = [
        "test_packaging_errors",
        "tests.test_cuda",
        "tests.test_utils",
        "tests.test_comfy_install",
        "tests.test_comfy",
        "tests.test_payload_mapping",
        "test_shared_model_manager",
        "test_mm_lora",
    ]
    MODULES_TO_RUN_LAST = [
        "tests.test_inference",
        "tests.test_horde_inference",
        "tests.test_horde_inference_img2img",
        "tests.test_horde_samplers",
        "tests.test_horde_inference_controlnet",
        "tests.test_horde_lora",
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
