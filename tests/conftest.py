import PIL.Image
import pytest

from hordelib.horde import HordeLib
from hordelib.shared_model_manager import SharedModelManager


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
    hordelib = HordeLib()
    return hordelib


@pytest.fixture(scope="session")
def shared_model_manager(hordelib_instance: HordeLib) -> type[SharedModelManager]:
    SharedModelManager()
    SharedModelManager.loadModelManagers(
        codeformer=True,
        compvis=True,
        controlnet=True,
        esrgan=True,
        gfpgan=True,
        safety_checker=True,
        lora=True,
        blip=True,
        clip=True,
    )
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

    return SharedModelManager


@pytest.fixture(scope="class")
def test_stable_diffusion_model(shared_model_manager: type[SharedModelManager]):
    shared_model_manager.loadModelManagers(compvis=True)
    shared_model_manager.manager.load("Deliberate")
    yield
    shared_model_manager.manager.unload_model("Deliberate")


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
    """Modifies test items in place to ensure test modules run in a given order."""
    MODULES_FIRST = [
        "tests.test_cuda",
        "tests.test_comfy_install",
        "tests.test_clip",
    ]
    MODULES_FIRST.reverse()
    MODULES_LAST = [
        "tests.test_horde_inference_controlnet",
        "tests.test_horde_inference_img2img",
        "tests.test_horde_inference_painting",
        "tests.test_horde_inference",
        "tests.test_horde_lora",
    ]
    # `test.scripts` must run first because it downloads the legacy database
    module_mapping = {item: item.module.__name__ for item in items}

    sorted_items = items.copy()
    # Iteratively move tests of each module to the end of the test queue
    for module in MODULES_LAST:
        sorted_items = [it for it in sorted_items if module_mapping[it] != module] + [
            it for it in sorted_items if module_mapping[it] == module
        ]

    for module in MODULES_FIRST:
        sorted_items = [it for it in sorted_items if module_mapping[it] == module] + [
            it for it in sorted_items if module_mapping[it] != module
        ]
    items[:] = sorted_items
