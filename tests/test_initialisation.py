# test_initialisation.py
from hordelib.comfy_horde import Comfy_Horde
from hordelib.horde import HordeLib


def test_find_comfyui(init_horde):
    import execution

    assert hasattr(execution, "get_input_data")


def test_instantiation(hordelib_instance: HordeLib):
    assert isinstance(hordelib_instance.generator, Comfy_Horde)


def test_path():  # XXX
    from hordelib.config_path import set_system_path

    set_system_path()
