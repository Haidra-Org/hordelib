# test_initialisation.py


def test_find_comfyui():
    import execution

    assert hasattr(execution, "get_input_data")


def test_instantiation():
    from hordelib.comfy_horde import Comfy_Horde

    _ = Comfy_Horde()
    assert isinstance(_, Comfy_Horde)


def test_path():  # XXX
    from hordelib.config_path import set_system_path

    set_system_path()
