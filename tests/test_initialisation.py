# test_initialisation.py
import importlib.machinery
import os
import sys
import types


def test_find_comfyui():
    from hordelib.ComfyUI import execution

    assert hasattr(execution, "get_input_data")


def test_instantiation():
    from hordelib.config_path import set_system_path

    set_system_path()

    from hordelib.comfy_horde import Comfy_Horde

    _ = Comfy_Horde()
    assert isinstance(_, Comfy_Horde)
