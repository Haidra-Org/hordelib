# test_initialisation.py


def test_find_comfyui():
    import hordelib.ComfyUI

    _ = hordelib.ComfyUI.execution


def test_instantiation():
    from hordelib.comfy import Comfy

    _ = Comfy()
