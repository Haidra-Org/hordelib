# test_initialisation.py
import importlib


def test_find_comfyui():
    import hordelib.ComfyUI

    assert hordelib.ComfyUI is not None

    executionLoader = importlib.find_loader("hordelib.ComfyUI.execution")
    assert executionLoader is not None


def test_instantiation():
    from hordelib.comfy import Comfy

    _ = Comfy()
    assert isinstance(_, Comfy)
