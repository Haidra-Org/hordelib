import pytest


def test_no_initialise_comfy_horde():
    """This tests the safe-guard which ensures hordelib.initialise() has been called.."""
    import hordelib.comfy_horde

    with pytest.raises(RuntimeError):
        hordelib.comfy_horde.Comfy_Horde()
