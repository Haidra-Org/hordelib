import pytest


def test_no_initialise_comfy_horde():
    """This tests the safe-guard which ensures hordelib.initialise() has been called.."""
    import hordelib.comfy_horde

    with pytest.raises(RuntimeError):
        hordelib.comfy_horde.Comfy_Horde()


def test_no_initialise_in_process_backend():
    """The in-process execution backend must enforce the same initialise()-before-use contract."""
    from hordelib.execution.in_process import InProcessComfyBackend

    with pytest.raises(RuntimeError):
        InProcessComfyBackend().start()


def test_execution_interface_importable_without_comfy():
    """The bridge interface and graph utilities must never require ComfyUI to import."""
    from hordelib.execution import ExecutionBackend, OutputArtifact, VRAMStats  # noqa: F401
    from hordelib.execution.graph_utils import apply_dotted_params, reconnect_input  # noqa: F401
