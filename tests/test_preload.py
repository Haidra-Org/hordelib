"""GPU-free regression guard for controlnet-annotator preload ordering.

The worker's ``download_models`` flow runs ``hordelib.initialise()`` and then
``SharedModelManager.preload_annotators()`` *without* ever constructing a ``HordeLib``.
Custom nodes (the ``comfyui_controlnet_aux`` package that registers ``AIO_Preprocessor``)
only register when the ``Comfy_Horde`` backend is built, so the preload routine must
construct a ``HordeLib`` itself. The rest of the suite always has a session-scoped
``HordeLib`` already built before it reaches preload, which is exactly why this gap went
unnoticed — so this test exercises the cold path with everything comfy/GPU monkeypatched
out.
"""

import hordelib.comfy_horde
import hordelib.horde
import hordelib.preload as preload


def test_preload_constructs_hordelib_before_node_lookup(monkeypatch):
    """``download_all_controlnet_annotators`` must build a HordeLib before looking up nodes."""
    constructed: list[object] = []

    class _FakeNode:
        def execute(self, *args, **kwargs):
            return None

    class _FakeHordeLib:
        CONTROLNET_IMAGE_PREPROCESSOR_MAP = {"canny": "CannyEdgePreprocessor"}

        def __init__(self):
            constructed.append(self)

    def _fake_get_node_class(class_type: str) -> type:
        assert constructed, (
            "preload looked up node class "
            f"{class_type!r} before constructing HordeLib; AIO_Preprocessor would be "
            "unregistered (the worker download_models regression)"
        )
        return _FakeNode

    monkeypatch.setattr(hordelib.horde, "HordeLib", _FakeHordeLib)
    monkeypatch.setattr(hordelib.comfy_horde, "get_node_class", _fake_get_node_class)
    monkeypatch.setattr(preload, "_preload_completed", False)

    assert preload.download_all_controlnet_annotators()
    assert constructed, "preload did not construct a HordeLib instance"
