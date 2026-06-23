"""Unit tests for the seamless-tiling VAE helpers in ``node_model_loader``.

Regression guard for the Z-Image load crash: its checkpoint returns a VAE whose ``first_stage_model`` is
None, so the tiling helpers must skip rather than fault. A fault here previously took down the whole model
load and churned the worker's inference processes into a recovery loop.

The helpers live in ``hordelib.nodes.node_model_loader``, which imports ``comfy`` at module scope; ``comfy``
is only importable once ``hordelib.initialise()`` has run, so these depend on the session ``init_horde``
fixture and import the helpers locally rather than at collection time.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


@pytest.mark.usefixtures("init_horde")
class TestVaeTilingGuards:
    def test_regular_vae_skips_when_first_stage_model_is_none(self) -> None:
        from hordelib.nodes.node_model_loader import make_regular_vae

        # No exception is the assertion: a None first_stage_model must be a no-op, not an AttributeError.
        make_regular_vae(SimpleNamespace(first_stage_model=None))

    def test_circular_vae_skips_when_first_stage_model_is_none(self) -> None:
        from hordelib.nodes.node_model_loader import make_circular_vae

        make_circular_vae(SimpleNamespace(first_stage_model=None))

    def test_regular_vae_skips_when_attribute_absent(self) -> None:
        from hordelib.nodes.node_model_loader import make_regular_vae

        make_regular_vae(SimpleNamespace())

    def test_regular_vae_applies_when_first_stage_model_present(self) -> None:
        from hordelib.nodes.node_model_loader import make_regular_vae

        vae = SimpleNamespace(first_stage_model=MagicMock())
        make_regular_vae(vae)
        vae.first_stage_model.apply.assert_called_once()

    def test_circular_vae_applies_when_first_stage_model_present(self) -> None:
        from hordelib.nodes.node_model_loader import make_circular_vae

        vae = SimpleNamespace(first_stage_model=MagicMock())
        make_circular_vae(vae)
        vae.first_stage_model.apply.assert_called_once()
