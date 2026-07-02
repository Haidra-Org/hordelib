"""The committed node schema snapshot matches the live, pinned ComfyUI.

The binding audit trusts ``hordelib/pipeline/comfy_node_inputs.json``; this test is what
keeps that trust honest. It needs an initialised ComfyUI with custom nodes loaded, so it runs
with the comfy-initialised suites (the same environments that run the GPU oracles). When it
fails after a ComfyUI version bump, regenerate the snapshot and re-audit::

    uv run --no-sync python -m hordelib.pipeline.node_schemas
"""

from hordelib.comfy_horde import Comfy_Horde
from hordelib.pipeline.node_schemas import NODE_INPUTS_FILE, collect_node_input_schemas, schemas_to_json


def test_node_schema_snapshot_is_current(init_horde: None) -> None:
    Comfy_Horde()

    live = schemas_to_json(collect_node_input_schemas())
    committed = NODE_INPUTS_FILE.read_text(encoding="utf-8")

    assert live == committed, (
        "The committed ComfyUI node schema snapshot is stale; regenerate it with "
        "`uv run --no-sync python -m hordelib.pipeline.node_schemas` and re-run the "
        "pipeline audits/tests."
    )
