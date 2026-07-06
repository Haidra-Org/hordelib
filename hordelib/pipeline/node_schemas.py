"""Committed ComfyUI node input schemas, used to audit binding targets GPU-free.

ComfyUI silently drops any prompt input whose name is not declared by the node class's
``INPUT_TYPES()`` (``ComfyUI/execution.py``, ``get_input_data``), so a typo'd input name in a
binding target never errors anywhere at runtime. The registration audit closes that hole by
checking every bound input name against this snapshot of the node schemas.

The snapshot (:data:`NODE_INPUTS_FILE`) is committed data covering exactly the node classes
hordelib's graphs use (post Horde node replacement). It is regenerated against the pinned
ComfyUI version with::

    uv run --no-sync python -m hordelib.pipeline.node_schemas

and ``tests/meta/test_node_schema_freshness.py`` fails when the vendored ComfyUI (or a graph's
node vocabulary) drifts from it, so a ComfyUI version bump that changes an input name is
caught at CI time instead of silently zeroing a parameter.
"""

import functools
import json
from dataclasses import dataclass
from pathlib import Path

NODE_INPUTS_FILE = Path(__file__).parent / "comfy_node_inputs.json"
"""The committed snapshot of node input names per class_type."""

_EXTRA_SNAPSHOT_CLASSES = (
    "HordeLoraLoader",
    "HordeConditioningOutput",
    "HordeConditioningInput",
    "HordeLatentOutput",
    "HordeLatentInput",
)
"""Classes absent from the packaged graph files but inserted at runtime: ``HordeLoraLoader`` by the
LoRA patch step, and the disaggregated-stage IO nodes (:mod:`hordelib.nodes.node_stage_io`) by the
stage-graph cut helpers (:mod:`hordelib.execution.stage_graph`)."""


@dataclass(frozen=True)
class NodeInputSchema:
    """The input names one node class accepts, split as ComfyUI declares them."""

    class_type: str
    required: tuple[str, ...]
    optional: tuple[str, ...]

    @functools.cached_property
    def all_inputs(self) -> frozenset[str]:
        """Every input name the class accepts (required and optional)."""
        return frozenset(self.required) | frozenset(self.optional)


@functools.cache
def load_node_input_schemas() -> dict[str, NodeInputSchema]:
    """Load the committed node input schema snapshot, keyed by class_type.

    Raises:
        FileNotFoundError: If the snapshot has never been generated; run
            ``python -m hordelib.pipeline.node_schemas`` (requires an initialisable ComfyUI).
    """
    raw = json.loads(NODE_INPUTS_FILE.read_text(encoding="utf-8"))
    return {
        class_type: NodeInputSchema(
            class_type=class_type,
            required=tuple(entry.get("required", [])),
            optional=tuple(entry.get("optional", [])),
        )
        for class_type, entry in raw.items()
        if not class_type.startswith("_")
    }


def _packaged_graph_class_types() -> set[str]:
    """All post-replacement node class_types used by the packaged pipeline graphs."""
    from hordelib.pipeline.definition import PACKAGED_PIPELINES_DIR
    from hordelib.pipeline.graph import ComfyGraph

    class_types: set[str] = set(_EXTRA_SNAPSHOT_CLASSES)
    for graph_file in sorted(PACKAGED_PIPELINES_DIR.glob("pipeline_*.json")):
        class_types |= ComfyGraph.from_file(graph_file).class_types()
    return class_types


def collect_node_input_schemas() -> dict[str, NodeInputSchema]:
    """Read the live ``INPUT_TYPES()`` of every class the packaged graphs use.

    Only valid in a process where ``hordelib.initialise()`` has run and custom nodes are
    loaded (a ``Comfy_Horde`` instance has been constructed); use the committed snapshot via
    :func:`load_node_input_schemas` everywhere else.

    Raises:
        RuntimeError: If a class is not registered with ComfyUI, or its ``INPUT_TYPES()``
            call fails.
    """
    from hordelib.comfy_horde import get_node_class

    schemas: dict[str, NodeInputSchema] = {}
    for class_type in sorted(_packaged_graph_class_types()):
        node_class = get_node_class(class_type)
        input_types = getattr(node_class, "INPUT_TYPES", None)
        if input_types is None:
            raise RuntimeError(f"Node class {class_type!r} declares no INPUT_TYPES()")
        try:
            declared = input_types()
        except Exception as input_types_error:
            raise RuntimeError(f"INPUT_TYPES() failed for node class {class_type!r}") from input_types_error
        schemas[class_type] = NodeInputSchema(
            class_type=class_type,
            required=tuple(declared.get("required", {})),
            optional=tuple(declared.get("optional", {})),
        )
    return schemas


def schemas_to_json(schemas: dict[str, NodeInputSchema]) -> str:
    """Serialize schemas in the committed snapshot format (stable ordering)."""
    payload = {
        class_type: {
            "required": list(schema.required),
            "optional": list(schema.optional),
        }
        for class_type, schema in sorted(schemas.items())
    }
    return json.dumps(payload, indent=2) + "\n"


def write_snapshot() -> Path:
    """Regenerate the committed snapshot from the live ComfyUI (see module docstring)."""
    NODE_INPUTS_FILE.write_text(schemas_to_json(collect_node_input_schemas()), encoding="utf-8")
    return NODE_INPUTS_FILE


def _main() -> None:
    import hordelib

    hordelib.initialise(setup_logging=False)
    from hordelib.comfy_horde import Comfy_Horde

    Comfy_Horde()
    path = write_snapshot()
    print(f"Wrote {path}")


if __name__ == "__main__":
    _main()
