"""A typed wrapper over ComfyUI API-format pipeline graphs.

``ComfyGraph`` is the only sanctioned way for pipeline code to manipulate graph JSON; the raw
dict helpers live in :mod:`hordelib.execution.graph_utils`. Node ``_meta.title`` values are the
canonical handles (a KSampler titled ``sampler`` is addressed as ``sampler.steps``).
"""

import copy
import json
from pathlib import Path
from typing import Any, NamedTuple, Self

from hordelib.execution.graph_utils import (
    GraphDict,
    apply_dotted_params,
    fix_node_names,
    fix_pipeline_types,
    reconnect_input,
)

HORDE_NODE_REPLACEMENTS = {
    "CheckpointLoaderSimple": "HordeCheckpointLoader",
    "UNETLoader": "HordeCheckpointLoader",
    "SaveImage": "HordeImageOutput",
    "LoadImage": "HordeImageLoader",
    "LoraLoader": "HordeLoraLoader",
}
"""ComfyUI standard node types replaced by Horde-specific implementations at load time."""


class NodeRef(NamedTuple):
    """A reference to a node output, e.g. ``NodeRef("model_loader", 0)``."""

    title: str
    output: int = 0


class ComfyGraph:
    """An API-format pipeline graph with title-keyed nodes."""

    def __init__(self, graph: GraphDict):
        self._graph = graph

    @classmethod
    def from_file(cls, path: Path) -> Self:
        """Load a pipeline JSON file, applying Horde node replacements and title renaming."""
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        raw = fix_pipeline_types(raw, HORDE_NODE_REPLACEMENTS)
        return cls(fix_node_names(raw))

    def copy(self) -> "ComfyGraph":
        """Return a deep copy of this graph (templates hand out copies for mutation)."""
        return ComfyGraph(copy.deepcopy(self._graph))

    @property
    def raw(self) -> GraphDict:
        """The underlying title-keyed graph dict (mutable view).

        For patch steps that delegate to the pure dict-level functions in
        :mod:`hordelib.pipeline.patches`; everything else should use the typed methods.
        """
        return self._graph

    def has_node(self, title: str) -> bool:
        return title in self._graph

    def node(self, title: str) -> dict[str, Any]:
        """Return the node dict for a title.

        Raises:
            KeyError: If no node with that title exists.
        """
        return self._graph[title]

    def node_titles(self) -> list[str]:
        return list(self._graph.keys())

    def class_types(self) -> set[str]:
        """All class_types used in the graph (for allowlist auditing)."""
        return {node["class_type"] for node in self._graph.values() if "class_type" in node}

    def set_input(self, target: str, value: Any) -> None:
        """Set a node input via a dotted ``title.input`` path.

        Raises:
            KeyError: If the target node does not exist in this graph.
        """
        skipped = apply_dotted_params(self._graph, {target: value})
        if skipped:
            raise KeyError(f"Cannot set {target!r}: node not present in this graph")

    def set_inputs(self, params: dict[str, Any], *, strict: bool = False) -> int:
        """Set multiple dotted inputs; returns the number skipped (missing nodes).

        Args:
            params: Mapping of dotted ``title.input`` paths to values.
            strict: If True, raise if any target node is missing.
        """
        skipped = apply_dotted_params(self._graph, params)
        if strict and skipped:
            raise KeyError(f"{skipped} parameter(s) targeted nodes not present in this graph")
        return skipped

    def connect(self, target: str, source: NodeRef | str) -> None:
        """Connect a dotted node input to another node's output.

        Raises:
            KeyError: If the input or the source node does not exist.
        """
        source_ref = NodeRef(source) if isinstance(source, str) else source
        result = reconnect_input(self._graph, target, source_ref.title)
        if result is None:
            raise KeyError(f"Cannot connect {target!r} to {source_ref.title!r}: input or node missing")
        if source_ref.output != 0:
            # reconnect_input only rewires the node name; set the output index explicitly
            keys = target.split(".")
            if "inputs" not in keys:
                keys.insert(1, "inputs")
            current: Any = self._graph
            for key in keys:
                current = current[key]
            current[1] = source_ref.output

    def add_node(self, title: str, class_type: str, inputs: dict[str, Any]) -> NodeRef:
        """Add a new node to the graph, returning a reference to its first output.

        ``NodeRef`` values in inputs are converted to connection lists.

        Raises:
            ValueError: If a node with that title already exists.
        """
        if title in self._graph:
            raise ValueError(f"Node {title!r} already exists in this graph")
        converted_inputs = {
            name: ([value.title, value.output] if isinstance(value, NodeRef) else value)
            for name, value in inputs.items()
        }
        self._graph[title] = {
            "inputs": converted_inputs,
            "class_type": class_type,
            "_meta": {"title": title},
        }
        return NodeRef(title)

    def to_api_dict(self) -> GraphDict:
        """Return the graph as a plain dict for the execution backend (deep copy)."""
        return copy.deepcopy(self._graph)
