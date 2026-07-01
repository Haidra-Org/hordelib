"""Explicit, tier-ordered pipeline selection over registered definitions.

Each registered :class:`hordelib.pipeline.definition.PipelineDefinition` declares *when* it
applies via its typed :class:`hordelib.pipeline.definition.Selector`; the registry orders
definitions by (tier, order, registration order) and returns the first match. This replaces
both the legacy if/elif tree in ``HordeLib._get_appropriate_pipeline`` and the interim
lambda-predicate/magic-priority registry.

Registration audits every definition against its graph (see :func:`audit_definition`), so a
typo'd binding target or missing output node fails at import time instead of silently doing
nothing at materialization time.

The registry is generic over the payload model and a family-specific selection context, so
each pipeline family (image generation, post-processing, future modalities) gets its own
fully typed registry. The context type is deliberately unconstrained: families that select on
model facts use ``ModelContext``, while simpler families may use a lighter object (or None).
"""

import json
from collections import Counter

from loguru import logger
from pydantic import BaseModel

from hordelib.pipeline.definition import PipelineDefinition, SelectionTier
from hordelib.pipeline.graph import HORDE_OUTPUT_CLASS_TYPES

__all__ = [
    "PipelineRegistry",
    "audit_definition",
]


def audit_definition(definition: PipelineDefinition) -> list[str]:
    """Validate a definition against its graph, returning every violation found.

    Checks:
        - The graph file's raw ``_meta.title`` values are unique (``fix_node_names`` would
          otherwise silently drop colliding nodes).
        - Every binding target's node title exists in the loaded graph. (Input *names* are
          not validated against node schemas; that needs ComfyUI imports and is future work.)
        - At least one output is declared; every output node exists and its post-replacement
          class type is a registered Horde output class for its kind.
        - A non-fallback selector declares at least one criterion.

    Args:
        definition: The pipeline definition to audit.

    Returns:
        list[str]: Human-readable violations; empty when the definition is sound.
    """
    violations: list[str] = []

    raw_nodes = json.loads(definition.graph_file.read_text(encoding="utf-8"))
    raw_titles = [
        node["_meta"]["title"]
        for node in raw_nodes.values()
        if isinstance(node, dict) and isinstance(node.get("_meta"), dict) and "title" in node["_meta"]
    ]
    duplicate_titles = sorted(
        title
        for title, count in Counter(raw_titles).items()
        if count > 1 and title not in definition.known_title_collisions
    )
    if duplicate_titles:
        violations.append(
            f"graph {definition.graph_file.name} has duplicate node titles {duplicate_titles}; "
            "colliding nodes would be silently dropped at load time",
        )

    graph = definition.load_graph()
    node_titles = set(graph.node_titles())

    unbound_targets = [
        binding.target for binding in definition.bindings if binding.target.split(".", 1)[0] not in node_titles
    ]
    if unbound_targets:
        violations.append(f"bindings target nodes absent from the graph: {unbound_targets}")

    if not definition.outputs:
        violations.append("no outputs declared; every pipeline must declare at least one output node")

    for output in definition.outputs:
        if output.node not in node_titles:
            violations.append(f"declared output node {output.node!r} does not exist in the graph")
            continue
        allowed_class_types = HORDE_OUTPUT_CLASS_TYPES.get(output.kind, frozenset())
        node_class_type = graph.node(output.node).get("class_type")
        if node_class_type not in allowed_class_types:
            violations.append(
                f"output node {output.node!r} has class_type {node_class_type!r}, "
                f"not a registered {output.kind} output class ({sorted(allowed_class_types)})",
            )

    fallback_needs_no_criteria = definition.selector.tier is SelectionTier.FALLBACK
    if not fallback_needs_no_criteria and not definition.selector.has_criteria():
        violations.append(
            f"selector in tier {definition.selector.tier.name} declares no criteria and would "
            "shadow every lower-precedence pipeline",
        )

    return violations


class PipelineRegistry[PayloadT: BaseModel, ContextT]:
    """Holds a family's pipeline definitions and selects among them.

    Registration audits each definition against its graph (see :func:`audit_definition`);
    selection walks the definitions in declared precedence order and returns the first whose
    selector matches.
    """

    def __init__(self) -> None:
        self._definitions: list[PipelineDefinition[PayloadT, ContextT]] = []

    def register(self, definition: PipelineDefinition[PayloadT, ContextT]) -> None:
        """Register a definition, auditing it against its graph.

        Raises:
            ValueError: If a definition with the same name is already registered, or the
                definition fails the audit (see :func:`audit_definition`).
        """
        if any(existing.name == definition.name for existing in self._definitions):
            raise ValueError(f"Pipeline {definition.name!r} is already registered")

        violations = audit_definition(definition)
        if violations:
            details = "\n  - ".join(violations)
            raise ValueError(f"Pipeline definition {definition.name!r} failed its audit:\n  - {details}")

        self._definitions.append(definition)
        # Stable sort preserves registration order within a (tier, order) slot
        self._definitions.sort(key=lambda d: (-d.selector.tier, d.selector.order))

    def select(self, payload: PayloadT, context: ContextT) -> PipelineDefinition[PayloadT, ContextT] | None:
        """Return the highest-precedence definition whose selector matches, or None."""
        for definition in self._definitions:
            if definition.selector.matches(payload, context):
                logger.debug(
                    "Pipeline selected: name={}, tier={}, order={}",
                    definition.name,
                    definition.selector.tier.name,
                    definition.selector.order,
                )
                return definition
        return None

    def get(self, name: str) -> PipelineDefinition[PayloadT, ContextT] | None:
        """Return the registered definition with the given name, or None."""
        for definition in self._definitions:
            if definition.name == name:
                return definition
        return None

    def all_definitions(self) -> list[PipelineDefinition[PayloadT, ContextT]]:
        """Return all registered definitions in selection precedence order."""
        return list(self._definitions)
